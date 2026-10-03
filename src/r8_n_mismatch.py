"""R8-1  源数目失配的**受控**敏感性（CSSP 投稿前意见 Major 7 的严格版）。

现有 Table 39（`r5_source_number.py`）测的是「门限的严格程度会不会改变**自动定阶**的结果」——
即把势函数峰检测喂给不同掩码，看它自己估出的 N̂。那不是"给错 N 会怎样"。

本条补的是审稿人真正问的那一维：**故意把聚类列数设成 N̂ = N + d（d = −2…+2），
在同一个方向上量角度误差**。因为：
  · 漏一列在匹配度量里代价 90°，多一列只花"最近对应方向的夹角"，两种错不对称；
  · τ 的标定式 Q_{1−α}(χ²_{2M})/(2M) 只依赖 M 与 α，不依赖 N，
    但 **p* = 1 − 2^{−1/N} 依赖 N**，所以 N 是理论里的关键参数，必须量出它对算法的实际影响。

三种掩码与 `r5_source_number.py` 完全一致（同一套 base2d + 能量判据），
所以掩码之间只差能量准则，跨表可对齐：
  collin    仅共线+均衡（无能量准则）
  med0.02   经典 e > 0.02·median(e)
  nf        本文 e > τ·ν̂（标定到虚警率 α=1e-4）

用法：
    python3 r8_n_mismatch.py --seeds 10 --workers 8 --out ../results/r8_n_mismatch.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from nfr import nfr_mask, point_energies

M_OBS, F, T, SNR_DB = 2, 33, 64, 20.0
CASES = [(k, n) for n in (3, 4, 5, 6) for k in ("tf_p05", "tf_p20")]
DELTAS = (-2, -1, 0, 1, 2)


def masks_of(X: np.ndarray) -> dict[str, np.ndarray]:
    """与 r5_source_number.py / r2_noise_family.py 同一套判据。"""
    Xr, Xi = X.real, X.imag
    eps = 1e-15
    nr, ni = np.linalg.norm(Xr, axis=0), np.linalg.norm(Xi, axis=0)
    base = ((np.abs(np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)) > 0.98)
            & (nr / np.maximum(nr + ni, eps) > 0.1)
            & (ni / np.maximum(nr + ni, eps) > 0.1))
    E = point_energies(X)
    med = float(np.median(E))
    out = {"collin": ssp_directions(X, base)}
    out["med0.02"] = ssp_directions(X, base & (E > 0.02 * med).reshape(F, T))
    m_nf, _ = nfr_mask(X, M_OBS, alpha=1e-4)
    out["nf"] = ssp_directions(X, m_nf)
    return out


def run_case(arg) -> list[dict]:
    key, n_src, seed = arg
    p = D.make_problem(M_OBS, n_src, F, T, key, SNR_DB, seed=1000 + seed)
    X, A = p["X_tf"], p["A"]
    ms = masks_of(X)
    out = []
    for tag, U in ms.items():
        for d in DELTAS:
            k = n_src + d
            if k < 1 or U.shape[1] < 1:
                err = 90.0
            else:
                A_hat = kmeans_sphere(U, k, np.random.default_rng(seed))
                err = float(MT.mixing_matrix_angle_error_deg(A, A_hat))
            out.append(dict(case=f"{key}_N{n_src}", key=key, n_true=n_src, method=tag,
                            seed=seed, delta=d, n_clusters=int(max(k, 0)),
                            A=err, n_pts=int(U.shape[1]),
                            keep=float(U.shape[1]) / (F * T)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    args = [(k, n, s) for k, n in CASES for s in range(a.seeds)]
    ctx = mp.get_context("spawn")
    recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for r in ex.map(run_case, args):
            recs += r

    print(f"{'mask':<10}", *[f"{'d=' + str(d):>9}" for d in DELTAS])
    for tag in ("collin", "med0.02", "nf"):
        row = [np.mean([r["A"] for r in recs if r["method"] == tag and r["delta"] == d])
               for d in DELTAS]
        print(f"{tag:<10}", *[f"{v:>9.3f}" for v in row])
    print(f"\n保留率  collin {100*np.mean([r['keep'] for r in recs if r['method']=='collin']):.1f}%"
          f"  med0.02 {100*np.mean([r['keep'] for r in recs if r['method']=='med0.02']):.1f}%"
          f"  nf {100*np.mean([r['keep'] for r in recs if r['method']=='nf']):.1f}%")
    # 低 N 与高 N 分开看（N 是理论里的关键参数）
    for n_ in (3, 6):
        row = [np.mean([r["A"] for r in recs if r["method"] == "nf"
                        and r["delta"] == d and r["n_true"] == n_]) for d in DELTAS]
        print(f"  nf, N={n_}", *[f"{v:>8.3f}" for v in row])

    with open(a.out, "w") as f:
        json.dump(dict(records=recs, cases=CASES, deltas=list(DELTAS),
                       n_seeds=a.seeds, snr_db=SNR_DB,
                       elapsed_s=time.perf_counter() - t0), f)
    print(f"\n记录 {len(recs)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
