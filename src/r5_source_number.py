"""R5-5  W4：门限的严格程度会不会影响"源数目自动估计"？

审稿人指出：算法（与多数基线）都假设 N 已知；而门限的严格程度可能直接影响下游
自动定阶（如势函数法）的准确性——因为定阶本身就是在方向密度上找峰。

做法：把势函数法的**峰检测部分**单独拿出来，喂给它由不同掩码产生的方向集，
比较估计出的 N̂ 与真值（同一批实例、同一套峰判据）。四种掩码：
  · collin      仅共线+均衡（**无能量准则**）
  · med0.02     经典  e > 0.02·median(e)
  · med5        经典  e > 5·median(e)（更严格）
  · nf          本文  e > τ·ν̂（标定到虚警率）

峰判据取 baselines.potential_function_A 的同一套逻辑（角度网格 + 局部极大 + 显著性阈值
+ 近峰抑制），只把输入的方向集换成上面的掩码，因此差异只来自掩码。

用法：
    python3 r5_source_number.py --seeds 10 --workers 10 --out ../results/r5_source_number.json
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

M_OBS, F, T = 2, 33, 64
CASES = ([("tf_p02", 4), ("tf_p05", 4), ("tf_p10", 4), ("tf_p20", 4), ("tf_p40", 4),
          ("tf_gauss", 4)]
         + [(k, n) for n in (3, 5, 6) for k in ("tf_p05", "tf_p20")])
N_GRID, MIN_SEP_DEG = 1440, 8.0
PEAK_RATIOS = [0.1, 0.2]


def count_peaks(U: np.ndarray, peak_ratio: float) -> int:
    """M=2 时的显著峰个数（与 baselines.potential_function_A 同判据）。"""
    if U.shape[1] < 2:
        return 0
    C = np.abs(U.T @ U)
    np.fill_diagonal(C, 0.0)
    nn = 1.0 - np.max(C, axis=1)
    h = float(np.clip(np.median(nn[nn > 0]) if np.any(nn > 0) else 0.05, 0.01, 0.3))
    th = np.linspace(0.0, np.pi, N_GRID, endpoint=False)
    V = np.stack([np.cos(th), np.sin(th)], axis=0)
    d = 1.0 - np.abs(V.T @ U)
    rho = np.exp(-(d ** 2) / (2 * h ** 2)).sum(axis=1)
    is_pk = (rho > np.roll(rho, 1)) & (rho >= np.roll(rho, -1))
    cand = np.where(is_pk)[0]
    if cand.size == 0:
        return 1
    cand = cand[rho[cand] > peak_ratio * rho.max()]
    if cand.size == 0:
        return 1
    min_sep = max(int(0.5 * np.deg2rad(MIN_SEP_DEG) / (np.pi / N_GRID)), 3)
    sel: list[int] = []
    for i in cand[np.argsort(rho[cand])[::-1]]:
        if all(min(abs(i - j), N_GRID - abs(i - j)) > min_sep for j in sel):
            sel.append(int(i))
    return len(sel)


def run_case(arg) -> list[dict]:
    key, n_src, seed = arg
    p = D.make_problem(M_OBS, n_src, F, T, key, 20.0, seed=1000 + seed)
    X, A = p["X_tf"], p["A"]
    E = point_energies(X)
    med = float(np.median(E))
    mk_collin = ssp_directions(X, (np.abs(np.sum(X.real * X.imag, axis=0) /
                                         np.maximum(np.linalg.norm(X.real, axis=0) *
                                                    np.linalg.norm(X.imag, axis=0), 1e-15)) > 0.98)
                              & (np.linalg.norm(X.real, axis=0) /
                                 np.maximum(np.linalg.norm(X.real, axis=0) +
                                            np.linalg.norm(X.imag, axis=0), 1e-15) > 0.1)
                              & (np.linalg.norm(X.imag, axis=0) /
                                 np.maximum(np.linalg.norm(X.real, axis=0) +
                                            np.linalg.norm(X.imag, axis=0), 1e-15) > 0.1))
    # 口径统一（重要）：全稿其余各表（Tables 6/15/19）里的 "conventional median t_e=..." 都是
    # **共线+均衡 与 能量判据相与**的结果，所以这里也必须把 base 加到四种掩码上。
    # 第一版只对 med0.02 / med5 施加了能量判据（保留 95% / 19%），而 collin 只有 base（27%），
    # 于是"门限严格程度"这一比较里混进了"有没有共线判据"这个无关变量——行标写着同一个名字，
    # 含义却与别的表不同（实测 95% vs Table 19 的 24.7%）。
    base2d = ((np.abs(np.sum(X.real * X.imag, axis=0) /
                      np.maximum(np.linalg.norm(X.real, axis=0) *
                                 np.linalg.norm(X.imag, axis=0), 1e-15)) > 0.98)
              & (np.linalg.norm(X.real, axis=0) /
                 np.maximum(np.linalg.norm(X.real, axis=0) +
                            np.linalg.norm(X.imag, axis=0), 1e-15) > 0.1)
              & (np.linalg.norm(X.imag, axis=0) /
                 np.maximum(np.linalg.norm(X.real, axis=0) +
                            np.linalg.norm(X.imag, axis=0), 1e-15) > 0.1)).reshape(F, T)
    masks = {"collin": ssp_directions(X, base2d)}
    for tag, thr in (("med0.02", 0.02), ("med5", 5.0)):
        m = base2d & (E > thr * med).reshape(F, T)
        masks[tag] = ssp_directions(X, m)
    m_nf, _ = nfr_mask(X, M_OBS, alpha=1e-4)
    masks["nf"] = ssp_directions(X, m_nf)

    out = []
    for tag, U in masks.items():
        for pr_ in PEAK_RATIOS:
            n_hat = count_peaks(U, pr_)
            # 用估计出的列数聚类并报误差（漏检/多检都惩罚）
            if n_hat >= 1:
                A_hat = kmeans_sphere(U, n_hat, np.random.default_rng(seed))
                err = float(MT.mixing_matrix_angle_error_deg(A, A_hat))
            else:
                err = 90.0
            out.append(dict(case=f"{key}_N{n_src}", key=key, n_true=n_src,
                            method=tag, peak_ratio=pr_, seed=seed,
                            n_est=int(n_hat), abs_err=int(abs(n_hat - n_src)),
                            A=err, n_pts=int(U.shape[1]),
                            keep=float(U.shape[1]) / (F * T)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    args = [(k, n, s) for k, n in CASES for s in range(a.seeds)]
    ctx = mp.get_context("spawn")
    recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for r in ex.map(run_case, args):
            recs += r

    for pr_ in PEAK_RATIOS:
        print(f"\n########## 峰显著性阈值 peak_ratio = {pr_} ##########")
        print(f"{'method':<10}{'精确定阶':>10}{'平均|ΔN|':>10}{'偏低':>7}{'偏高':>7}"
              f"{'角度误差':>10}{'保留率':>9}")
        for tag in ("collin", "med0.02", "med5", "nf"):
            v = [r for r in recs if r["method"] == tag and r["peak_ratio"] == pr_]
            exact = np.mean([r["n_est"] == r["n_true"] for r in v])
            dlt = np.array([r["n_est"] - r["n_true"] for r in v])
            print(f"{tag:<10}{exact*100:>9.1f}%{np.mean(np.abs(dlt)):>10.2f}"
                  f"{np.mean(dlt < 0)*100:>6.1f}%{np.mean(dlt > 0)*100:>6.1f}%"
                  f"{np.mean([r['A'] for r in v]):>10.3f}"
                  f"{100*np.mean([r['keep'] for r in v]):>8.1f}%")
        print(f"  （保守 vs 激进：collin 相对 med5；本文 nf 与两者比较）")

    with open(a.out, "w") as f:
        json.dump(dict(records=recs, cases=CASES, peak_ratios=PEAK_RATIOS,
                       n_seeds=a.seeds, elapsed_s=time.perf_counter() - t0), f)
    print(f"\n记录 {len(recs)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
