"""R5-4  W3：逐频带噪声底——把「自检触发即丢弃门限」换成「换一个更窄的参考尺度」。

审稿人指出：中等稀疏度 + 非理想噪声是常见的现实场景，此时下尾估计会严重偏高
（Table 2 的 ratio 2.87、真实 babble 场景低估 66×），而自检只能"识别"失效、不能修。
他建议在自检触发时引入一种自适应降级策略（例如改进后的鲁棒统计量），而不是完全丢弃门限。

R4 已经否掉了"换成 MAD 类稳健统计量"这条路：babble 的失效不是**污染**而是**非平稳**——
噪声功率沿频率变化 5.8 倍（高斯为 0.10），深下尾读的是安静频点的能量，于是把底低估 66×。

因此本脚本检验一个与失效机理**匹配**的降级方案：**逐频带**估计噪声底，让参考尺度
跟随频率变化。要点：

  · 每个频带内点数减少 → 分位数估计方差上升（代价是可测的）；
  · 频带内的噪声若近似平稳，则低估被消除（收益也是可测的）。

口径：带宽 B ∈ {1, 4, 16, 64}（1 = 现用的全局底）。对每个频带 b，
用该带内的逐点能量做下尾估计得 ν̂_b，掩码为 e(f,t) > τ·ν̂(f)；其余判据不变。

用法：
    python3 r5_band_floor.py --seeds 4 --workers 8 --out ../results/r5_band_floor.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import metrics as MT
import realdata as RD
from baselines import kmeans_sphere, ssp_directions, ssp_mask_from_complex
from nfr import estimate_noise_power, point_energies, threshold_ratio

ALPHA = 1e-4
M_OBS, DUR = 2, 3.0
BANDS = [1, 4, 16, 64]
CASES = [
    ("babble", dict(win=1024, n_sources=4, snr_db=20.0, noise="babble")),
    ("babble0", dict(win=1024, n_sources=4, snr_db=0.0, noise="babble")),
    ("gauss", dict(win=1024, n_sources=4, snr_db=20.0, noise="gauss")),
    ("clean", dict(win=1024, n_sources=4, snr_db=None, noise=None)),
]

_CORPUS = None


def corpus(root: str):
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = RD.scan_corpus(root)
    return _CORPUS


def band_bounds(F: int, nb: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, F, nb + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(nb)]


def run_case(arg) -> list[dict]:
    tag, kw, nb, seed, root = arg
    pool = RD.load_pool(corpus(root), n_files=40, dur_s=DUR, seed=seed,
                        min_speakers=kw["n_sources"])
    p = RD.make_real_problem(pool, m_obs=M_OBS, dur_s=DUR, seed=1000 + seed,
                             hop=kw["win"] // 2, **kw)
    X, A = p["X_tf"], p["A"]
    noise = p["noise_tf"]
    F = X.shape[1]
    s2_true = (float(np.mean(np.abs(noise) ** 2)) if noise is not None else 0.0)
    nu_true = M_OBS * s2_true
    tau = threshold_ratio(M_OBS, ALPHA)
    base = ssp_mask_from_complex(X, thr_cos=0.98, thr_part_ratio=0.1,
                                 thr_energy_ratio=0.0)
    out = []
    n_base = int(base.sum())
    for nb_ in BANDS:
        nu_hat = np.full(F, np.nan)
        for (f0, f1) in band_bounds(F, nb_):
            Eb = point_energies(X[:, f0:f1, :])
            if Eb.size < 40:
                nu_hat[f0:f1] = np.nan
                continue
            s2b = float(estimate_noise_power(Eb, M_OBS)["s2"])
            nu_hat[f0:f1] = M_OBS * s2b
        nu_hat = np.nan_to_num(nu_hat, nan=float(np.nanmedian(nu_hat)))
        # 逐点门限：nu_hat 是逐**频点**的底，展平到 (f,t) 顺序后必须 reshape 回 (F,T)。
        # 与 §5.4 的完整判据同构：能量判据与共线+均衡判据相与，并施加保留自检。
        Ener = (point_energies(X) > (tau * np.repeat(nu_hat, X.shape[2]))).reshape(F, X.shape[2])
        mk = base & Ener
        if int(mk.sum()) < max(10, int(0.035 * n_base)):
            mk = base                       # 自检触发 -> 退回不过能量门限
        A_hat = kmeans_sphere(ssp_directions(X, mk), kw["n_sources"],
                              np.random.default_rng(seed))
        est_ratio = (float(np.mean(nu_hat)) / nu_true) if nu_true > 0 else float("nan")
        out.append(dict(case=tag, bands=nb_, seed=seed,
                        nu_ratio=est_ratio,
                        A=float(MT.mixing_matrix_angle_error_deg(A, A_hat)),
                        keep=float(mk.mean()),
                        n_points=int(X.shape[1] * X.shape[2])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    args = [(tag, kw, nb, s, a.corpus) for tag, kw in CASES
            for nb in BANDS for s in range(a.seeds)]
    ctx = mp.get_context("spawn")
    recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for r in ex.map(run_case, args):
            recs += r

    print(f"{'case':<10}{'bands':>7}{'ν̂/ν':>10}{'保留%':>9}{'角度误差':>10}")
    for tag, _ in CASES:
        for nb in BANDS:
            v = [r for r in recs if r["case"] == tag and r["bands"] == nb]
            if not v:
                continue
            print(f"{tag:<10}{nb:>7}{np.nanmean([r['nu_ratio'] for r in v]):>10.3f}"
                  f"{np.mean([r['keep'] for r in v])*100:>9.2f}"
                  f"{np.mean([r['A'] for r in v]):>10.3f}")

    with open(a.out, "w") as f:
        json.dump(dict(records=recs, bands=BANDS, n_seeds=a.seeds,
                       elapsed_s=time.perf_counter() - t0), f)
    print(f"\n记录 {len(recs)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
