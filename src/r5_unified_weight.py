"""R5-6  Q2：虚警率校准（决定支撑集）与能量软加权（决定支撑集内的影响）能否统一？

审稿人问：§9.5 里"用能量加权方向"比硬门限在真实语音上更好，这实际上把"分类/过滤"
变成了"软注意力/加权"；是否可以把 NF-SSP 的虚警率校准与这种软加权机制结合成一个统一框架？

统一方式其实很直接：**同一批方向被赋予两类作用**
  · 门限 决定**谁进入**（支撑集），由虚警率 α 标定 ⇒ 控制"噪声点的数量"；
  · 权重 决定**进入者各自的影响力**，取 ‖x‖²/ν̂（能量可靠度）⇒ 控制"方向的可靠度"。
两者在 §9.5 被证明控制的是不同的量（前者对应 P(噪声被接纳)，后者对应 P(方向可靠)），
因此组合起来原则上可以同时取两者之长。

本脚本在真实语音的 15 个配置上比较四种方案（同一批问题实例、同一套球面 k-means）：
  · nf        硬门限（门限 + 无权聚类）—— 论文方法
  · nf+w      门限选定支撑集，**再**用 ‖x‖²/ν̂ 加权聚类
  · base+w    只做共线+均衡，用 ‖x‖²/ν̂ 加权（等价于第三轮的 NF-W，无门限）
  · nf+w4     门限 + 权重 ‖x‖²/ν̂·|cos|⁴

用法：
    python3 r5_unified_weight.py --seeds 6 --workers 8 --out ../results/r5_unified_weight.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

import metrics as MT
import realdata as RD
from baselines import kmeans_sphere, ssp_directions, ssp_mask_from_complex
from nfr import estimate_noise_power, nfr_mask, point_energies
from weighted_variant import kmeans_sphere_weighted

ALPHA = 1e-4
M_OBS, DUR = 2, 3.0
CONFIGS = [
    ("R1_res", "w256", dict(win=256, n_sources=4, snr_db=20.0)),
    ("R1_res", "w512", dict(win=512, n_sources=4, snr_db=20.0)),
    ("R1_res", "w1024", dict(win=1024, n_sources=4, snr_db=20.0)),
    ("R1_res", "w2048", dict(win=2048, n_sources=4, snr_db=20.0)),
    ("R2_n", "N3", dict(win=1024, n_sources=3, snr_db=20.0)),
    ("R2_n", "N5", dict(win=1024, n_sources=5, snr_db=20.0)),
    ("R2_n", "N6", dict(win=1024, n_sources=6, snr_db=20.0)),
    ("R3_snr", "snr00", dict(win=1024, n_sources=4, snr_db=0.0)),
    ("R3_snr", "snr10", dict(win=1024, n_sources=4, snr_db=10.0)),
    ("R3_snr", "snr30", dict(win=1024, n_sources=4, snr_db=30.0)),
    ("R3_snr", "snr40", dict(win=1024, n_sources=4, snr_db=40.0)),
    ("R4_dense", "d1", dict(win=1024, n_sources=4, snr_db=20.0, n_dense=1)),
    ("R4_dense", "d2", dict(win=1024, n_sources=4, snr_db=20.0, n_dense=2)),
    ("R5_noise", "babble", dict(win=1024, n_sources=4, snr_db=20.0, noise="babble")),
    ("R5_noise", "clean", dict(win=1024, n_sources=4, snr_db=None, noise=None)),
]

_CORPUS = None


def corpus(root: str):
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = RD.scan_corpus(root)
    return _CORPUS


def run_case(arg) -> list[dict]:
    fam, tag, kw, seed, root = arg
    need = kw["n_sources"] - kw.get("n_dense", 0)
    pool = RD.load_pool(corpus(root), n_files=40, dur_s=DUR, seed=seed,
                        min_speakers=need)
    p = RD.make_real_problem(pool, m_obs=M_OBS, dur_s=DUR, seed=1000 + seed,
                             hop=kw["win"] // 2, **kw)
    X, A = p["X_tf"], p["A"]
    n = kw["n_sources"]
    E = point_energies(X)

    m_base = ssp_mask_from_complex(X, thr_cos=0.98, thr_part_ratio=0.1,
                                  thr_energy_ratio=0.0)
    m_nf, dg = nfr_mask(X, M_OBS, alpha=ALPHA)
    nu_hat = M_OBS * float(estimate_noise_power(E, M_OBS)["s2"])

    Xr, Xi = X.real, X.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    cos4 = (np.abs(np.sum(Xr * Xi, axis=0) /
                   np.maximum(nr * ni, 1e-15)) ** 4).reshape(-1)

    # 与真实实验归档完全同构的两种"惯例"掩码（experiments_real 的 mask_max / mask_median）
    m_max = m_base & (E.reshape(X.shape[1:]) >= 0.05 * E.max())
    m_med = ssp_mask_from_complex(X, thr_cos=0.98, thr_part_ratio=0.1,
                                  thr_energy_ratio=0.02)

    recs = []
    # 2×N 设计：{能量准则} × {是否用同一套能量权}。这样"加权带来的增益"与
    # "参考尺度带来的增益"可以分离——审稿人问的正是两者能否统一。
    plans = {
        "nf": (m_nf, None),
        "nf+w": (m_nf, "energy"),
        "nf+w4": (m_nf, "energy_cos4"),
        "max": (m_max, None),
        "max+w4": (m_max, "energy_cos4"),
        "med0.02": (m_med, None),
        "med0.02+w4": (m_med, "energy_cos4"),
        "base+w": (m_base, "energy"),
    }
    for name, (mk, wkind) in plans.items():
        U = ssp_directions(X, mk)
        if wkind is None:
            A_hat = kmeans_sphere(U, n, np.random.default_rng(seed))
        else:
            sel = np.flatnonzero(mk.reshape(-1))
            w = E[sel] / max(nu_hat, 1e-30)
            if wkind == "energy_cos4":
                w = w * cos4[sel]
            A_hat = kmeans_sphere_weighted(U, n, np.random.default_rng(seed), w)
            if A_hat is None:
                A_hat = kmeans_sphere(U, n, np.random.default_rng(seed))
        recs.append(dict(case=f"{fam}:{tag}", method=name, seed=seed, n=n,
                         A=float(MT.mixing_matrix_angle_error_deg(A, A_hat)),
                         keep=float(mk.mean()),
                         gate_on=bool(dg["gate_on"])))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    args = [(fam, tag, kw, s, a.corpus) for fam, tag, kw in CONFIGS
            for s in range(a.seeds)]
    ctx = mp.get_context("spawn")
    recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for r in ex.map(run_case, args):
            recs += r

    cases = sorted({r["case"] for r in recs})
    methods = ["med0.02", "med0.02+w4", "max", "max+w4", "nf", "nf+w", "nf+w4", "base+w"]
    print(f"\n{'config':<18}" + "".join(f"{m:>10}" for m in methods))
    for c in cases:
        row = f"{c:<18}"
        for m in methods:
            v = [r["A"] for r in recs if r["case"] == c and r["method"] == m]
            row += f"{np.mean(v):>10.3f}"
        print(row)
    print(f"{'MEAN':<18}" + "".join(
        f"{np.mean([r['A'] for r in recs if r['method'] == m]):>10.3f}" for m in methods))

    print("\n=== 配对检验（同配置同种子，n=配置数×种子数）===")
    for ref, cand in (("nf", "nf+w"), ("nf", "nf+w4"), ("max", "nf+w4"),
                      ("max", "max+w4"), ("max", "nf"), ("nf", "base+w"),
                      ("nf+w", "nf+w4"), ("med0.02", "med0.02+w4")):
        A_ = {(r["case"], r["seed"]): r["A"] for r in recs if r["method"] == ref}
        B_ = {(r["case"], r["seed"]): r["A"] for r in recs if r["method"] == cand}
        ks = sorted(set(A_) & set(B_))
        av = np.array([A_[k] for k in ks])
        bv = np.array([B_[k] for k in ks])
        p = stats.ttest_rel(av, bv).pvalue
        print(f"  {ref:<8} → {cand:<8} n={len(ks):<4} {av.mean():.3f} → {bv.mean():.3f}"
              f"  差 {bv.mean()-av.mean():+.3f}  p={p:.2e}")
    for ref, cand in (("nf", "nf+w"), ("max", "nf+w4")):
        n_win = sum(1 for c in cases
                    if np.mean([r["A"] for r in recs if r["case"] == c and r["method"] == cand])
                    < np.mean([r["A"] for r in recs if r["case"] == c and r["method"] == ref]))
        print(f"  {cand} 优于 {ref} 的配置数：{n_win}/{len(cases)}")

    with open(a.out, "w") as f:
        json.dump(dict(records=recs, n_seeds=a.seeds,
                       elapsed_s=time.perf_counter() - t0), f)
    print(f"\n记录 {len(recs)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
