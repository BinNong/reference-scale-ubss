"""参考尺度的条件不变性：系数是否需要随 (M, 时频点数) 重新标定？

这是"哪一方更可迁移"的决定性判据：
  · 噪声底标定 τ = Q_{1−α}(χ²_{2M})/(2M) —— **构造上**只依赖 (M, α)；
  · 最大值参考 c·max(e) —— max 是极值统计量，**随样本数 F·T 漂移**，
    且能量分布随 M 变化，故 c 需要随之重标定。

本脚本对每条参考尺度族，在 (M, N, 稀疏度, 时频点数) 网格上求最优系数，
考察最优系数的漂移幅度；同时给出固定 α=1e-4 的 NF-SSP 在这些格子上的表现。

用法: ../.venv/bin/python scale_invariance.py --out ../results/scale_invariance.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions, ssp_mask_from_complex
from nfr import nfr_mask, threshold_ratio

MED = [0.02, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
MAXC = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6]


def _err(p, mask, seed=0):
    if mask.sum() < p["n"]:
        return float("nan")
    A = kmeans_sphere(ssp_directions(p["X_tf"], mask), p["n"],
                      np.random.default_rng(seed))
    return MT.mixing_matrix_angle_error_deg(p["A"], A)


def _m_med(p, te):
    return ssp_mask_from_complex(p["X_tf"], thr_cos=0.98, thr_energy_ratio=te)


def _m_max(p, c):
    base = ssp_mask_from_complex(p["X_tf"], thr_cos=0.98, thr_energy_ratio=0.0)
    e = np.sum(np.abs(p["X_tf"]) ** 2, axis=0)
    return base & (e >= c * e.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="../results/scale_invariance.json")
    a = ap.parse_args()

    rows = []
    print("=" * 110)
    print("最优系数的条件漂移（合成网格，3 seeds 均值）")
    print("=" * 110)
    h = (f"{'M':>2s}{'N':>3s}{'p':>6s}{'F×T':>9s}{'点数':>8s} │"
         f"{'最优点数':>9s}{'te*':>8s}{'分段最优':>9s} │{'c*':>7s}{'max参考最优':>12s} │"
         f"{'NF(α=1e-4)':>11s}{'NF/最优':>8s}")
    print(h); print("-" * len(h))

    for M in [2, 3, 4]:
        for N in [4, 6]:
            for key, pv in [("tf_p05", 0.05), ("tf_p20", 0.20), ("tf_p40", 0.40)]:
                for F, T in [(33, 64), (65, 128), (129, 256)]:
                    probs = [D.make_problem(M, N, F, T, key, 20.0, seed=1000 + s)
                             for s in range(a.seeds)]
                    med = {t: np.nanmean([_err(p, _m_med(p, t), s)
                                          for s, p in enumerate(probs)]) for t in MED}
                    mx = {c: np.nanmean([_err(p, _m_max(p, c), s)
                                         for s, p in enumerate(probs)]) for c in MAXC}
                    nf = np.nanmean([_err(p, nfr_mask(p["X_tf"], M, alpha=1e-4)[0], s)
                                     for s, p in enumerate(probs)])
                    te_s = min(med, key=lambda k: med[k])
                    c_s = min(mx, key=lambda k: mx[k])
                    best = min(min(med.values()), min(mx.values()))
                    n_pts = probs[0]["F"] * probs[0]["T"]
                    rows.append(dict(M=M, N=N, p=pv, F=F, T=T, n_pts=n_pts,
                                     te_star=te_s, err_med=med[te_s],
                                     c_star=c_s, err_max=mx[c_s], nf=nf, best=best,
                                     med_tbl={str(k): float(v) for k, v in med.items()},
                                     max_tbl={str(k): float(v) for k, v in mx.items()}))
                    print(f"{M:>2d}{N:>3d}{pv:>6.2f}{f'{F}x{T}':>9s}{n_pts:>8d} │"
                          f"{te_s:>9g}{med[te_s]:>8.2f} │ {c_s:>6g}{mx[c_s]:>12.2f} │"
                          f"{nf:>11.2f}{nf/best:>8.2f}")

    print("\n" + "=" * 110)
    print("漂移与安全性汇总")
    print("=" * 110)
    for name, key in [("中位数参考 te*", "te_star"), ("最大值参考 c*", "c_star")]:
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"  {name}: 取值 {sorted(set(v))}   最大/最小 = {v.max()/v.min():.1f}×")

    def safety(tbl_key, grid):
        """固定系数在全部格子上的最差比值（相对每格最优）与 90 分位。"""
        out = {}
        for k in grid:
            ratios = []
            for r in rows:
                e = r[tbl_key][str(k)]
                if np.isfinite(e) and r["best"] > 0:
                    ratios.append(e / r["best"])
            ratios = np.array(ratios)
            out[k] = (float(np.median(ratios)), float(np.max(ratios)),
                      int(np.sum(ratios > 10)))
        return out

    print(f"\n  {'固定系数':>10s}{'比值中位':>10s}{'最差比值':>10s}{'灾难格数(>10×)':>16s}")
    print("  --- 中位数参考 ---")
    for k, (md, mxr, nbad) in safety("med_tbl", MED).items():
        print(f"  {k:>10g}{md:>10.2f}{mxr:>10.1f}{nbad:>16d}")
    print("  --- 最大值参考 ---")
    for k, (md, mxr, nbad) in safety("max_tbl", MAXC).items():
        print(f"  {k:>10g}{md:>10.2f}{mxr:>10.1f}{nbad:>16d}")
    nf_ratio = np.array([r["nf"] / r["best"] for r in rows if r["best"] > 0])
    print(f"\n  NF-SSP（固定 α=1e-4）: 比值中位 {np.median(nf_ratio):.2f}  "
          f"最差 {nf_ratio.max():.1f}  灾难格数 {int(np.sum(nf_ratio > 10))}")
    print("  （比值 = 该配置下的误差 / 该配置下两种参考族合计能达到的最小误差）")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"med_grid": MED, "max_grid": MAXC, "rows": rows}, f, indent=2)
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
