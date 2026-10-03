"""R5-3  Q1：固定 α=1e-4 在「噪声底估计已经偏高」的格子上会不会误杀低能量单源点？

审稿人问：在极低 SNR（0 dB）且源较密集（p=0.20）时，信号与噪声能量重叠，σ̂² 已经偏高
（Table 2 的 ratio 1.57）。这种情况下固定的 α=1e-4 是否会把门限抬得过高、误杀大量
低能量的单源点？

做法：在这些格子上扫 α ∈ [1e-6, 1e-1]，同时报
  · σ̂²/σ²（估计偏差，解释门限为什么会被抬高）
  · τ（闭式倍数）
  · 单源召回 / 查准 / F1（**直接回答"误杀"**）
  · 保留率与下游角度误差

用法：
    python3 r5_alpha_cell.py --seeds 10 --workers 10 --out ../results/r5_alpha_cell.json
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
from nfr import estimate_noise_power, nfr_mask, point_energies, threshold_ratio
from r2_ssp_quality import score

CELLS = [("tf_p20", 0.0), ("tf_p20", 10.0), ("tf_p10", 0.0), ("tf_p40", 0.0)]
ALPHAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
M_OBS, N_SRC, F, T = 2, 4, 33, 64


def run_cell(arg) -> list[dict]:
    (key, snr), alpha, seed = arg
    p = D.make_problem(M_OBS, N_SRC, F, T, key, snr, seed=1000 + seed)
    X, A = p["X_tf"], p["A"]
    J = (np.abs(p["S_tf"]) > 1e-12).sum(axis=0)
    E = point_energies(X)
    est = estimate_noise_power(E, M_OBS)
    base = (np.abs(np.sum(X.real * X.imag, axis=0) /
                   np.maximum(np.linalg.norm(X.real, axis=0) * np.linalg.norm(X.imag, axis=0), 1e-15)) > 0.98)

    mk, dg = nfr_mask(X, M_OBS, alpha=alpha, use_self_check=False)
    s = score(mk, J, X, A, base)
    A_hat = kmeans_sphere(ssp_directions(X, mk), N_SRC, np.random.default_rng(seed))
    rec = dict(case=key, snr_db=snr, alpha=alpha, seed=seed,
               s2_ratio=float(est["s2"] / (np.mean(np.abs(p["noise_tf"]) ** 2))),
               tau=float(threshold_ratio(M_OBS, alpha)),
               A=float(MT.mixing_matrix_angle_error_deg(A, A_hat)),
               gate_on=bool(dg["gate_on"]))
    rec.update({k: s[k] for k in ("keep_frac", "precision", "recall", "f1",
                                  "noise_reject", "multi_reject")})
    return [rec]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    args = [(c, al, s) for c in CELLS for al in ALPHAS for s in range(a.seeds)]
    ctx = mp.get_context("spawn")
    recs: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for r in ex.map(run_cell, args):
            recs += r

    for cell in CELLS:
        key, snr = cell
        sub = [r for r in recs if r["case"] == key and r["snr_db"] == snr]
        if not sub:
            continue
        print(f"\n########## {key}, SNR {snr:g} dB ##########")
        print(f"{'α':>8}{'τ':>8}{'σ̂²/σ²':>9}{'保留%':>8}{'单源召回':>10}"
              f"{'查准':>8}{'F1':>7}{'多源拒收':>10}{'角度误差':>10}")
        for al in ALPHAS:
            v = [r for r in sub if r["alpha"] == al]
            g = lambda k: float(np.mean([r[k] for r in v]))
            mark = "  <-- 论文取值" if al == 1e-4 else ""
            print(f"{al:>8.0e}{g('tau'):>8.3f}{g('s2_ratio'):>9.2f}{g('keep_frac')*100:>8.2f}"
                  f"{g('recall'):>10.3f}{g('precision'):>8.3f}{g('f1'):>7.3f}"
                  f"{g('multi_reject'):>10.3f}{g('A'):>10.3f}{mark}")
        # 与 α=1e-4 相比，召回最好的 α
        r4 = np.mean([r["recall"] for r in sub if r["alpha"] == 1e-4])
        best = min(ALPHAS, key=lambda al: -np.mean([r["recall"] for r in sub if r["alpha"] == al]))
        rb = np.mean([r["recall"] for r in sub if r["alpha"] == best])
        a4 = np.mean([r["A"] for r in sub if r["alpha"] == 1e-4])
        ab = np.mean([r["A"] for r in sub if r["alpha"] == best])
        print(f"  召回最高的 α = {best:.0e}：召回 {rb:.3f}（1e-4 为 {r4:.3f}），"
              f"角度误差 {ab:.3f}（1e-4 为 {a4:.3f}）")

    with open(a.out, "w") as f:
        json.dump(dict(records=recs, cells=CELLS, alphas=ALPHAS, n_seeds=a.seeds,
                       elapsed_s=time.perf_counter() - t0), f)
    print(f"\n记录 {len(recs)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
