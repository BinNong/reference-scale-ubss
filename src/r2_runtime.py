"""R2-5  运行时间分项拆解（审稿意见第十六节）。

把 NF 门限拆成可以单独计时的阶段，证明「噪声底标定几乎免费」，
而不是只报告总运行时间。同时给出合成（2112 点）与真实语音（win1024）
两个尺度，因为论文 §9.4 的「门限 0.2 s vs 共享 16.7 s」是真实尺度上的数。

阶段：
  E1 逐点能量            ‖x(f,t)‖²
  E2 噪声底估计          次序统计量 + 卡方分位数反演 + 中位数合并
  E3 门限与保留自检       τ·ν̂ 比较 + 计数判定
  E4 聚类                球面 k-means
  E5 恢复                去偏 ℓ1
  合计

用法：
    python3 r2_runtime.py --seeds 20 --out ../results/r2_runtime.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import data as D
from nfr import estimate_noise_power, point_energies, threshold_ratio, nfr_mask
from baselines import kmeans_sphere, ssp_directions, l1_recover

M_OBS, N_SRC, F, T = 2, 4, 33, 64


def time_once(X_tf, X_all, n_true, m_obs, reps=1):
    """对一个问题计时各阶段，返回各阶段秒数（reps 次取中位数）。"""
    acc = {k: [] for k in ("E1_energy", "E2_noise_floor", "E3_gate_and_check",
                           "E4_clustering", "E5_recovery", "total")}
    for r in range(reps):
        t0 = time.perf_counter()
        E = point_energies(X_tf)
        t1 = time.perf_counter()
        est = estimate_noise_power(E, m_obs)
        t2 = time.perf_counter()
        tau = threshold_ratio(m_obs, 1e-4)
        nu_hat = m_obs * est["s2"]
        e2 = np.sum(np.abs(X_tf) ** 2, axis=0)
        Xr, Xi = X_tf.real, X_tf.imag
        nr = np.linalg.norm(Xr, axis=0)
        ni = np.linalg.norm(Xi, axis=0)
        eps = 1e-15
        cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
        d = np.maximum(nr + ni, eps)
        base = (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)
        n_keep = int((base & (e2 > tau * nu_hat)).sum())
        need = max(10, int(np.ceil(0.035 * max(int(base.sum()), 1))))
        mask = (base & (e2 > tau * nu_hat)) if (n_keep >= need) else base
        t3 = time.perf_counter()
        U = ssp_directions(X_tf, mask)
        A_hat = kmeans_sphere(U, n_true, np.random.default_rng(r))
        t4 = time.perf_counter()
        l1_recover(A_hat, X_all)
        t5 = time.perf_counter()
        acc["E1_energy"].append(t1 - t0)
        acc["E2_noise_floor"].append(t2 - t1)
        acc["E3_gate_and_check"].append(t3 - t2)
        acc["E4_clustering"].append(t4 - t3)
        acc["E5_recovery"].append(t5 - t4)
        acc["total"].append(t5 - t0)
    return {k: float(np.median(v)) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--root", default="../data/ls_corpus")
    ap.add_argument("--real-seeds", type=int, default=3)
    ap.add_argument("--out", default="../results/r2_runtime.json")
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args()

    rows, out = [], {}
    print(f"\n=== 合成（F={F}, T={T}, {F*T} 个 TF 点，{a.seeds} seeds，取中位数）===")
    keys = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40"]
    acc_sum = None
    for k in keys:
        vals = []
        for s in range(a.seeds):
            p = D.make_problem(M_OBS, N_SRC, F, T, k, 20.0, seed=1000 + s)
            vals.append(time_once(p["X_tf"], p["X_all"], N_SRC, M_OBS))
        agg = {kk: float(np.median([v[kk] for v in vals])) for kk in vals[0]}
        rows.append(dict(domain="synth", case=k, **agg))
        acc_sum = agg if acc_sum is None else acc_sum
    # 合成汇总（对 5 档取中位数）
    agg = {kk: float(np.median([r[kk] for r in rows])) for kk in rows[0] if kk not in ("domain", "case")}
    hdr = (f"{'stage':<24}{'per call (s)':>14}{'share':>9}")
    print(hdr)
    print("-" * len(hdr))
    for kk in ("E1_energy", "E2_noise_floor", "E3_gate_and_check",
               "E4_clustering", "E5_recovery", "total"):
        print(f"{kk:<24}{agg[kk]:>14.5f}{agg[kk]/agg['total']*100:>8.2f}%")
    gate = agg["E1_energy"] + agg["E2_noise_floor"] + agg["E3_gate_and_check"]
    print(f"{'gate subtotal':<24}{gate:>14.5f}{gate/agg['total']*100:>8.2f}%")
    out["synth"] = agg

    if not a.no_real:
        try:
            import realdata as RD
            corpus = RD.scan_corpus(a.root)
            pool = RD.load_pool(corpus, n_files=40, dur_s=3.0, seed=0, min_speakers=4)
            vals = []
            for s in range(a.real_seeds):
                prob = RD.make_real_problem(pool, m_obs=M_OBS, n_sources=4, win=1024,
                                            dur_s=3.0, seed=1000 + s, snr_db=20.0,
                                            hop=512)
                vals.append(time_once(prob["X_tf"], prob["X_all"], prob["n"], M_OBS))
            rg = {kk: float(np.median([v[kk] for v in vals])) for kk in vals[0]}
            npts = int(np.prod(prob["X_tf"].shape[1:]))
            print(f"\n=== 真实语音（win=1024，{npts} 个 TF 点，{a.real_seeds} seeds）===")
            print(hdr)
            print("-" * len(hdr))
            for kk in ("E1_energy", "E2_noise_floor", "E3_gate_and_check",
                       "E4_clustering", "E5_recovery", "total"):
                print(f"{kk:<24}{rg[kk]:>14.5f}{rg[kk]/rg['total']*100:>8.2f}%")
            g2 = rg["E1_energy"] + rg["E2_noise_floor"] + rg["E3_gate_and_check"]
            print(f"{'gate subtotal':<24}{g2:>14.5f}{g2/rg['total']*100:>8.2f}%")
            out["real_win1024"] = dict(**rg, n_tf_points=npts)
        except Exception as e:                                    # noqa: BLE001
            print(f"\n[跳过真实尺度计时] {type(e).__name__}: {e}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": vars(a), "per_case": rows, "summary": out},
                  open(a.out, "w"), indent=1)
        print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
