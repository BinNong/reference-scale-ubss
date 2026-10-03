#!/usr/bin/env python3
"""R4 / Minor 4：重建 Table 2（盲噪声底估计量的性质），并说明它与 Table 22 的口径差异。

背景：Table 2 的数值此前**没有任何脚本产出**（`noisefloor_check.py`/`blind_noisefloor_check.py`
是早期版本、只打 stdout 且用的是另一套估计量 E1/E2）。本脚本按 Table 2 的原始设置
（$N=4$、$M=2$、SNR 20 dB、$q_{\max}=0.02$）重算，并做两件事：

  1. 输出机器可读记录 `results/r4_table2.json`；
  2. 分别在 5 个种子（Table 2 的口径）与 8 个种子（Table 22 的口径）下算 dense 行，
     以判定审稿人指出的 77.1 vs 80.6–87.5 是否只是种子数之差。

用法:
  python3 r4_table2.py --out ../results/r4_table2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
from nfr import estimate_noise_power, point_energies

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]


def one(key: str, seeds: int) -> dict:
    rows = []
    for s in range(seeds):
        p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
        s2_true = float(np.mean(np.abs(p["noise_tf"]) ** 2))
        est = estimate_noise_power(point_energies(p["X_tf"]), 2)
        rows.append((s2_true, float(est["s2"]), float(est["s2"] / s2_true),
                     float(est["spread"])))
    a = np.array(rows)
    return dict(case=key, n_seeds=seeds,
                s2_true=float(np.mean(a[:, 0])), s2_hat=float(np.mean(a[:, 1])),
                ratio=float(np.mean(a[:, 2])), ratio_sd=float(np.std(a[:, 2], ddof=1)),
                spread=float(np.mean(a[:, 3])),
                ratio_per_seed=[float(x) for x in a[:, 2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/r4_table2.json")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    print(f"=== Table 2 重建（{a.seeds} seeds，N=4, M=2, SNR 20 dB, q_max=0.02）===")
    print(f"{'p':<9}{'sigma2_true':>13}{'sigma2_hat':>13}{'ratio':>9}{'spread':>9}")
    recs = []
    for key in LADDER:
        r = one(key, a.seeds)
        recs.append(r)
        lab = key.replace("tf_", "")
        print(f"{lab:<9}{r['s2_true']:>13.4f}{r['s2_hat']:>13.4f}"
              f"{r['ratio']:>9.2f}{r['spread']:>9.2f}")

    print("\n=== 种子数敏感性：dense 行在 5 与 8 个种子下 ===")
    d5, d8 = one("tf_gauss", 5), one("tf_gauss", 8)
    print(f"  5 seeds: ratio = {d5['ratio']:.1f}  (per-seed {['%.1f'%x for x in d5['ratio_per_seed']]})")
    print(f"  8 seeds: ratio = {d8['ratio']:.1f}  (per-seed {['%.1f'%x for x in d8['ratio_per_seed']]})")
    print(f"  5 seeds 的最小/最大: {min(d5['ratio_per_seed']):.1f} / {max(d5['ratio_per_seed']):.1f}")
    print(f"  8 seeds 的最小/最大: {min(d8['ratio_per_seed']):.1f} / {max(d8['ratio_per_seed']):.1f}")

    json.dump(dict(n_sources=4, m_obs=2, snr_db=20.0, q_max=0.02,
                   seeds=a.seeds, rows=recs,
                   dense_5=d5, dense_8=d8), open(a.out, "w"), indent=1)
    print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
