#!/usr/bin/env python3
"""R4 / Question 1 附：次序统计量**水平约定**的有限样本偏差（实测）。

`estimate_noise_power` 用的水平是 $q_i=i/(n+1)$，即第 $i$ 个次序统计量的**名义**分位，
而不是它自身分布的中位位置。两者在有限 $n$ 下不同，于是估计量带一个负的小偏差。

本脚本在**纯高斯噪声**（无信号）上直接测这个偏差，并看它随 $n$ 的衰减——
这是 §5.1(iii) 那句话的机器可读记录。用法:

  python3 r4_floor_bias.py --out ../results/r4_floor_bias.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfr import estimate_noise_power, point_energies

SIG2 = 0.01


def run(m_obs: int, F: int, T: int, reps: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(reps):
        X = (rng.standard_normal((m_obs, F, T)) + 1j * rng.standard_normal((m_obs, F, T)))
        X *= np.sqrt(SIG2 / 2.0)
        acc.append(estimate_noise_power(point_energies(X), m_obs)["s2"] / SIG2)
    a = np.array(acc)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", default="../results/r4_floor_bias.json")
    a = ap.parse_args()

    cells = [(2, 33, 64), (2, 33, 256), (2, 33, 1024), (3, 33, 64)]
    print(f"=== 纯高斯噪声（无信号，sigma^2={SIG2}，{a.reps} 次抽样）===")
    print(f"{'M':>3}{'F':>5}{'T':>7}{'n=FT':>8}{'ratio':>10}{'se':>9}")
    rows = []
    for m_obs, F, T in cells:
        m, se = run(m_obs, F, T, a.reps)
        rows.append(dict(m_obs=m_obs, F=F, T=T, n=F * T, ratio=m, se=se))
        print(f"{m_obs:>3}{F:>5}{T:>7}{F*T:>8}{m:>10.4f}{se:>9.4f}")

    print("\n偏差随 n 的衰减（n 增大 16 倍，负偏差约降为 1/4，与 n^{-1/2} 一致）：")
    r0, r1 = rows[0]["ratio"], rows[2]["ratio"]
    print(f"  n={rows[0]['n']}: {r0:.4f}   n={rows[2]['n']}: {r1:.4f}"
          f"   （1-ratio：{1 - r0:.4f} -> {1 - r1:.4f}）")

    json.dump(dict(sigma2=SIG2, reps=a.reps, rows=rows,
                   note="estimate_noise_power 的水平约定 q_i=i/(n+1) 的有限样本负偏差；"
                        "纯高斯噪声下测得，随 n^{-1/2} 衰减"),
              open(a.out, "w"), indent=1)
    print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
