"""Prop. 6 的正交性假设检验：活跃列不正交时，闭式偏离多少。

Prop. 6 把 J 个活跃源的信号能量写成 (E_J/2J)·χ²_{2J}，这**要求这 J 列正交**。
本文的设计规定最小列间夹角 12°，因此 J≥2 时 A_J^H A_J 的特征值远离 1
（一对 12° 的列给出 1±cos12° = 1.978 / 0.022），真实能量律是尺度差近 90 倍的
指数之和，与 χ²_{2J} 相去甚远。

本脚本在同一约定下比较三者：
  (a) mc_true  —— 用**真实 A** 蒙特卡洛求混合中位数比值
  (b) mc_orth  —— 同上但把活跃子集换成正交列（即 Prop. 6 的假设）
  (c) closed   —— nfr.mixture_median_ratio 的闭式（论文 Table 1 的闭式列）

用法（服务器 src 目录）：
    ../.venv/bin/python prop6_orthogonality.py --out ../results/prop6_orthogonality.json
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations

import numpy as np

import data as D
from nfr import mixture_median_ratio

N, M, SNR = 4, 2, 20.0
P_LIST = [0.02, 0.05, 0.10, 0.20, 0.40]


def mc_ratio(p: float, A: np.ndarray, rng: np.random.Generator,
             n_pts: int = 400_000, orthogonal: bool = False) -> float:
    """混合中位数 / 噪声底。s² = 1，故 ν = M。

    注意：J≥1 的点是**替换**其噪声分量，不是追加——首版把信号样本追加到数组末尾，
    等于人为放大了"仅噪声点"的比例，使中位数系统性偏低（p=0.10 处 1.16 vs 1.38）。
    """
    p_act = 1.0 - (1.0 - p) ** N
    e_act = (10.0 ** (SNR / 10.0)) * M / p_act          # 单个活跃源的平均能量
    nu = float(M)
    J = rng.binomial(N, p, size=n_pts)
    e = 0.5 * rng.chisquare(2 * M, size=n_pts)          # 先给每个点噪声分量
    for j in range(1, N + 1):
        idx = np.flatnonzero(J == j)
        if idx.size == 0:
            continue
        subsets = list(combinations(range(N), j))
        for sub, part in zip(subsets, np.array_split(idx, len(subsets))):
            if part.size == 0:
                continue
            if orthogonal:
                e_sig = 0.5 * e_act * rng.chisquare(2 * j, size=part.size)
            else:
                Aj = A[:, list(sub)]                      # M×j
                S = (rng.standard_normal((j, part.size))
                     + 1j * rng.standard_normal((j, part.size)))
                S *= math.sqrt(e_act / 2.0)               # 使 E|s_j|² = e_act
                W = Aj @ S
                e_sig = (W.real ** 2 + W.imag ** 2).sum(axis=0)
            e[part] = e_sig + 0.5 * rng.chisquare(2 * M, size=part.size)
    return float(np.median(e) / nu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/prop6_orthogonality.json")
    ap.add_argument("--n-a", type=int, default=5, help="A 的抽样个数")
    ap.add_argument("--n-pts", type=int, default=400_000)
    a = ap.parse_args()

    out = []
    print(f"{'p':<8}{'mc_true':>10}{'mc_orth':>10}{'closed':>10}{'闭合-真值':>12}{'正交-真值':>12}")
    for p in P_LIST:
        trues, orths = [], []
        for k in range(a.n_a):
            rng = np.random.default_rng(100 + k)
            A = D.gen_mixing_matrix(M, N, rng, min_angle_deg=12.0)
            trues.append(mc_ratio(p, A, rng, a.n_pts, orthogonal=False))
            orths.append(mc_ratio(p, A, rng, a.n_pts, orthogonal=True))
        cl = mixture_median_ratio(p, N, M, SNR)
        r_true, r_orth = float(np.mean(trues)), float(np.mean(orths))
        r_cl = float(cl["ratio"])
        out.append({"p": p, "pi0": float((1 - p) ** N), "r_mc_true": r_true,
                    "r_mc_true_std": float(np.std(trues)),
                    "r_mc_orth": r_orth, "r_closed": r_cl,
                    "in_noise_regime": bool(cl["in_noise_regime"]),
                    "closed_delta_abs": (r_cl - r_true) if math.isfinite(r_cl) else None,
                    "closed_delta_rel": ((r_cl - r_true) / r_true
                                         if math.isfinite(r_cl) else None),
                    "orth_delta_abs": r_orth - r_true})
        print(f"{p:<8}{r_true:>10.3f}{r_orth:>10.3f}"
              f"{(r_cl if math.isfinite(r_cl) else float('nan')):>10.3f}"
              f"{(r_cl - r_true) if math.isfinite(r_cl) else float('nan'):>12.3f}"
              f"{r_orth - r_true:>12.3f}")

    json.dump({"n_sources": N, "m_obs": M, "snr_db": SNR, "n_A": a.n_a,
               "n_points": a.n_pts, "rows": out},
              open(a.out, "w"), indent=1)
    print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
