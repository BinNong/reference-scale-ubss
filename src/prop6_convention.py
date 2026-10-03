"""判定 r(p) 闭式在膝上偏离的**真正原因**：约定错误 vs 正交性 vs 膝盖。

背景
----
论文 Prop. 6 的证明把单个活跃点的平均信号能量写成
    E_1 = M·s²·10^(SNR/10) / P(act),    P(act) = 1 − (1−p)^N
而生成器（data.py）的实际约定是：每条源在**整张 TF 平面**上平均功率为 1（_unit_power_tf），
于是活跃点上 E|s_n|² = 1/p，且 E_J = J/p。

注意 M·s²·10^(SNR/10) 恰等于平面平均总信号能量（= N，因每条源平面均值 1）。
把它摊到"至少一个源活跃"的比例 P(act) 上，得到的是**活跃点上总能量的均值**
（已对 J 平均），不是 J=1 时的值；随后又按 E_J = J·E_1 线性放大，等于把
Σ_J π_J E_J = E_1·Np/P(act) 重复计入。自洽的解法是
    E_1 = M·s²·10^(SNR/10) / (N·p)
两者在小 p 下重合（Np ≈ P(act)），在 p 增大时按 Np/P(act) 偏离（p=0.2 时 1.36×，
p=0.4 时 1.84×，稠密时 4×）。

本脚本给出四组可对照的量：
  r_emp    —— 直接从真实流水线取 median(||x||²)/ν  （= 论文 Table 1 的 measured 列）
  r_paper  —— 蒙特卡洛，用论文的 E_J 约定 + 真实 A
  r_fixed  —— 蒙特卡洛，用修正后的 E_J 约定 + 真实 A
  closed_* —— mixture_median_ratio 的闭式（论文约定 / 修正约定）
并单独核对：empirical E[energy | J=1] / ν 与两个预测值哪一个相符。

用法（服务器 src 目录）：
    ../.venv/bin/python prop6_convention.py --out ../results/prop6_convention.json
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
P_LIST = [0.02, 0.05, 0.10, 0.20, 0.40, 1.0]
TAG = {0.02: "tf_p02", 0.05: "tf_p05", 0.10: "tf_p10",
       0.20: "tf_p20", 0.40: "tf_p40", 1.0: "tf_gauss"}


def mc_ratio(p, A, s2, rng, n_pts, convention="fixed", orthogonal=False):
    """混合中位数 / ν。convention ∈ {'paper','fixed'}。"""
    p_act = 1.0 - (1.0 - p) ** N
    e_act = (10.0 ** (SNR / 10.0)) * M * s2 / (p_act if convention == "paper" else N * p)
    nu = M * s2
    J = rng.binomial(N, p, size=n_pts)
    e = 0.5 * s2 * rng.chisquare(2 * M, size=n_pts)
    for j in range(1, N + 1):
        idx = np.flatnonzero(J == j)
        if idx.size == 0:
            continue
        subs = list(combinations(range(N), j))
        for sub, part in zip(subs, np.array_split(idx, len(subs))):
            if part.size == 0:
                continue
            if orthogonal:
                e_sig = 0.5 * e_act * rng.chisquare(2 * j, size=part.size)
            else:
                S = (rng.standard_normal((j, part.size))
                     + 1j * rng.standard_normal((j, part.size)))
                S *= math.sqrt(e_act / 2.0)
                W = A[:, list(sub)] @ S
                e_sig = (W.real ** 2 + W.imag ** 2).sum(axis=0)
            e[part] = e_sig + 0.5 * s2 * rng.chisquare(2 * M, size=part.size)
    return float(np.median(e) / nu)


def closed_ratio(p, convention):
    """论文闭式；修正约定通过重标定 SNB 实现（等价于把 e_act 除以 Np/P(act)）。"""
    # 函数现在直接支持两种约定，不再需要用 SNB 平移来模拟
    return mixture_median_ratio(p, N, M, SNR,
                                convention="pact" if convention == "paper" else "np")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/prop6_convention.json")
    ap.add_argument("--n-a", type=int, default=5)
    ap.add_argument("--n-pts", type=int, default=400_000)
    a = ap.parse_args()

    # --- 先取真实流水线的经验量（每个 cfg 5 个种子）---
    emp = {}
    for p in P_LIST:
        rs, e1_over_nu, s2s = [], [], []
        for s in range(5):
            prob = D.make_problem(M, N, 33, 64, TAG[p], SNR, seed=1000 + s)
            X, S = prob["X_tf"], prob["S_tf"]
            s2 = float(np.mean(np.abs(prob["noise_tf"]) ** 2))
            nu = M * s2
            e = np.sum(np.abs(X) ** 2, axis=0).ravel()
            rs.append(float(np.median(e) / nu))
            # J = 恰有 1 条源活跃 的点上的平均信号能量 / ν
            act = np.abs(S) > 0
            J = act.sum(axis=0).ravel()
            sig = np.sum(np.abs(np.einsum("mn,nft->mft", prob["A"], S)) ** 2, axis=0).ravel()
            m1 = J == 1
            if m1.sum() > 0:
                e1_over_nu.append(float(sig[m1].mean() / nu))
            s2s.append(s2)
        emp[p] = {"r": float(np.mean(rs)), "r_std": float(np.std(rs)),
                  "E1_over_nu": float(np.mean(e1_over_nu)) if e1_over_nu else float("nan"),
                  "sigma2": float(np.mean(s2s))}

    rows = []
    print(f"{'p':<7}{'r_emp':>9}{'r_paper':>9}{'r_fixed':>9}"
          f"{'closed_p':>10}{'closed_f':>10}{'E1/nu经验':>11}{'E1/nu论文':>11}{'E1/nu修正':>11}")
    for p in P_LIST:
        s2 = emp[p]["sigma2"]
        rp, rf, ro = [], [], []
        for k in range(a.n_a):
            rng = np.random.default_rng(100 + k)
            A = D.gen_mixing_matrix(M, N, rng, min_angle_deg=12.0)
            rp.append(mc_ratio(p, A, s2, rng, a.n_pts, "paper"))
            rf.append(mc_ratio(p, A, s2, rng, a.n_pts, "fixed"))
            ro.append(mc_ratio(p, A, s2, rng, a.n_pts, "fixed", orthogonal=True))
        cp = closed_ratio(p, "paper")["ratio"]
        cf = closed_ratio(p, "fixed")["ratio"]
        p_act = 1.0 - (1.0 - p) ** N
        e1_paper = (10.0 ** (SNR / 10.0)) * M * s2 / p_act / (M * s2)
        e1_fixed = 1.0 / (p * M * s2)
        row = {"p": p, "pi0": float((1 - p) ** N), "r_emp": emp[p]["r"],
               "r_paper": float(np.mean(rp)), "r_fixed": float(np.mean(rf)),
               "r_fixed_orth": float(np.mean(ro)),
               "closed_paper": cp, "closed_fixed": cf,
               "E1_over_nu_emp": emp[p]["E1_over_nu"],
               "E1_over_nu_paper": e1_paper, "E1_over_nu_fixed": e1_fixed,
               "Np_over_Pact": float(N * p / p_act)}
        rows.append(row)
        f = lambda v: v if (v is not None and np.isfinite(v)) else float("nan")
        print(f"{p:<7}{emp[p]['r']:>9.3f}{np.mean(rp):>9.3f}{np.mean(rf):>9.3f}"
              f"{f(cp):>10.3f}{f(cf):>10.3f}{f(emp[p]['E1_over_nu']):>11.1f}"
              f"{e1_paper:>11.1f}{e1_fixed:>11.1f}")

    json.dump({"n_sources": N, "m_obs": M, "snr_db": SNR, "n_A": a.n_a,
               "n_points": a.n_pts, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
