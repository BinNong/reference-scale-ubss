#!/usr/bin/env python3
"""Table 3 的机器可读记录：闭式 τ(M,α) 下的实测误接纳率。

论文 Table 3 用 2×10^6 个噪声实现验证 Proposition 4（原 Proposition 9，R20 前编号）的校准。此前这份蒙特卡洛只存在于
`validate_nfr.py` 的 `v3_alpha_calibration()` 里、只打印不入档——也就是说 Table 3
（贡献 2 的核心实证）和 §7「Tables 1–4 each have a dedicated … record」那句话，
没有一个记录可对。本脚本补上这一条，并让 `r15_recount.py` 能逐格复算该表。

用法（从仓库根运行）：

    python3 src/r3_tau_mc.py --out ../results/r3_tau_mc.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfr import threshold_ratio

N_SAMPLES = 2_000_000
SEED = 0
# 稿件 Table 3 印的六格
CELLS = [(2, 1e-2), (2, 1e-3), (2, 1e-4), (3, 1e-3), (3, 1e-4), (4, 1e-4)]


def run(m_obs: int, alpha: float, n: int, rng: np.random.Generator) -> dict:
    """在 M 维复高斯噪声上实测 e > τ·M·σ² 的通过率。"""
    s2 = 1.0
    z = (rng.standard_normal((m_obs, n)) + 1j * rng.standard_normal((m_obs, n))) * np.sqrt(s2 / 2.0)
    e = np.sum(np.abs(z) ** 2, axis=0)
    tau = threshold_ratio(m_obs, alpha)
    emp = float(np.mean(e > tau * m_obs * s2))
    # 计数误差：二项分布的标准误，稿件题注引的就是这个量
    se = float(np.sqrt(max(emp * (1.0 - emp), 1e-30) / n))
    return dict(m_obs=m_obs, alpha=alpha, tau=tau, empirical=emp,
                binomial_se=se, rel_err=(emp - alpha) / alpha, n_samples=n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/r3_tau_mc.json")
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"=== τ(M,α) 的蒙特卡洛校准（每格 {a.n:,} 个噪声实现，种子 {SEED}）===")
    print(f"{'M':>3}{'α':>10}{'τ':>10}{'实测通过率':>14}{'二项 se':>12}{'相对误差':>10}")
    for m_obs, alpha in CELLS:
        r = run(m_obs, alpha, a.n, rng)
        rows.append(r)
        print(f"{m_obs:>3}{alpha:>10.0e}{r['tau']:>10.3f}{r['empirical']:>14.4e}"
              f"{r['binomial_se']:>12.2e}{r['rel_err']:>10.2%}")

    worst = max(abs(r["rel_err"]) for r in rows)
    n_in = sum(1 for r in rows if abs(r["empirical"] - r["alpha"]) <= 2 * r["binomial_se"])
    print(f"\n最大相对偏差 {worst:.2%}；{n_in}/{len(rows)} 格落在 ±2 个二项标准误内")

    json.dump(dict(n_samples=a.n, seed=SEED, cells=[list(c) for c in CELLS], rows=rows,
                   note="Table 3：闭式 τ(M,α) 下的实测误接纳率；2e6 个复高斯噪声实现/格，"
                        "二项标准误即稿件题注的计数误差"),
              open(a.out, "w"), indent=1)
    print(f"已写 {a.out}")


if __name__ == "__main__":
    main()
