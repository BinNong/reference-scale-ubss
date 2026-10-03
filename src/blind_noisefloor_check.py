"""盲噪声底估计是否足以支撑「单常数阈值」准则？

noisefloor_check.py 已确认：以**真值**噪声底为基准，单个常数 τ≈50
在全部 6 档稀疏度上都能复现每档最优（均值 2.04° vs 每档最优 1.87°），
且全面优于 MDDE（3.31°）。

但真值噪声底在盲分离中不可得。本脚本检验两个**盲估计器**：

  E1  μ̂ = K · min(e)                       （指数分布极值的尺度反推）
  E2  μ̂ = q_0.005 / (-ln 1-0.005)          （下尾分位数 + 已知指数分位函数）

若任一个能保持"单常数 τ 跨档有效"，则该准则可盲用，构成本文真正的贡献。
"""
from __future__ import annotations

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from noisefloor_check import LADDER, N_SEEDS, hard_mask_abs

TAU_GRID = [10, 20, 50, 100, 200]
Q = 0.005


def est_noise_floor(e_flat: np.ndarray) -> tuple[float, float]:
    """返回 (E1: K·min(e), E2: 下尾分位数拟合)。"""
    e = np.sort(e_flat)
    K = e.size
    e1 = float(K * e[0])
    qs = np.quantile(e, Q)
    e2 = float(qs / (-np.log(1.0 - Q)))
    return e1, e2


def main():
    print("盲噪声底估计器 vs 真值")
    print(f"  {'cfg':<10s}{'真值':>10s}{'E1=K·min':>12s}{'E2=q0.5%':>12s}{'E1/真值':>10s}{'E2/真值':>10s}")
    nf_true, nf_e1, nf_e2 = {}, {}, {}
    for key in LADDER:
        a, b, c = [], [], []
        for s in range(N_SEEDS):
            p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
            a.append(float(np.mean(np.abs(p["noise_tf"]) ** 2)))
            e1, e2 = est_noise_floor((np.abs(p["X_tf"]) ** 2).ravel())
            b.append(e1); c.append(e2)
        nf_true[key], nf_e1[key], nf_e2[key] = np.mean(a), np.mean(b), np.mean(c)
        print(f"  {key:<10s}{np.mean(a):>10.5f}{np.mean(b):>12.5f}{np.mean(c):>12.5f}"
              f"{np.mean(b)/np.mean(a):>10.2f}{np.mean(c)/np.mean(a):>10.2f}")

    mdde = {"tf_p02": 1.75, "tf_p05": 1.04, "tf_p10": 1.02,
            "tf_p20": 1.73, "tf_p40": 3.79, "tf_gauss": 10.56}

    for name, nf in [("E1 = K·min(e)", nf_e1), ("E2 = q_0.5% 拟合", nf_e2)]:
        print()
        print(f"=== 用 {name} 作为噪声底，单常数 τ 跨 6 档 ===")
        print(f"  {'τ':>6s}{'均值°':>9s}   逐档值")
        for tau in TAU_GRID:
            vals = []
            for key in LADDER:
                ae = []
                for s in range(N_SEEDS):
                    p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
                    rng = np.random.default_rng(s)
                    m = hard_mask_abs(p["X_tf"], 0.98, tau * nf[key])
                    A = kmeans_sphere(ssp_directions(p["X_tf"], m), 4, rng)
                    ae.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
                vals.append(float(np.mean(ae)))
            print(f"  {tau:>6.0f}{np.mean(vals):>9.2f}   " + " ".join(f"{v:.2f}" for v in vals))
        print(f"  {'MDDE':>6s}{np.mean(list(mdde.values())):>9.2f}   "
              + " ".join(f"{mdde[k]:.2f}" for k in LADDER))


if __name__ == "__main__":
    main()
