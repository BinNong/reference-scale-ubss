"""把"致命发现"转化为"新方向"：最优能量阈值为什么随稀疏度漂移？

发现
----
经典流水线只要把能量阈值打开（thr_energy_ratio 从 0.02 提到 5~20），
稀疏端 A 误差就从 15.36° 掉到 0.29°——全面优于 MDDE。但最优阈值本身
随稀疏度从 ~10（稀疏）漂移到 ~0.1（密集），跨度 100 倍。

假设（本脚本要验证的）
----------------------
最优阈值若以「噪声底」而非「能量中位数」为基准，则跨稀疏度是**常数**。
即存在 τ* 使 `e > τ* · noise_floor` 在所有档位上同时接近各自的最优值。
若成立，则本文真正的贡献是：**一个不需要知道 p 的盲阈值准则**，
而 MDDE 那套固定 τ_e=10 的软门限只是这条曲线上一个不够好的点。
"""
from __future__ import annotations

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions, ssp_mask_from_complex

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
N_SEEDS = 10
TAU_GRID = [0.02, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 300, 1000]


def hard_mask_abs(X_tf, thr_cos, thr_abs, thr_part_ratio=0.1):
    """硬 cos 判据 + 硬对称均衡判据 + **绝对**能量阈值（非相对中位数）。"""
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    tot = nr ** 2 + ni ** 2
    eps = 1e-15
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
    denom = np.maximum(nr + ni, eps)
    both = (nr / denom > thr_part_ratio) & (ni / denom > thr_part_ratio)
    return (np.abs(cos) > thr_cos) & both & (tot > thr_abs)


def kmeans_with_energy_mask(X_tf, X_all, n_true, thr_cos, thr_abs, rng):
    """硬 cos 阈值 + 绝对能量阈值。"""
    mask = hard_mask_abs(X_tf, thr_cos, thr_abs)
    A = kmeans_sphere(ssp_directions(X_tf, mask), n_true, rng)
    return A, int(mask.sum())


def main():
    print("每个档位：真值噪声底、能量中位数、比值 r = median / noise_floor")
    print(f"{'cfg':<10s}{'noise_floor':>13s}{'median_e':>12s}{'r':>9s}")
    ratios = {}
    for key in LADDER:
        nf, md = [], []
        for s in range(3):
            p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
            nt = p["noise_tf"]
            nf.append(float(np.mean(np.abs(nt) ** 2)))
            e = np.abs(p["X_tf"]) ** 2
            md.append(float(np.median(e.ravel())))
        ratios[key] = (float(np.mean(nf)), float(np.mean(md)), float(np.mean(md) / np.mean(nf)))
        n_, m_, r_ = ratios[key]
        print(f"{key:<10s}{n_:>13.5f}{m_:>12.5f}{r_:>9.3f}")

    print()
    print("=== 以噪声底为基准的绝对阈值扫描（A 误差°, 10 seeds）===")
    print(f"  {'cfg':<10s}" + "".join(f"{t:>8.2f}" for t in TAU_GRID) + f"{'最优τ':>9s}{'最优值':>8s}")
    best_tau, best_val = {}, {}
    for key in LADDER:
        nf = ratios[key][0]
        row = []
        for tau in TAU_GRID:
            ae = []
            for s in range(N_SEEDS):
                p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
                rng = np.random.default_rng(s)
                A, _ = kmeans_with_energy_mask(p["X_tf"], p["X_all"], 4, 0.98, tau * nf, rng)
                ae.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
            row.append(float(np.mean(ae)))
        k = int(np.argmin(row))
        best_tau[key], best_val[key] = TAU_GRID[k], row[k]
        print(f"  {key:<10s}" + "".join(f"{v:>8.2f}" for v in row)
              + f"{TAU_GRID[k]:>9.2f}{row[k]:>8.2f}")

    print()
    print("=== 单个固定 τ 跨全部 6 档的表现（vs 每档最优、vs MDDE）===")
    mdde = {"tf_p02": 1.75, "tf_p05": 1.04, "tf_p10": 1.02,
            "tf_p20": 1.73, "tf_p40": 3.79, "tf_gauss": 10.56}
    print(f"  {'τ':>8s}{'均值°':>9s}{'最差档°':>9s}   逐档值")
    for tau in [5, 10, 20, 50, 100]:
        vals = []
        for key in LADDER:
            nf = ratios[key][0]
            ae = []
            for s in range(N_SEEDS):
                p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
                rng = np.random.default_rng(s)
                A, _ = kmeans_with_energy_mask(p["X_tf"], p["X_all"], 4, 0.98, tau * nf, rng)
                ae.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
            vals.append(float(np.mean(ae)))
        print(f"  {tau:>8.0f}{np.mean(vals):>9.2f}{np.max(vals):>9.2f}   "
              + " ".join(f"{v:.2f}" for v in vals))
    print()
    print("  MDDE 逐档  : " + " ".join(f"{mdde[k]:.2f}" for k in LADDER)
          + f"   均值 {np.mean(list(mdde.values())):.2f}")
    print("  每档最优   : " + " ".join(f"{best_val[k]:.2f}" for k in LADDER)
          + f"   均值 {np.mean(list(best_val.values())):.2f}")


if __name__ == "__main__":
    main()
