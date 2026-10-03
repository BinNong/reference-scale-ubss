"""公平性检查：给经典流水线一个真正起作用的能量阈值，差距还剩多少？

动机
----
基线 ``ssp_mask_from_complex`` 默认 ``thr_energy_ratio=0.02``，即只要求
能量 > 中位数的 2%——在稀疏数据里这几乎不做任何过滤。而 MDDE 的能量门限
在 ``e/ē ≈ 10`` 处开始衰减。若把经典基线也换成同等激进的能量阈值就能追平，
那么本文的净增量只是"把阈值调得更大"，不足以构成贡献。

本脚本在阈值网格上跑经典流水线（硬阈值 + 球面 K-means，与论文的 SCA-L1 同构），
报告每个稀疏度档位上经典方法的**最优**表现，与 MDDE 对照。

用法（项目 src 目录下）:
    ../.venv/bin/python fairness_check.py
"""
from __future__ import annotations

import numpy as np
import torch

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions, ssp_mask_from_complex, l1_recover
from experiments_mdde import make_model, run_mdde

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
# 经典方法：硬 cos 阈值 × 硬能量阈值（相对中位数的倍数）
COS_GRID = [0.90, 0.95, 0.98, 0.99]
EN_GRID = [0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_SEEDS = 10


def classical_tuned(X_tf: np.ndarray, X_all: np.ndarray, n_true: int,
                    thr_cos: float, thr_en: float, rng) -> np.ndarray:
    """经典两步法：硬 SSP 掩码 + 球面 K-means。"""
    mask = ssp_mask_from_complex(X_tf, thr_cos=thr_cos, thr_energy_ratio=thr_en)
    return kmeans_sphere(ssp_directions(X_tf, mask), n_true, rng)


def main() -> dict:
    dev = torch.device("cpu")
    model = make_model().eval()
    out: dict = {"cos_grid": COS_GRID, "en_grid": EN_GRID, "n_seeds": N_SEEDS, "ladder": {}}

    for key in LADDER:
        grid: dict[str, tuple[float, float]] = {}
        for c0 in COS_GRID:
            for te in EN_GRID:
                ae, nn = [], []
                for s in range(N_SEEDS):
                    p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
                    rng = np.random.default_rng(s)
                    A_hat = classical_tuned(p["X_tf"], p["X_all"], 4, c0, te, rng)
                    ae.append(MT.mixing_matrix_angle_error_deg(p["A"], A_hat))
                    nn.append(int(ssp_mask_from_complex(
                        p["X_tf"], thr_cos=c0, thr_part_ratio=0.1,
                        thr_energy_ratio=te).sum()))
                grid[f"c{c0}_e{te}"] = (float(np.mean(ae)), float(np.mean(nn)))

        best_k = min(grid, key=lambda k: grid[k][0])
        default = grid["c0.98_e0.02"]

        md_ae = []
        for s in range(N_SEEDS):
            p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
            A_hat, _ = run_mdde(p, model, dev)
            md_ae.append(MT.mixing_matrix_angle_error_deg(p["A"], A_hat))

        out["ladder"][key] = {
            "mdde": float(np.mean(md_ae)),
            "classical_default": default[0],
            "classical_default_npts": default[1],
            "classical_best": grid[best_k][0],
            "classical_best_params": best_k,
            "classical_best_npts": grid[best_k][1],
            "grid": {k: v for k, v in grid.items()},
        }

        print(f"\n=== {key} ===")
        print(f"  MDDE（固定配置）           : {np.mean(md_ae):6.2f} deg")
        print(f"  classical default c0=.98 e=.02: {default[0]:6.2f} deg  (kept {default[1]:.0f} pts)")
        print(f"  classical BEST {best_k:>14s}: {grid[best_k][0]:6.2f} deg  (kept {grid[best_k][1]:.0f} pts)")
        print(f"  {'c0\\e':>7s} " + "".join(f"{te:>9.1f}" for te in EN_GRID))
        for c0 in COS_GRID:
            print(f"  {c0:>7.2f} " + "".join(f"{grid[f'c{c0}_e{te}'][0]:>9.2f}" for te in EN_GRID))

    return out


if __name__ == "__main__":
    import argparse
    import json
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/fairness_check.json")
    a = ap.parse_args()
    payload = main()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n已写出 {a.out}")
