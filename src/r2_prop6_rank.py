"""R2-7  Proposition 6 的近似究竟来自哪里：列夹角，还是活跃 Gram 的秩亏？

审稿意见 MC1 猜测闭式的偏差来自「活跃列不正交」。本脚本把这两件事分开测：
对固定的活跃源数 J，直接比较信号能量 ‖A_J s‖² 的真实分布与
「特征值换成均值」的近似分布，并报告 A_J^H A_J 的特征值谱。

若偏差随 J 增长而不随夹角变化，则主导因素是秩亏（M=2、J>M 时
J−2 个特征值恒为 0），而不是列间的接近程度。

用法：
    python3 r2_prop6_rank.py --seeds 40 --out ../results/r2_prop6_rank.json
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np

import data as D

M_OBS, N_SRC = 2, 4
N_PTS = 400_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--min-angle", type=float, default=12.0)
    ap.add_argument("--out", default="../results/r2_prop6_rank.json")
    a = ap.parse_args()

    rng = np.random.default_rng(12345)
    rows = []
    for s in range(a.seeds):
        A = D.gen_mixing_matrix(M_OBS, N_SRC, np.random.default_rng(2000 + s), a.min_angle)
        # 实际的最小两列夹角
        G = A.T @ A
        off = np.abs(G - np.eye(N_SRC))
        ic = np.unravel_index(np.argmax(off), off.shape)
        min_ang = float(np.degrees(np.arccos(np.clip(off[ic], 0, 1))))
        for J in range(1, N_SRC + 1):
            subs = list(combinations(range(N_SRC), J))
            devs, specs = [], []
            for sub in subs:
                Aj = A[:, list(sub)]
                lam = np.maximum(np.linalg.eigvalsh(Aj.T @ Aj), 1e-14)
                u = rng.exponential(1.0, size=(J, N_PTS))
                true = (lam[:, None] * u).sum(axis=0)          # 单位化：Ej/J 提出来
                orth = (lam.mean() * u).sum(axis=0)
                devs.append(float(np.median(orth) / np.median(true) - 1.0))
                specs.append([float(x) for x in lam])
            rows.append(dict(seed=s, J=J, min_angle_deg=min_ang,
                             dev=float(np.mean(devs)),
                             lam_spectrum=specs[0],
                             n_zero_modes=int(sum(1 for x in specs[0] if x < 1e-6))))

    print(f"\n=== 信号能量定律：真实 Gram vs 特征值取均值（min angle {a.min_angle}°）===")
    print(f"{'J':>3}{'特征值谱（首个子集）':>34}{'零模数':>8}{'中位数相对偏差':>16}")
    print("-" * 62)
    for J in range(1, N_SRC + 1):
        sub = [r for r in rows if r["J"] == J]
        sp = np.mean([r["lam_spectrum"] for r in sub], axis=0)
        print(f"{J:>3}{str(np.round(sp, 3).tolist()):>34}"
              f"{np.mean([r['n_zero_modes'] for r in sub]):>8.1f}"
              f"{np.mean([r['dev'] for r in sub])*100:>15.2f}%")

    print("\n（J≤M 时 Gram 满秩；J>M 时出现 J−M 个零模，"
          "而『特征值取均值』把它们替换成 1，偏差由此而来）")
    print(f"实测最小夹角：{np.mean([r['min_angle_deg'] for r in rows]):.2f}°"
          f"（范围 {min(r['min_angle_deg'] for r in rows):.2f}–"
          f"{max(r['min_angle_deg'] for r in rows):.2f}°）")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": vars(a), "records": rows}, open(a.out, "w"))
        print(f"已写 {a.out}")


if __name__ == "__main__":
    main()
