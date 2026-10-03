"""合成阶梯上"逐档最优经典门限"落在网格哪里（用于修正 §9.5 的措辞）。

§9.5 称"合成阶梯上逐档最优的 median 系数范围是 0.1 到 1000"，但 §7 声明的
oracle 网格是 EN_GRID = {0.1,0.5,1,2,5,10,20} × COS_GRID。本脚本把每一档的
最优 (c0, te) 找出来，给出可引用的实际范围。

用法（服务器 src 目录）：
    ../.venv/bin/python synth_tuned_grid.py --out ../results/synth_tuned_grid.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import data as D
from experiments_nfr import COS_GRID, EN_GRID, _classical_A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/synth_tuned_grid.json")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    keys = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
    out = []
    print(f"{'regime':<10}{'best c0':>9}{'best te':>9}{'A err':>9}{'te=0.02':>9}{'te=5':>9}")
    for key in keys:
        best, err = None, {}
        for te in EN_GRID:
            for c0 in COS_GRID:
                acc = []
                for s in range(a.seeds):
                    prob = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
                    A = _classical_A(prob, c0, te, np.random.default_rng(s))
                    import metrics as MT
                    acc.append(MT.mixing_matrix_angle_error_deg(prob["A"], A))
                m = float(np.mean(acc))
                err[(c0, te)] = m
                if best is None or m < err[best]:
                    best = (c0, te)
        e002 = err.get((0.98, 0.02), float("nan"))
        e5 = err.get((0.98, 5.0), float("nan"))
        out.append({"regime": key, "c0": best[0], "te": best[1],
                    "A_err": err[best], "A_err_te002": e002, "A_err_te5": e5})
        print(f"{key:<10}{best[0]:>9.2f}{best[1]:>9.1f}{err[best]:>9.3f}{e002:>9.3f}{e5:>9.3f}")

    tes = [r["te"] for r in out]
    print(f"\n最优 te: 范围 [{min(tes)}, {max(tes)}]  唯一值 {sorted(set(tes))}")

    json.dump({"cos_grid": COS_GRID, "energy_grid": EN_GRID, "seeds": a.seeds,
               "rows": out}, open(a.out, "w"), indent=1)
    print(f"已写 {a.out}")


if __name__ == "__main__":
    main()
