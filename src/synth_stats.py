"""合成对照的配对检验：校正"单次实现"bug 之后，加权到底有没有好处。

用法: ../.venv/bin/python synth_stats.py --json ../results/synth_reseed.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

LAD = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]


def arr(recs, m, case):
    return np.array([r["A"] for r in sorted(recs, key=lambda x: x["seed"])
                     if r["case"] == case and r["method"] == m and np.isfinite(r["A"])])


def pooled(recs, m):
    return np.concatenate([arr(recs, m, c) for c in LAD])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../results/synth_reseed.json")
    a = ap.parse_args()
    d = json.load(open(a.json))
    W, P = d["records"]["weighted"], d["records"]["purity"]

    print("=== weighted: NF vs NF-W（能量权）逐档配对 t 检验 (n=8) ===")
    print(f"{'case':<10}{'NF':>8}{'NF-W':>8}{'diff':>9}{'p':>12}")
    for c in LAD:
        x, y = arr(W, "NF", c), arr(W, "NF-W", c)
        print(f"{c:<10}{x.mean():>8.3f}{y.mean():>8.3f}"
              f"{y.mean()-x.mean():>9.3f}{stats.ttest_rel(x, y).pvalue:>12.2e}")
    x, y = pooled(W, "NF"), pooled(W, "NF-W")
    print(f"{'pooled':<10}{x.mean():>8.3f}{y.mean():>8.3f}"
          f"{y.mean()-x.mean():>9.3f}{stats.ttest_rel(x, y).pvalue:>12.2e}")

    print("\n=== purity: NF 为参照，池化配对 (n=48) ===")
    base = pooled(P, "NF")
    for m in ["NF+e", "NF+e·cos4", "NF+e·ρ2", "NF+e·ρ4", "NF+e·ρ8",
              "base+e", "base+e·ρ4"]:
        b = pooled(P, m)
        print(f"  NF vs {m:<12} NF={base.mean():>6.3f}  该法={b.mean():>6.3f}"
              f"  diff={b.mean()-base.mean():+6.3f}  p={stats.ttest_rel(base, b).pvalue:.2e}")

    print("\n=== purity: 各变体逐档均值 ===")
    met = ["NF", "NF+e", "NF+e·cos4", "NF+e·ρ2", "NF+e·ρ4", "NF+e·ρ8", "base+e", "base+e·ρ4"]
    print(f"{'case':<10}" + "".join(f"{m:>11s}" for m in met))
    for c in LAD:
        print(f"{c:<10}" + "".join(f"{arr(P, m, c).mean():>11.2f}" for m in met))
    print(f"{'MEAN':<10}" + "".join(f"{pooled(P, m).mean():>11.2f}" for m in met))


if __name__ == "__main__":
    main()
