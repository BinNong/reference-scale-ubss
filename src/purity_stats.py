"""纯度加权在真实语音（全 15 配置 × 8 seeds）上的配对检验。

关键问题：ρ（纯度）能否胜过 |cos|^4（共线统计量）？
若不能，则"追踪纯度的权重"是一个被检验的否定结果，应如实写入手稿。

用法: ../.venv/bin/python purity_stats.py --json ../results/purity_stage2.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

MET = ["NF", "NF+e", "NF+e·cos4", "NF+e·ρ2", "NF+e·ρ4", "NF+e·ρ8", "base+e", "base+e·ρ4"]


def arr(recs, m, case, metric="A"):
    return np.array([r[metric] for r in sorted(recs, key=lambda x: x["seed"])
                     if r["case"] == case and r["method"] == m and np.isfinite(r[metric])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../results/purity_stage2.json")
    a = ap.parse_args()
    d = json.load(open(a.json))
    recs = [r for r in d["records"] if r.get("dom") == "real"]
    cases = sorted({r["case"] for r in recs})
    print(f"真实配置 {len(cases)} 个，记录 {len(recs)} 条\n")

    print("=== 真实语音 A 误差（8 seeds 均值，度）===")
    print(f"{'case':<18}" + "".join(f"{m:>11s}" for m in MET))
    for c in cases:
        print(f"{c:<18}" + "".join(f"{arr(recs, m, c).mean():>11.2f}" for m in MET))
    row = f"{'MEAN':<18}"
    for m in MET:
        v = np.concatenate([arr(recs, m, c) for c in cases])
        row += f"{v.mean():>11.2f}"
    print(row)

    pool = {m: np.concatenate([arr(recs, m, c) for c in cases]) for m in MET}

    print("\n=== 池化配对 t 检验（n = 15 配置 × 8 seeds）===")
    for ref, cand in [("NF", "NF+e"), ("NF", "NF+e·cos4"), ("NF", "NF+e·ρ2"),
                      ("NF", "NF+e·ρ4"), ("NF", "NF+e·ρ8"),
                      ("NF+e", "NF+e·cos4"), ("NF+e", "NF+e·ρ2"), ("NF+e", "NF+e·ρ8"),
                      ("NF+e·cos4", "NF+e·ρ2"), ("NF+e·cos4", "NF+e·ρ4"),
                      ("NF+e·cos4", "NF+e·ρ8"),
                      ("NF+e", "base+e")]:
        x, y = pool[ref], pool[cand]
        p = stats.ttest_rel(x, y).pvalue
        print(f"  {ref:<11} {x.mean():>6.3f}  vs  {cand:<11} {y.mean():>6.3f}"
              f"   diff={y.mean()-x.mean():+6.3f}   p={p:.2e}")

    print("\n=== 逐配置胜负：cos4 与 ρ 族的 A 误差比较 ===")
    win = {m: 0 for m in ["NF+e·ρ2", "NF+e·ρ4", "NF+e·ρ8"]}
    for c in cases:
        base = arr(recs, "NF+e·cos4", c).mean()
        for m in win:
            if arr(recs, m, c).mean() < base:
                win[m] += 1
    print(f"  （共 {len(cases)} 配置）ρ 优于 cos4 的配置数：", win)


if __name__ == "__main__":
    main()
