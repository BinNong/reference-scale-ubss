#!/usr/bin/env python3
"""复核第四轮新增的四张表（Table 27–30）与 §5.3/§9/§5.1 中手工敲入的数字。

判据与 `verify_revision.py` 相同：把**手稿里写的值**与**归档记录**逐条对照，
不靠记忆、不靠肉眼。用法:

  python3 verify_r4.py            # 纯标准库 + numpy，本地可跑
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
MS = ROOT / "paper" / "manuscript.md"
FAIL: list[str] = []


def check(label: str, claimed, actual, tol: float = 0.02) -> None:
    """逐项比较；claimed/actual 都可以是元组（一个表格行里写了好几个量）。"""
    if isinstance(claimed, tuple) or isinstance(actual, tuple):
        cs, as_ = tuple(claimed), tuple(actual)
        for i, (c, a) in enumerate(zip(cs, as_)):
            check(f"{label} [{i}]", c, a, tol)
        return
    ok = (claimed == actual) if isinstance(claimed, int) and isinstance(actual, int) \
        else abs(float(claimed) - float(actual)) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<52} 稿 {claimed}  归档 {actual}")
    if not ok:
        FAIL.append(label)


def main() -> None:
    recs = json.loads((R / "r4_guard_errors.json").read_text())["records"]
    ms = MS.read_text() if MS.exists() else None

    # ---------------- Table 27（守卫错误分布与消融）----------------
    print("=== Table 27：守卫错误分布 ===")
    fam = {"sparse": ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40"],
           "dense": ["tf_gauss"],
           "structured": ["td_chirp", "td_sinusoid", "td_amfm", "td_impulse"],
           "laplace": ["tf_lap", "tf_b06"]}
    want = {"sparse": (150, 4), "dense": (60, 2), "structured": (40, 8), "laplace": (20, 3)}
    tot_e = 0
    for k, cases in fam.items():
        sub = [r for r in recs if r["case"] in cases]
        e = sum(r["error"] for r in sub)
        check(f"  {k}: 决策数", want[k][0], len(sub), 0)
        check(f"  {k}: 实质错误", want[k][1], e, 0)
        tot_e += e
    check("  合计错误", 17, tot_e, 0)
    check("  合计决策", 270, len(recs), 0)
    check("  假开启", 14, sum(1 for r in recs if r["error"] and r["decision"] == "ON"), 0)
    check("  假回退", 3, sum(1 for r in recs if r["error"] and r["decision"] == "OFF"), 0)
    check("  SNR<=5 的错误数", 4,
          sum(1 for r in recs if r["error"] and r["snr_db"] <= 5), 0)
    check("  retention 单独绑定（合成）", 68,
          sum(1 for r in recs if r["decision"] == "OFF" and r["retention_ok"] is False
              and r["spread_ok"] and r["base_ok"]), 0)
    check("  spread 单独绑定（合成）", 0,
          sum(1 for r in recs if r["spread_ok"] is False and r["retention_ok"] and r["base_ok"]), 0)
    check("  去掉 spread 后的错误数", 17, sum(r["error_no_spread"] for r in recs), 0)

    # spread 在真实语音里的绑定次数
    rr = json.loads((R / "real_results.json").read_text())["records"]
    nf = [r for r in rr if r["method"] == "NF-SSP"]
    bound = 0
    for r in nf:
        need = max(10, math.ceil(0.035 * max(int(r["n_base"]), 1)))
        sp_ok = np.isfinite(r["spread"]) and r["spread"] <= 0.5
        if (not r["gate_on"]) and int(r["n_keep"]) >= need and int(r["n_base"]) >= 10 and not sp_ok:
            bound += 1
    check("  spread 单独绑定（真实语音）", 5, bound, 0)
    check("  真实语音决策数", 120, len(nf), 0)

    # ---------------- Table 28（密度前端）----------------
    print("\n=== Table 28：密度前端 ===")
    d = json.loads((R / "r4_density_frontend.json").read_text())
    for eps, claim in (("0.02", (74, 3.51, 3.92)), ("0.05", (91, 6.54, 2.11)),
                       ("0.1", (98, 8.45, 2.17))):
        r = d["by_eps"][eps]
        keep = np.mean([v["db_kept_frac"] for v in r["per"].values()]) * 100
        check(f"  eps={eps}: 保留率(%)", claim[0], round(keep), 1.0)
        check(f"  eps={eps}: 经典", claim[1], r["mean_classical"], 0.01)
        check(f"  eps={eps}: NF", claim[2], r["mean_nf"], 0.01)
    p10 = d["by_eps"]["0.1"]["per"]
    check("  eps=0.10, p=0.02: 经典", 15.36, p10["tf_p02"]["A_classical"], 0.01)
    check("  eps=0.10, p=0.02: NF", 0.36, p10["tf_p02"]["A_nf"], 0.01)
    check("  eps=0.10, p=0.05: 经典", 13.00, p10["tf_p05"]["A_classical"], 0.01)
    check("  eps=0.10, p=0.05: NF", 0.32, p10["tf_p05"]["A_nf"], 0.01)
    p02 = d["by_eps"]["0.02"]["per"]
    check("  eps=0.02, p=0.02: 经典", 0.52, p02["tf_p02"]["A_classical"], 0.01)
    check("  eps=0.02, dense: 保留率(%)", 29, round(p02["tf_gauss"]["db_kept_frac"] * 100), 1.0)
    check("  eps=0.02, dense: 经典", 18.08, p02["tf_gauss"]["A_classical"], 0.02)
    check("  eps=0.02, dense: NF", 21.09, p02["tf_gauss"]["A_nf"], 0.02)

    # ---------------- Table 29（鲁棒噪声底）----------------
    print("\n=== Table 29：鲁棒噪声底 ===")
    rf = json.loads((R / "r4_robust_floor.json").read_text())["records"]

    def mean(dom, kind, p, field, real_kind=None):
        key = "p" if dom == "synth" else "win"
        sub = [r for r in rf if r["dom"] == dom and r["kind"] == kind
               and (r[key] == p if dom == "synth" else True)]
        return float(np.nanmean([r[field] for r in sub]))

    rows = [("gauss", 0.05, (0.14, 1.09, 1.56, 1.49, 0.40, 0.39, 0.40)),
            ("gauss", 0.40, (0.14, 3.07, 51.9, 23.2, 1.39, 10.13, 3.93)),
            ("laplace", 0.40, (0.22, 1.60, 53.0, 23.0, 1.19, 9.86, 4.42)),
            ("impulsive", 0.40, (0.15, 0.93, 52.9, 18.6, 1.02, 11.19, 5.11)),
            ("colored", 0.40, (0.12, 3.05, 52.5, 23.1, 1.31, 9.65, 6.00)),
            ("uniform", 0.40, (0.09, 5.24, 52.8, 23.0, 2.61, 10.76, 4.25))]
    for kind, p, c in rows:
        check(f"  {kind} p={p}: stat", c[0], round(mean("synth", kind, p, "stat_noise"), 2), 0.01)
        check(f"  {kind} p={p}: ratio paper/mad/q25",
              (c[1], c[2], c[3]),
              (round(mean("synth", kind, p, "ratio_paper"), 2),
               round(mean("synth", kind, p, "ratio_mad"), 1),
               round(mean("synth", kind, p, "ratio_q25"), 1)), 0.05)
        check(f"  {kind} p={p}: A paper", c[4], round(mean("synth", kind, p, "A_paper"), 2), 0.01)
        check(f"  {kind} p={p}: A mad", c[5], round(mean("synth", kind, p, "A_mad"), 2), 0.01)
        check(f"  {kind} p={p}: A q25", c[6], round(mean("synth", kind, p, "A_q25"), 2), 0.01)
    for kind, c in (("babble", (5.80, 0.015, 0.62, 0.25, 1.22, 0.59, 0.69)),
                    ("gauss", (0.10, 1.61, 2.55, 2.34, 0.56, 0.45, 0.47))):
        sub = [r for r in rf if r["dom"] == "real" and r["kind"] == kind]
        chk = lambda f: float(np.nanmean([r[f] for r in sub]))
        check(f"  real {kind}: stat", c[0], round(chk("stat_noise"), 2), 0.01)
        check(f"  real {kind}: ratio paper/mad/q25",
              (c[1], c[2], c[3]),
              (round(chk("ratio_paper"), 3), round(chk("ratio_mad"), 2), round(chk("ratio_q25"), 2)),
              0.01)
        check(f"  real {kind}: A paper/mad/q25", (c[4], c[5], c[6]),
              (round(chk("A_paper"), 2), round(chk("A_mad"), 2), round(chk("A_q25"), 2)), 0.01)

    # ---------------- Table 2 / Q1 / Q3 ----------------
    print("\n=== Table 2、Q1、Q3 ===")
    t2 = json.loads((R / "r4_table2.json").read_text())
    for row, claim in zip(t2["rows"], [1.01, 1.07, 1.24, 1.57, 2.87, 77.14]):
        check(f"  Table 2 {row['case']}: ratio", claim, round(row["ratio"], 2), 0.01)
    check("  Table 2 dense（8 seeds）", 84.2, round(t2["dense_8"]["ratio"], 1), 0.05)
    check("  Table 2 dense 逐种子下限", 55.6, round(min(t2["dense_8"]["ratio_per_seed"]), 1), 0.05)
    check("  Table 2 dense 逐种子上限", 99.6, round(max(t2["dense_8"]["ratio_per_seed"]), 1), 0.05)

    bias = json.loads((R / "r4_floor_bias.json").read_text())["rows"]
    check("  §5.1(iii) n=2112", 0.977, round(bias[0]["ratio"], 3), 0.001)
    check("  §5.1(iii) n=8448", 0.984, round(bias[1]["ratio"], 3), 0.001)
    check("  §5.1(iii) n=33792", 0.994, round(bias[2]["ratio"], 3), 0.001)

    pc = {r["p"]: r for r in json.loads((R / "prop6_convention.json").read_text())["rows"]}
    check("  Q1 p=0.20 闭式(1/Np) 相对误差(%)", 0.2,
          round((pc[0.2]["closed_fixed"] - pc[0.2]["r_emp"]) / pc[0.2]["r_emp"] * 100, 1), 0.1)
    check("  Q1 p=0.20 闭式(1/Pact) 相对误差(%)", 35.2,
          round((pc[0.2]["closed_paper"] - pc[0.2]["r_emp"]) / pc[0.2]["r_emp"] * 100, 1), 0.2)

    grid = json.loads((R / "real_grid.json").read_text())
    cases = sorted({k.split("|")[0] for k in grid})
    cbest = [min(grid[f"{c}|max"], key=lambda k: grid[f"{c}|max"][k]) for c in cases]
    best = [min(grid[f"{c}|max"].values()) for c in cases]
    fixed = [grid[f"{c}|max"]["0.05"] for c in cases]
    # §A.7 的手稿值从手稿抓取（该段是硬换行的，用容忍空白的正则），不写死常量。
    _ms = MS.read_text()

    def _claim(pat, name):
        m = re.search(pat, _ms, re.S)
        if not m:
            FAIL.append(f"手稿读不到 {name}")
            print(f"  FAIL 手稿里读不到 {name}")
            return float("nan")
        return float(m.group(1))

    q_lo = _claim(r"coefficient runs over\s+\$([\d.]+)\$–\$[\d.]+\$", "§A.7 max 系数下限")
    q_cap = _claim(r"coefficient runs over\s+\$[\d.]+\$–\$([\d.]+)\$", "§A.7 max 系数上限")
    q_fix = _claim(r"coefficient at \$0\.05\$ costs \$([\d.]+)°\$", "§A.7 固定 0.05 的均值")
    q_opt = _claim(r"on average against \$([\d.]+)°\$ for the\s+per-configuration optimum", "§A.7 逐档最优均值")
    check(f"  Q3 max 系数下限（稿 {q_lo}）", q_lo, min(float(x) for x in cbest), 1e-9)
    check(f"  Q3 max 系数上限（稿 {q_cap}）", q_cap, max(float(x) for x in cbest), 1e-9)
    check(f"  Q3 固定 0.05 的均值（稿 {q_fix}）", q_fix, float(np.mean(fixed)), 0.01)
    check(f"  Q3 逐档最优的均值（稿 {q_opt}）", q_opt, float(np.mean(best)), 0.01)

    print("\n" + "=" * 74)
    print("全部通过：第四轮新增表格的数字与归档一致。" if not FAIL
          else f"仍有 {len(FAIL)} 项不一致：{FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
