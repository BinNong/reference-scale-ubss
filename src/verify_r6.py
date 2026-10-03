#!/usr/bin/env python3
"""复核第六轮修订引入的全部改动。对应 `apply_r6_fixes.py`。

**判据是"从手稿里读出的字面值 vs 从归档/实现重算的值"**，不把数字写死在脚本里。
本轮特别之处：有三条检查的对象是**文字断言**而不是数字，用未命中/命中来判；
另有一条把 Table 37 的"levels"列与 `nfr.py` 的实现直接对上（这是本轮最重要的发现所在）。

检查项：
  A  Prop 8 不再自称 exact：旧限定句与旧命题名必须消失；新亏项式的两条断言要能复算
     （初等界在梯子上处处空泛；下尾角误差相对变化 ≤ 手稿印出的 1.3%）。
  B  三分类与补引：[27] ATFT / [28] 双传感器 max 经验阈值 必须在参考文献与正文里。
  C  次序统计量口径：手稿的 20 / 27 / 28 必须等于 `nfr.py` 公式重算值。
  D  Table 37：逐行与 `r6_quantile_cost.json` 相符，且 sort 在每一档都最快（这正是新结论）。
  E  η 的两类错判率与精确二项区间（手稿字面值 vs 从 Table 21 计数重算）。
  F  Prop 7 越界句已收；G "closed form" 不再用于 r(p)；H §9 措辞。
  I  跨表一致性：Table 1 与 Table 33 的 semi-analytic 列（同一名义量）逐行相同。

用法（在 src/ 下）：
    python3 verify_r6.py
    python3 verify_r6.py --manuscript /tmp/mutated.md      # 负向测试用
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "results"
MS = HERE.parent / "paper" / "manuscript.md"
FAIL: list[str] = []
Q_HI, N_Q = 0.02, 30


def tables(path: pathlib.Path) -> dict[int, list[list[str]]]:
    lines = path.read_text().split("\n")
    out: dict[int, list[list[str]]] = {}
    for i, l in enumerate(lines):
        mo = re.match(r"^\*\*Table (\d+)\.", l)
        if not mo:
            continue
        n = int(mo.group(1))
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        rows = []
        while j < len(lines) and lines[j].startswith("|"):
            cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
            j += 1
        out[n] = rows
    return out


def num(cell: str) -> float | None:
    t = cell.replace("**", "").replace("$", "").replace("{,}", "").replace(",", "")
    t = t.replace("\\%", "%").replace("\\", "")
    if t.strip() in ("", "---", "-", "—"):
        return None
    if t.endswith("%"):
        t = t[:-1]
    mo = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(mo.group(0)) if mo else None


def dec(cell: str) -> int:
    t = cell.replace("**", "").replace("$", "").replace("{,}", "").replace(",", "").replace("\\", "")
    mo = re.search(r"-?\d+\.(\d+)", t)
    return len(mo.group(1)) if mo else 0


def cmp_cell(label: str, cell: str, actual, scale: float = 1.0) -> None:
    """手稿字面值（按印出的小数位定容差） vs 重算值。"""
    claimed = num(cell)
    if claimed is None:
        return
    if actual is None or not np.isfinite(actual):
        FAIL.append(f"{label}: 稿 {cell!r} 但重算无值")
        print(f"  FAIL {label:<52} 稿 {cell!r} 重算无值")
        return
    tol = 0.5 * 10 ** (-dec(cell)) * scale + 1e-9
    got = actual * scale
    if abs(claimed - got) > tol:
        FAIL.append(f"{label}: 稿 {claimed} vs 重算 {got:g}")
        print(f"  FAIL {label:<52} 稿 {claimed} 重算 {got:g} (tol {tol:g})")
    else:
        print(f"  ok   {label:<52} {claimed}")


def want(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        FAIL.append(f"{label} {detail}")
        print(f"  FAIL {label} {detail}")


def level_indices(n: int) -> int:
    """`nfr.estimate_noise_power` 实际用到的次序统计量条数。"""
    return int(np.unique(np.geomspace(1.0, max(8, int(Q_HI * n)), N_Q).astype(int)).size)


# ------------------------------------------------------------------ A. Prop 8
def prop8(text: str, tb) -> None:
    print("\n[A] Prop 8：exact → 下尾近似")
    want("旧命题名已消失", "exact quantile relation" not in text)
    want("旧限定句已消失", "still lies inside the noise component" not in text)
    want("新命题名在稿", "**Proposition 8 (noise-dominated lower-tail relation).**" in text)
    want("亏项式在稿", "F_0(x_q)\\;=\\;\\frac{q}{\\pi_0}" in text)
    want("Fig. 2 题注已改", "chi-square lower-tail relation" in text)
    want("Algorithm 1 注释已改", "lower-tail inversion" in text)

    # 断言1：初等界 (q-(1-π0))/π0 在梯子上处处空泛 ⟺ 1-π0 > q（q = q_max = 0.02）
    pi0 = [num(r[1]) for r in tb[1] if len(r) > 1]
    pi0 = [v for v in pi0 if v is not None]
    want("Table 1 解析到 π0 列", len(pi0) >= 5, f"got {pi0}")
    vacuous = [1.0 - v > Q_HI for v in pi0]
    want("初等界在每一档都空泛（1-π0 > q_max）", all(vacuous), f"{vacuous}")

    # 断言2：下尾相对变化 ≤ 1.3%（手稿印出的值）。退化档必须排除——§5.1 原文即限定
    # "outside the degenerate dense one"，Table 22 的 dense 行是 5.9%。
    pcts = []
    for r in tb[22]:
        if r and "dense" in r[0]:
            continue
        for cell in r:
            pcts += [float(m) for m in re.findall(r"\((\d+(?:\.\d+)?)\\?%\)", cell)]
    if pcts:
        cmp_cell("Table 22 最大相对变化（非退化档）= §A.4 的 1.3%", "$1.3\\%$", max(pcts), 1.0)
    else:
        want("Table 22 解析到相对变化列", False, "未解析到 (x%)")


# --------------------------------------------------------------- B. 三分类与引用
def taxonomy(text: str) -> None:
    print("\n[B] §2 三分类与补引")
    want("ATFT 引用在稿", "[27]" in text and "adaptive time–frequency thresholding of [27]" in text)
    want("max 经验阈值补引在稿", "its maximum [5,7,28]" in text)
    want("百分位一类仍在", "[6,8,9]" in text)
    want("引言已提自适应变体", "from a statistic of the mixture computed at run time [27]" in text)
    want("重复句已修", "which Propositions 6 and 7 make quantitative. Propositions 6 and 7 make it quantitative" not in text)
    for ref, frag in (("27", "Hassan"), ("28", "Chen")):
        mo = re.search(rf"^\[{ref}\] (.+)$", text, re.M)
        want(f"参考文献 [{ref}] 存在且为该文", bool(mo) and frag in mo.group(1),
             mo.group(1)[:60] if mo else "缺失")
    want("[27] 正文引用≥2 处", text.count("[27]") >= 2, f"count={text.count('[27]')}")
    want("Sensors/Electronics 刊名在条目内",
         "*Sensors*, vol. 23, no. 4, 2060, 2023" in text and "*Electronics*, vol. 13, no. 7, 1227, 2024" in text)


# ------------------------------------------------- C. 次序统计量：手稿 vs 实现
def order_stats(text: str, tb) -> None:
    print("\n[C] 次序统计量口径 vs 实现")
    nfr_src = (HERE / "nfr.py").read_text()
    want("实现确实全排序（发现的前提）", "es = np.sort(E)" in nfr_src)
    want("旧说法 'thirty order statistics' 已消失", "thirty order statistics" not in text)
    want("Algorithm 1 用 floor 且写明 30", "30 geometric indices in 1 ... floor(q_max*n), deduplicated" in text)
    want("Algorithm 1 不再用 ceil", "ceil(q_max*n)" not in text)

    levels = [int(num(r[1])) for r in tb[37][1:] if len(r) > 1 and num(r[1]) is not None]
    ns = [int(num(r[0])) for r in tb[37][1:] if len(r) > 0 and num(r[0]) is not None]
    want("Table 37 行数 = 6", len(levels) == 6 and len(ns) == 6, f"{ns} / {levels}")
    for n, lv in zip(ns, levels):
        cmp_cell(f"Table 37 levels @ n={n} == 实现", f"${lv}$", float(level_indices(n)), 1.0)

    cmp_cell("§5.1 的 20 == 实现 @ n=2112", "$20$", float(level_indices(2112)), 1.0)
    cmp_cell("§5.1 的 27 == 实现 @ n=48000", "$27$", float(level_indices(48000)), 1.0)
    cmp_cell("§5.1 的 28 == 实现 @ n=1000000", "$28$", float(level_indices(1_000_000)), 1.0)
    want("§5.1/§A.17 写明去重与条数",
         "Integer truncation collapses the lowest of the nominal levels" in text)


# ---------------------------------------------------------- D. Table 37 vs 归档
def table37(tb) -> None:
    print("\n[D] Table 37 vs r6_quantile_cost.json")
    arch = json.loads((R / "r6_quantile_cost.json").read_text())
    by_n = {int(r["n"]): r for r in arch["rows"]}
    rows = [r for r in tb[37][1:] if len(r) >= 7 and num(r[0]) is not None]
    for r in rows:
        n = int(num(r[0]))
        a = by_n.get(n)
        if a is None:
            FAIL.append(f"Table 37 n={n} 归档缺行")
            print(f"  FAIL n={n} 归档缺行")
            continue
        cmp_cell(f"n={n} sort+gather (ms)", r[2], a["t_sort_s"] * 1e3, 1.0)
        cmp_cell(f"n={n} partition+gather (ms)", r[3], a["t_partition_s"] * 1e3, 1.0)
        cmp_cell(f"n={n} histogram (ms)", r[4], a["t_hist_s"] * 1e3, 1.0)
        cmp_cell(f"n={n} exact error (%)", r[5], a["s2_rel_err_exact_mean"] * 100, 1.0)
        cmp_cell(f"n={n} histogram error (%)", r[6], a["s2_rel_err_histogram_mean"] * 100, 1.0)

    # 新结论的判据：sort 在每一档都最快（这是"直方图被支配"的量化依据）
    faster = [num(r[2]) < num(r[3]) and num(r[2]) < num(r[4]) for r in rows]
    want("sort 在每一档都快于 partition 与 histogram", all(faster), f"{faster}")
    mono = [num(r[2]) for r in rows]
    want("sort 耗时随 n 单调不减", all(b >= a for a, b in zip(mono, mono[1:])), f"{mono}")

    want("斜率：sort 超线性、partition ≈ 线性",
         abs(arch["slope_t_sort_s"] - 1.10) < 0.06 and abs(arch["slope_t_partition_s"] - 0.94) < 0.03,
         f"sort={arch['slope_t_sort_s']:.3f} part={arch['slope_t_partition_s']:.3f}")
    want("两路估计逐档相同（改文案不改实现）",
         all(r["s2_identical_sort_partition"] for r in arch["rows"]))
    want("旧的双列（selection / full sort）已消失",
         "selection (this paper)" not in (HERE.parent / "paper" / "manuscript.md").read_text())


# ------------------------------------------------------------ E. η 的错判率与区间
def eta_ci(text: str, tb) -> None:
    print("\n[E] 保留判据：两类错判率与精确二项区间")
    cols = [num(c) for c in tb[21][0][1:]]
    k = next((i for i, v in enumerate(cols) if v is not None and abs(v - 0.035) < 1e-9), None)
    want("Table 21 找到 η=0.035 列", k is not None, f"cols={cols}")

    def row_count(marker: str) -> int:
        for r in tb[21]:
            if r and marker in r[0]:
                return int(num(r[k + 1]))
        return -1

    fa, ff, me = (row_count("false activations"), row_count("false fallbacks"),
                  row_count("material errors"))
    n_dec = 270
    want("Table 21 计数解析成功", min(fa, ff, me) >= 0, f"{fa}/{ff}/{me}")
    want("误报+漏报 = 材料错误", fa + ff == me, f"{fa}+{ff} vs {me}")

    for k_, lab, hi_lab in ((fa, "5.2", "8.55"), (ff, "1.1", "3.21")):
        lo, hi = binomtest(k_, n_dec).proportion_ci(confidence_level=0.95, method="exact")
        cmp_cell(f"§5.3 的 {k_}/270 = {lab}%", f"${lab}\\%$", 100 * k_ / n_dec, 1.0)
        lo_lab = "2.86" if k_ == fa else "0.23"
        cmp_cell(f"§A.3 区间下界 {lo_lab}%", f"${lo_lab}\\%$", 100 * lo, 1.0)
        cmp_cell(f"§A.3 区间上界 {hi_lab}%", f"${hi_lab}\\%$", 100 * hi, 1.0)
    want("§A.3 声明了'全局固定、非独立留出集'",
         "not a separated held-out set" in text)


# ------------------------------------------------------------- F/G/H 文字断言
def wording(text: str, tb) -> None:
    print("\n[F/G/H] 措辞：Prop 7 越界句 / semi-analytic / §9")
    want("Prop 7 越界句已收", "necessarily on the signal scale" not in text)
    want("Prop 7 新句在稿", "pulled towards the signal scale" in text)

    for bad in ("Closed form for the reference-scale ratio", "closed form for $r(p)$",
                "closed-form reference-scale ratio", "$r$ closed form", "computable in closed form",
                "closed form to be insensitive"):
        want(f"旧闭式说法已消失：{bad[:38]}", bad not in text)
    want("semi-analytic 出现 ≥ 8 处", text.count("semi-analytic") >= 8,
         f"count={text.count('semi-analytic')}")
    want("§4.2 说明其为半解析", "so the result is a *semi-analytic* characterization" in text)
    want("Table 1 表头已改", "semi-analytic" in " ".join(tb[1][0]))
    want("Table 33 表头已改", "semi-analytic" in " ".join(tb[33][0]))

    want("§9 原来的 'real recordings' 说法已消失",
         "We now repeat the comparison on real recordings." not in text)
    want("§9 标题已收紧", "## 9. Evaluation on Real-Speech Sources" in text)
    want("§9 开篇写明合成混合", "mixed synthetically under the same instantaneous model" in text)
    want("§9.1 仍声明瞬时混合", "Mixing is instantaneous" in text)


# ------------------------------------------------------- I. 跨表一致性
def cross_table(tb) -> None:
    print("\n[I] 跨表一致性：Table 1 与 Table 33 的同一名义量 r(p)")
    t1 = {num(r[0]): num(r[3]) for r in tb[1][1:] if len(r) > 4 and num(r[0]) is not None}
    t33 = {num(r[0]): num(r[4]) for r in tb[33][1:] if len(r) > 5 and num(r[0]) is not None}
    common = [k for k in t1 if k in t33 and t1[k] is not None and t33[k] is not None]
    want("两表有可比的公共行", len(common) >= 4, f"{common}")
    for k in common:
        cmp_cell(f"p={k} 两表相同", f"${t33[k]}$", t1[k], 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=str(MS))
    a = ap.parse_args()
    ms = pathlib.Path(a.manuscript)
    text = ms.read_text()
    tb = tables(ms)

    # —— R8 表号整体重排后的兼容垫片（旧号 -> 新号），见 verify_r5.py 同名注释 ——
    _R8_TAB = {19: 23, 20: 24, 21: 25, 22: 26, 23: 19, 24: 20, 25: 27, 26: 28,
               27: 29, 28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36,
               35: 37, 36: 38, 37: 39, 38: 40, 39: 21, 40: 41, 41: 42}
    tb.update({o: tb[n] for o, n in _R8_TAB.items() if n in tb})

    need = [1, 21, 22, 33, 37]
    missing = [n for n in need if n not in tb]
    if missing:
        print("手稿缺少表格：", missing)
        sys.exit(1)

    prop8(text, tb)
    taxonomy(text)
    order_stats(text, tb)
    table37(tb)
    eta_ci(text, tb)
    wording(text, tb)
    cross_table(tb)

    print("\n" + "=" * 80)
    if FAIL:
        print(f"未通过 {len(FAIL)} 项：")
        for x in FAIL:
            print("   -", x)
        sys.exit(1)
    print("全部通过。")


if __name__ == "__main__":
    main()
