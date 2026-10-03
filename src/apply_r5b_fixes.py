#!/usr/bin/env python3
"""第五轮修订的**补正**：由 `verify_r5.py` 的复核结果驱动的四类改动。

1. Table 39 重写。两个 `conventional median` 掩码原先**只有能量判据、没有共线+均衡**，
   与全稿其它表中同一名称的定义（Tables 6/15/19）不一致，于是"门限严格程度"这一比较里
   混进了"有没有共线判据"这个无关变量；`points kept` 一列也没有任何口径能复现。
   修法是统一加 base 后重跑 `r5_source_number.py`（6.9 s），本脚本把新值与新结论落稿。
2. 三处**四舍五入错误**（复核器抓到）：Table 17 的 max/N=5、Table 36 的 2.14、Table 35 的 96.2%。
3. Table 17 新增两列的**小数位不规范**（0.6 / 0.7 / 1.0 / 1.2 / 0.9）统一为两位。
4. §7 的可复现性段补入 R5 的 8 个归档（此前只列到 R4）。

改完必须重跑 `verify_r5.py`（它直接解析手稿表格，是真正的回归测试）。
用法：  python3 apply_r5b_fixes.py [--dry]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

E: list[tuple[str, str, str]] = []


def add(tag: str, old: str, new: str) -> None:
    E.append((tag, old, new))


# --------------------------------------------------------- 1. Table 39 重写
add("Table 39 caption",
"""**Table 39.** Automatic order selection from the mask. The potential-function peak detector (the same criteria as in the baselines: angle grid, local maxima, significance threshold $\\rho>0.2\\rho_{\\max}$, near-peak suppression) is applied to the direction set each mask produces, on the ladder and on $N\\in\\{3,\\dots,6\\}$ cells (ten seeds each, $960$ decisions). "exact" is the fraction for which $\\hat N=N$; the last column is the angle error obtained by clustering with the *estimated* number of columns, so that a missing column is penalised.""",
"""**Table 39.** Automatic order selection from the mask. The potential-function peak detector (the same criteria as in the baselines: angle grid, local maxima, significance threshold $\\rho>0.2\\rho_{\\max}$, near-peak suppression) is applied to the direction set each mask produces, on the ladder and on $N\\in\\{3,\\dots,6\\}$ cells (ten seeds each, $960$ decisions). All four masks share the same collinearity-plus-balance base, so that the rows differ only in the energy criterion — the two median rows add $e>t_e\\,\\mathrm{median}(e)$ to it, which is the rule as it appears in Tables 6, 15 and 19. "exact" is the fraction for which $\\hat N=N$; "angle error at $\\hat N$" is obtained by clustering with the *estimated* number of columns, so that a missing column is penalised; "points kept" is the fraction of TF points the mask retains, averaged over the same decisions.""")

add("Table 39 body",
"""| collinearity + balance only | 29.2% | 1.27 | 42.5% | 28.3% | 15.12 | 19.2% |
| conventional median $t_e{=}0.02$ | 20.8% | 1.68 | 18.3% | 60.8% | 10.59 | 91.0% |
| conventional median $t_e{=}5$ | 45.0% | 0.79 | 40.8% | 14.2% | 14.64 | 14.6% |
| **NF-SSP** | **44.2%** | **0.79** | 54.2% | **1.7%** | 15.61 | 19.4% |""",
"""| collinearity + balance only | 29.2% | 1.27 | 42.5% | 28.3% | 15.12 | 26.9% |
| conventional median $t_e{=}0.02$ | 29.2% | 1.27 | 42.5% | 28.3% | 15.14 | 26.3% |
| conventional median $t_e{=}5$ | 40.8% | 0.88 | 55.0% | 4.2% | 18.16 | 11.4% |
| **NF-SSP** | **44.2%** | **0.79** | 54.2% | **1.7%** | 15.61 | 19.6% |""")

add("Table 39 后的结论段",
"""The gate does change the answer, and in a specific direction. The conventional default, which as
Section 7 notes is effectively inactive, over-counts in $61\\%$ of the decisions — the noise it leaves
in creates spurious peaks — while the calibrated gate almost never over-counts ($1.7\\%$) and instead
under-counts in $54\\%$ of them, so that the two masks bracket the truth from opposite sides and reach
about the same exact-order rate by different routes ($20.8\\%$ against $44.2\\%$). A stricter median
multiple mimics the calibrated gate's bias. The consequence for practice is not the exact-order rate
but the last column: because a missing column costs $90°$ in the matching metric while a spurious one
costs the angle of its nearest counterpart, under-counting is the more expensive error, and clustering
at $\\hat N$ is better for the conventional default than for the calibrated gate ($10.59°$ against
$15.61°$) even though the latter estimates the order better. Order selection should therefore not be
run on the calibrated gate's output: the useful division of labour is to estimate the order from the
ungated or lightly gated set — where the unweighted collinearity mask and the relaxed median rule are
both competitive — and to apply the gate afterwards to the directions that are to be clustered. We
report this as a measurement of an interaction rather than as a solution, and it is one more reason
why the paper supplies $N$ to every method rather than claiming an end-to-end pipeline.""",
"""The gate does change the answer, and in a specific direction. The conventional default is
*indistinguishable from removing the energy criterion altogether* — same exact-order rate, same mean
error, same over-counting — because it retains $26.3\\%$ of the points against $26.9\\%$ for the ungated
set: this is what "effectively inactive" looks like when it is measured rather than asserted. The
calibrated gate is the only rule that nearly eliminates over-counting, from $28.3\\%$ to $1.7\\%$, and
it pays for that by under-counting in $54.2\\%$ of decisions; a stricter median multiple behaves the
same way ($4.2\\%$ against $55.0\\%$). The consequence for practice is not the exact-order rate but the
angle error at $\\hat N$: because a missing column costs $90°$ in the matching metric while a spurious
one costs the angle of its nearest counterpart, under-counting is the more expensive error, and
clustering at $\\hat N$ is better on the ungated and default sets ($15.12°$ and $15.14°$) than on the
calibrated one ($15.61°$), while the stricter median is worst of the four ($18.16°$). Order selection
should therefore not run on the calibrated gate's output: the useful division of labour is to estimate
the order from the ungated or lightly gated set, and to apply the gate afterwards to the directions
that are to be clustered. We report this as a measurement of an interaction rather than as a
solution, and it is one more reason why the paper supplies $N$ to every method rather than claiming
an end-to-end pipeline.""")

# --------------------------------------------- 2. 三处四舍五入错误（复核器抓到）
add("Table 17: max c=0.05 的 N=5 行 1.13 → 1.12",
    "| $N=5$ | 1.37 | 0.81 | 0.79 | 0.81 | 8.59 | 1.13 | 1.11 |",
    "| $N=5$ | 1.37 | 0.81 | 0.79 | 0.81 | 8.59 | 1.12 | 1.11 |")
add("Table 36: p=.20/SNR10/α=1e-1 的 2.14 → 2.13",
    "| 1.59 | 14.58% / 0.276 / 2.14 |",
    "| 1.59 | 14.58% / 0.276 / 2.13 |")
add("Table 35: median 行保留率 96.2% → 96.1%",
    "| conventional median $t_e{=}0.02$ | 16.03 | 9.27 | 7.60 | 5.40 | 6.92 | 18.83 | 10.68 | 11.75 | 96.2% |",
    "| conventional median $t_e{=}0.02$ | 16.03 | 9.27 | 7.60 | 5.40 | 6.92 | 18.83 | 10.68 | 11.75 | 96.1% |")

# ------------------------------------------- 3. Table 17 新增两列的小数位统一
add("Table 17: 小数位规范化（两列新增列全部两位）",
"""| win 256 | 1.03 | 1.01 | 1.01 | 1.02 | 8.75 | 1.45 | 1.26 |
| win 512 | 0.76 | 0.60 | 0.59 | 0.60 | 9.56 | 0.88 | 0.75 |
| win 1024 | 0.78 | 0.45 | 0.45 | 0.45 | 8.78 | 0.58 | 0.69 |
| win 2048 | 0.84 | 0.70 | 0.69 | 0.70 | 9.59 | 0.65 | 0.92 |
| $N=3$ | 0.37 | 0.44 | 0.44 | 0.44 | 9.64 | 0.61 | 0.6 |
| $N=5$ | 1.37 | 0.81 | 0.79 | 0.81 | 8.59 | 1.12 | 1.11 |
| $N=6$ | 1.85 | 1.01 | 1.00 | 1.02 | 8.43 | 1.04 | 1.2 |
| SNR $0$ dB | 1.41 | 0.86 | 0.85 | 7.32 | 14.75 | 1.06 | 0.98 |
| SNR $10$ dB | 0.74 | 0.54 | 0.54 | 0.56 | 11.29 | 0.7 | 0.72 |
| SNR $30$ dB | 0.75 | 0.43 | 0.43 | 0.43 | 4.68 | 0.63 | 0.62 |
| SNR $40$ dB | 0.75 | 0.42 | 0.42 | 0.43 | 1.37 | 0.69 | 0.63 |
| 1 dense source | 1.13 | 0.83 | 0.82 | 0.84 | 10.77 | 0.98 | 1.14 |
| 2 dense sources | 1.56 | 1.49 | 0.90 | 1.45 | 14.09 | 1.19 | 1.64 |
| babble noise | 2.62 | 0.80 | 0.79 | 0.80 | 2.75 | 0.9 | 1.0 |
| noise-free | 0.79 | 0.44 | 0.43 | 0.44 | 0.86 | 0.7 | 0.64 |""",
"""| win 256 | 1.03 | 1.01 | 1.01 | 1.02 | 8.75 | 1.45 | 1.26 |
| win 512 | 0.76 | 0.60 | 0.59 | 0.60 | 9.56 | 0.88 | 0.75 |
| win 1024 | 0.78 | 0.45 | 0.45 | 0.45 | 8.78 | 0.58 | 0.69 |
| win 2048 | 0.84 | 0.70 | 0.69 | 0.70 | 9.59 | 0.65 | 0.92 |
| $N=3$ | 0.37 | 0.44 | 0.44 | 0.44 | 9.64 | 0.61 | 0.60 |
| $N=5$ | 1.37 | 0.81 | 0.79 | 0.81 | 8.59 | 1.12 | 1.11 |
| $N=6$ | 1.85 | 1.01 | 1.00 | 1.02 | 8.43 | 1.04 | 1.20 |
| SNR $0$ dB | 1.41 | 0.86 | 0.85 | 7.32 | 14.75 | 1.06 | 0.98 |
| SNR $10$ dB | 0.74 | 0.54 | 0.54 | 0.56 | 11.29 | 0.70 | 0.72 |
| SNR $30$ dB | 0.75 | 0.43 | 0.43 | 0.43 | 4.68 | 0.63 | 0.62 |
| SNR $40$ dB | 0.75 | 0.42 | 0.42 | 0.43 | 1.37 | 0.69 | 0.63 |
| 1 dense source | 1.13 | 0.83 | 0.82 | 0.84 | 10.77 | 0.98 | 1.14 |
| 2 dense sources | 1.56 | 1.49 | 0.90 | 1.45 | 14.09 | 1.19 | 1.63 |
| babble noise | 2.62 | 0.80 | 0.79 | 0.80 | 2.75 | 0.90 | 1.00 |
| noise-free | 0.79 | 0.44 | 0.43 | 0.44 | 0.86 | 0.70 | 0.64 |""")

# --------------------------------------------------- 4. §7 补入 R5 的归档清单
add("§7 归档清单补入 R5",
"""`r4_table2.json` for Table 2, and `r4_floor_bias.json` for the pure-noise bias of the level convention. Code, the fixed configurations, the dependency list and every machine-readable record named above are available at [repository to be inserted]""",
"""`r4_table2.json` for Table 2, and `r4_floor_bias.json` for the pure-noise bias of the level convention. The fifth round adds a further $3{,}812$ records: `r5_density_baseline.json` ($960$) and `r5_density_real.json` ($256$) for the head-to-head density comparison of Table 31 and Table 32, `r5_complex.json` ($180$) for the complex- and convolutive-mixing study of Tables 33--35, `r5_alpha_cell.json` ($240$) for the $\\alpha$ sensitivity of Table 36, `r5_complexity.json` for the quantile step of Table 37, `r5_band_floor.json` ($256$) for the banded floor of Table 38, `r5_source_number.json` ($960$) for the order selection of Table 39, and `r5_unified_weight.json` ($960$) for the gate-plus-weight design of Section 9.5 and Fig. 11. Code, the fixed configurations, the dependency list and every machine-readable record named above are available at [repository to be inserted]""")


# ------------------- 5. 复核器第二轮抓到的四处四舍五入错（Table 17 一处 / Table 37 三处）
add("Table 17: 2 dense sources 的 max+same weight 1.64 → 1.63",
    "| 2 dense sources | 1.56 | 1.49 | 0.90 | 1.45 | 14.09 | 1.19 | 1.64 |",
    "| 2 dense sources | 1.56 | 1.49 | 0.90 | 1.45 | 14.09 | 1.19 | 1.63 |")
add("Table 37: n=2112 全排序 0.08 → 0.07 ms",
    "| $2{,}112$ | 0.30 ms | 0.08 ms | 0.13 ms | 142 | 1.022 | 1.030 |",
    "| $2{,}112$ | 0.30 ms | 0.07 ms | 0.13 ms | 142 | 1.022 | 1.030 |")
add("Table 37: n=8192 直方图 σ̂²/σ² 0.999 → 0.998",
    "| $8{,}192$ | 0.61 ms | 0.37 ms | 0.25 ms | 75 | 1.059 | 0.999 |",
    "| $8{,}192$ | 0.61 ms | 0.37 ms | 0.25 ms | 75 | 1.059 | 0.998 |")
add("Table 37: n=524288 每点 69 → 68 ns",
    "| $524{,}288$ | 35.9 ms | 35.5 ms | 9.74 ms | 69 | 0.998 | 0.911 |",
    "| $524{,}288$ | 35.9 ms | 35.5 ms | 9.74 ms | 68 | 0.998 | 0.911 |")


# 第六轮（R6）重写了下列条目的目标文本（Prop 8 命名、Table 1 列名、§1 与 §9 措辞、
# Table 37 整体重测、§7 归档清单），这些条目本身已无需再执行；保留登记以免重跑该轮时
# 误判为需要改动。已核对：在 R6 之前的手稿上它们仍为 0 未命中（见 MEMORY）。
_SUPERSEDED_R6 = {
    '§7 归档清单补入 R5',
    'Table 37: n=2112 全排序 0.08 → 0.07 ms',
    'Table 37: n=8192 直方图 σ̂²/σ² 0.999 → 0.998',
    'Table 37: n=524288 每点 69 → 68 ns',
}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--dry", action="store_true", help="只报告将改动哪些条目")
    a = a.parse_args()

    s = MS.read_text()
    changed = skipped = missing = 0
    for tag, old, new in E:
        if tag in _SUPERSEDED_R6:
            print(f"  [R6 已承接] {tag}")
            skipped += 1
            continue
        if new in s:
            print(f"  [跳过] {tag}")
            skipped += 1
            continue
        if old not in s:
            print(f"  [未命中] {tag}")
            missing += 1
            continue
        s = s.replace(old, new, 1)
        changed += 1
        print(f"  [已改] {tag}")

    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    MS.write_text(s)
    print(f"\n已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # 结构自检：表号连续、`$` 配对
    L = s.split("\n")
    tabs = sorted(int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s))
    misfit = sum(1 for l in L if l.count("$") % 2 == 1 and not l.strip().startswith("$$"))
    print("表号:", "连续" if tabs == list(range(1, len(tabs) + 1)) else tabs,
          f"| 共 {len(tabs)} 张 | 行内 $ 失配 {misfit}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
