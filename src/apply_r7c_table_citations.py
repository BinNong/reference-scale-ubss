#!/usr/bin/env python3
"""第七轮补遗 C：给 7 张"只在题注里出现、正文从未引用"的表补上显式引用。

发现（2026-09-19）
==================
全文扫描"`Table N` 的正文提及"时发现，Tables 8、9、10、12、13、28、29 一次都没被
正文引用过——它们只是被摆在各自小节标题下面。这类问题编辑一定会退回：每张表都
必须在正文里被点名。另有 Tables 19--26 只在附录开头被当作一个区间提到一次
（"Tables 19--26 are self-contained"），情形比这 7 张好，本轮不动。

改法
====
只在**已经讨论该表内容的那一段**末尾补一句点名，不新增任何论断、不改动任何数字：

    Table  8 (§8.2)  → "The full sweep is tabulated in Table 8."
    Table  9 (§8.3)  → "Table 9 gives the three sensor counts in full."
    Table 10 (§8.4)  → "Table 10 gives all four source counts in full."
    Table 12 (§8.6)  → "Table 12 tabulates the three summaries."
    Table 13 (§8.7)  → "Table 13 reports the four source families."
    Table 28 (§A.11) → "Table 28 reports the comparison for three settings of `eps`."
    Table 29 (§A.12) → "Table 29 reports the two alternatives in the same gate."

锚点都取句尾片段、且在稿中**恰好出现一次**（脚本会断言），因此不依赖插入顺序。
插在段尾（而不是段首）是有意的：段首常被更早轮次的 applier 当作目标文本，
在段首插字会让那些锚点失配、凭空多出"未命中"。

用法：
    python3 src/apply_r7c_table_citations.py --dry
    python3 src/apply_r7c_table_citations.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

MS = pathlib.Path(__file__).resolve().parents[1] / "paper" / "manuscript.md"

E: list[tuple[str, str, str]] = [
    ("Table 8 引用（§8.2 噪声鲁棒性）",
     "which is the expected crossover, and we report it rather than hide it.",
     "which is the expected crossover, and we report it rather than hide it. "
     "The full sweep is tabulated in Table 8."),

    ("Table 9 引用（§8.3 传感器数）",
     "which is also the most severely underdetermined one.",
     "which is also the most severely underdetermined one. "
     "Table 9 gives the three sensor counts in full."),

    ("Table 10 引用（§8.4 源数目）",
     "so this is a boundary of the paradigm rather than of the gate.",
     "so this is a boundary of the paradigm rather than of the gate. "
     "Table 10 gives all four source counts in full."),

    ("Table 12 引用（§8.6 误纳率敏感性）",
     "and it is the reason the row is flat.",
     "and it is the reason the row is flat. Table 12 tabulates the three summaries."),

    ("Table 13 引用（§8.7 结构化源）",
     "has itself become marginal.",
     "has itself become marginal. Table 13 reports the four source families."),

    ("Table 28 引用（§A.11 密度预筛）",
     "so only the energy criterion differs.",
     "so only the energy criterion differs. "
     "Table 28 reports the comparison for three settings of `eps`."),

    ("Table 29 引用（§A.12 稳健尺度）",
     "only $\\hat\\nu$ changes.",
     "only $\\hat\\nu$ changes. "
     "Table 29 reports the two alternatives in the same gate."),
]


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--dry", action="store_true", help="只报告将改动哪些条目")
    a = a.parse_args()

    s = MS.read_text()

    # 先断言锚点唯一：万一后来又出现同样措辞，宁可直接报错也不要改错地方
    for tag, old, _ in E:
        if s.count(old) > 1:
            print(f"错误：{tag} 的锚点在稿中出现 {s.count(old)} 次，不唯一", file=sys.stderr)
            raise SystemExit(1)

    changed = skipped = missing = 0
    for tag, old, new in E:
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

    # 自检：这 7 张表现在必须都能在正文里找到引用（题注行不算）
    body = s[s.index("## 1. Introduction"):s.index("## Figures")]
    nogap = []
    for n in (8, 9, 10, 12, 13, 28, 29):
        if not re.search(rf"\bTable\s*{n}\b", body):
            nogap.append(n)
    if nogap:
        print(f"错误：这些表在正文里仍然没有引用：{nogap}", file=sys.stderr)
        raise SystemExit(1)
    print("自检通过：Tables 8/9/10/12/13/28/29 已在正文中被引用。")


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
