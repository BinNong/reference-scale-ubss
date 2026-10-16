#!/usr/bin/env python3
"""README 的两张映射表自检：Records→scripts 与 Figures→scripts 必须和稿件现状一致。

动机：稿件 `paper/manuscript.md:418` 与 Declarations 明确承诺
"The repository README names every record and maps each table and figure to the script and
record that produce it"。R14 系列把表号整体顺延过三次（≥23 减 4，随后 ≥13 减 1），
README 的两张表却停在 R8 之前——`r2_runtime.json` 标着 Table 27（现为"参考尺度分类"）、
`scale_invariance.json` 标着 Fig. 7 + §8.8（现为 Fig. 11，§8.8 已不存在）。稿件侧的承诺
因此对不上，而 `check_refs.py` 只读稿件、`r15_recount.py` 只对数，都没有一层看 README。

本脚本读 `r15_recount.py` 产出的 record→tables 映射，和 README 表格里的表号对照。

用法（从仓库根运行）：

    python3 src/check_readme_map.py            # 对照
    python3 src/check_readme_map.py --fix-hint # 不一致时打印建议写法
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
README = os.path.join(ROOT, "README.md")
RECOUNT = os.path.join(ROOT, "src", "r15_recount.py")
RECOUNT_JSON = os.path.join(ROOT, "results", "r15_recount.json")


def read_records_table(md: str) -> dict[str, str]:
    """README 的 Records → scripts 表：record -> 'Produces' 列原文。"""
    i = md.index("### Records → scripts")
    j = md.index("### Figures → scripts")
    out = {}
    for line in md[i:j].split("\n"):
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        name = cells[0].strip("`")
        if name.endswith(".json"):
            name = name[: -len(".json")]        # truth 的键不带扩展名
        out[name] = cells[2]
    return out


def tables_in(text: str) -> set[int]:
    """'Tables 6–12' / 'Table 45' / 'Tables 13–14, 36–37' -> {6,...,12,45,...}。"""
    nums: set[int] = set()
    for m in re.finditer(r"Tables?\s+((?:\d+(?:\s*(?:,|and|–|—|-|to)\s*\d+)*)+)", text):
        blob = m.group(1)
        # 先展开 "6–12" 这类区间
        for a, b in re.findall(r"(\d+)\s*(?:–|—|-|to)\s*(\d+)", blob):
            nums.update(range(int(a), int(b) + 1))
        blob2 = re.sub(r"\d+\s*(?:–|—|-|to)\s*\d+", "", blob)
        nums.update(int(x) for x in re.findall(r"\d+", blob2))
    return nums


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-hint", action="store_true")
    a = ap.parse_args()

    with open(README, encoding="utf-8") as fh:
        md = fh.read()
    readme = read_records_table(md)

    # record -> tables：优先用已生成的 JSON，否则现算
    if os.path.exists(RECOUNT_JSON):
        with open(RECOUNT_JSON) as fh:
            truth = {k: sorted(v) for k, v in json.load(fh)["record_tables"].items()}
    else:
        r = subprocess.run([sys.executable, RECOUNT, "--json", RECOUNT_JSON],
                           capture_output=True, text=True)
        if r.returncode not in (0, 1):
            print(r.stdout[-2000:] + r.stderr[-2000:])
            return 2
        with open(RECOUNT_JSON) as fh:
            truth = {k: sorted(v) for k, v in json.load(fh)["record_tables"].items()}

    bad = []
    for name, produces in sorted(readme.items()):
        claimed = tables_in(produces)
        actual = set(truth.get(name, []))
        if not actual:
            continue                       # 该记录不支撑任何表（provenance / 被取代）
        # 稿件里显式点名该记录的附录节（如 A.15）不算表号，可忽略
        missing = actual - claimed
        extra = claimed - actual
        if missing or extra:
            bad.append((name, sorted(claimed), sorted(actual), sorted(missing), sorted(extra)))

    print(f"README 记录表 {len(readme)} 行；r15_recount 派生 {len(truth)} 条映射")
    if not bad:
        print("✓ 一致：README 的每个表号都与记录算出的相符")
        return 0
    print(f"\n✗ {len(bad)} 条记录的表号与现状不符：")
    for name, claimed, actual, missing, extra in bad:
        print(f"  · {name}: README 写 Tables {claimed or '—'}，实为 {actual or '—'}"
              + (f"（缺 {missing}）" if missing else "")
              + (f"（多 {extra}）" if extra else ""))
        if a.fix_hint:
            print(f"      Produces 列应含 Tables {actual}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
