r"""R11：把 Funding 声明从占位符改为"未获得资助"的正式声明。

R10e 建 Declarations 时，`**Funding.**` 一行留的是占位符 `[funding information to be inserted]`，
并在 `verify_r10` 的 G 组把它断言为"投稿前须填"。作者确认**本项目没有基金资助**，
于是按 Springer 的标准写法如实声明，而不是删掉该项——Declarations 的 Funding 是必填项，
写"无资助"是完整声明，留空或删节才是缺口。

措辞取 Springer Nature 作者指南给出的 no-funding 模板句式（"The authors declare that no
funds, grants, or other support were received..."），本稿把它限定到三个具体环节
(preparation / study / publication)，与 Ethics、Competing interests 两项的语域一致。

**只改一行，不动任何数字、不触发表号与文献编号。**

用法：
    python3 src/apply_r11_funding.py --dry
    python3 src/apply_r11_funding.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")

OLD = r"**Funding.** [funding information to be inserted]"
NEW = (r"**Funding.** The authors declare that no funds, grants, or other support were received"
       "\n" r"during the preparation, study or publication of this article.")

E = [("Declarations 的 Funding 一项：占位符 → 无资助正式声明", OLD, NEW)]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args()
    p = pathlib.Path(a.manuscript) if a.manuscript else MS
    s = p.read_text()
    changed = skipped = missing = 0

    for tag, old, new in E:
        n = s.count(old)
        if n == 0:
            if new in s:
                print(f"  [已改，跳过] {tag}")
                skipped += 1
                continue
            print(f"  ✗ 未命中（原文不存在）：{tag}")
            missing += 1
            continue
        if n > 1:
            print(f"  ✗ 原文不唯一（{n} 次）：{tag}")
            missing += 1
            continue
        if old in new:
            die(f"条目 {tag!r} 的 new 包含完整的 old —— 会破坏幂等")
        s = s.replace(old, new, 1)
        print(f"  ✓ {tag}")
        changed += 1

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # ------------------------------------------------------------------ 终检
    t = p.read_text()
    print("\n=== 终检 ===")
    for probe, want_ok in [
        ("[funding information to be inserted]", False),
        ("**Funding.** The authors declare that no funds, grants, or other support were received", True),
        ("during the preparation, study or publication of this article.", True),
    ]:
        ok = (probe in t) == want_ok
        print(f"  {'OK  ' if ok else 'FAIL'} {'存在' if want_ok else '不存在'}：{probe[:62]}")
        if not ok:
            die("终检失败")

    if t.count("**Funding.**") != 1:
        die(f"`**Funding.**` 应恰为 1 次，实为 {t.count('**Funding.**')}")
    print("  OK   `**Funding.**` 恰为 1 次")

    # Declarations 五项齐全——Funding 改成正式声明后仍须是完整的一套
    for item in ("**Data availability.**", "**Code availability.**", "**Ethics.**",
                 "**Competing interests.**", "**Funding.**"):
        if item not in t:
            die(f"Declarations 缺项：{item}")
    print("  OK   Declarations 五项齐全")

    refs = [int(x) for x in re.findall(r"(?m)^\[(\d+)\]", t)]
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", t)]
    for lab, seq in (("文献", refs), ("表", caps), ("附录小节", secs), ("图", figs)):
        ok = seq == list(range(1, len(seq) + 1))
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}：共 {len(seq)}，1..{max(seq)} 连续无重复")
        if not ok:
            die(f"{lab}编号不连续或有重复：{seq}")
    print("\n✓ 全部终检通过")


if __name__ == "__main__":
    main()
