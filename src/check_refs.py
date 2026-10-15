"""全文交叉引用自检：悬空引用、编号断档、图号顺序、前向引用、节号、表格引用语义。

动机（2026-10-06）：Assumptions 块曾放在 §6，而 §4.2 的命题 6 在用它——读者读到那里时
A2 还没定义，且没有前向指针（R14d 已修）。同类问题只能靠系统扫描发现，于是有本脚本。
同族工具：`check_tables.py`（表格内 mean 行与列的一致性）、`r15_recount.py`（每个数字 vs 归档）、
各 `verify_r*.py`（数字 vs 归档）。

检查六类：
  A 悬空引用   —— 引用了不存在的 Proposition/Corollary/Table/Fig/Algorithm/Assumption；
  B 编号断档   —— 某族编号不是 1..N 连续（曾抓到 Proposition 5 被引 3 次却无定义）；
  C 图号顺序   —— 每条 Fig. N 必须在正文被引，且首次引用次序与题注号一致；
  D 节号/附录号 —— Section N.M 与 Appendix A.N 必须都有对应标题（结构搬迁后必查）。
                 **含枚举形式**：`Sections 8.6 and 8.8`、`Sections 4.1, 4.2 and 4.3`。
                 第一版只匹配 `Sections? N.M` 的单项，于是附录导言里
                 "moved from Sections 8.6 and 8.8" 里的 8.8（该节已不存在）整轮漏检；
  E 前向引用   —— 引用位置早于定义位置。§1/§2 里指后文属正常，报"提示"；
                 §3 及之后还出现，报"可疑"（判失败）；
  F 表格引用语义 —— `Table N` 出现处的上下文要含该表的关键词。编号连续性查不出
                 "一个语义正确的旧编号指向了内容不同的新表"——R14 三次顺延表号后，
                 稿件里曾有 11 处这种引用，A–E 全绿。这里用关键词清单兜底，报"提示"。

用法：
    python3 src/check_refs.py                    # 查 paper/manuscript.md
    python3 src/check_refs.py --manuscript <md>  # 查别的文件（负向测试用）
    python3 src/check_refs.py --strict           # 前向引用一律判失败
    python3 src/check_refs.py --semantic         # 表格语义自检升级为判失败
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

# 定义行的写法：加粗标签开头
FAMILIES = {
    "Proposition": r"\*\*Proposition (\d+)\s",
    "Corollary": r"\*\*Corollary (\d+)\s*[:(]",
    "Lemma": r"\*\*Lemma (\d+)\s*[:(]",
    "Theorem": r"\*\*Theorem (\d+)\s*[:(]",
    "Assumption": r"\*\*A(\d)\b",
    "Algorithm": r"\*\*Algorithm (\d+)\*\*",
}
# 引用时的写法（正文里出现的）
REF_PATTERNS = {
    "Proposition": r"\bPropositions? (\d+)",
    "Corollary": r"\bCorollary (\d+)",
    "Lemma": r"\bLemma (\d+)",
    "Theorem": r"\bTheorem (\d+)",
    "Assumption": r"\bA(\d)\b",
    "Algorithm": r"\bAlgorithm (\d+)",
}
# 前向引用出现在这些节里算正常（引言/相关工作指后文）
BENIGN_SECTIONS = ("1. Introduction", "2. Related Work", "Abstract")

# F 表格引用语义自检：拿**表题本身**当关键词源。一个正确的引用，其上下文应当与它
# 指向的那张表的标题共享内容词；编号连续性查不出「一个语义正确的旧编号指向了内容
# 不同的新表」——R14 三次顺延表号后稿件里曾有 11 处这种引用，A–E 全绿。
# 由表题派生而不是手写清单：清单会腐烂（第一版手写了 45 组关键词，17 组过窄产生假报）。
STOP = set("""the and for with that this from into over under than then when what which
whose where while were been have has had not are was will would could should each
also such very more most less least only same other another between across after
before during about against because both either neither upon within without
table figure section appendix shows show given uses use used using mean means
paper study report reports reporting here there their they them than find found
paper's value values level levels case cases""".split())


def _content_words(text: str) -> set[str]:
    """表题里的内容词：去 LaTeX、去停用词、长度 >= 4。"""
    t = re.sub(r"\$[^$]*\$", " ", text)            # 去行内公式
    t = re.sub(r"\\[a-zA-Z]+", " ", t)             # 去 LaTeX 命令
    t = re.sub(r"[^A-Za-z]+", " ", t).lower()
    return {w for w in t.split() if len(w) >= 4 and w not in STOP}


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(2)


def sections(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(0).lstrip("# ").strip())
            for m in re.finditer(r"(?m)^#{2,3} .*$", text)]


def section_of(pos: int, heads: list[tuple[int, str]]) -> str:
    cur = "(front matter)"
    for p, h in heads:
        if p <= pos:
            cur = h
        else:
            break
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=str(MS))
    ap.add_argument("--strict", action="store_true", help="前向引用也判失败")
    ap.add_argument("--semantic", action="store_true",
                    help="表格引用语义自检（F）也判失败；默认只报提示")
    a = ap.parse_args()

    p = pathlib.Path(a.manuscript)
    if not p.exists():
        die(f"找不到手稿：{p}")
    t = p.read_text()
    heads = sections(t)

    # 图清单是"题注列表"而不是定义点：插图由构建层插到首次引用处，
    # 所以清单要从引用扫描里排除，只留正文引用。
    i_fig = t.index("## Figures") if "## Figures" in t else len(t)
    i_ref = t.index("## References") if "## References" in t else len(t)
    prose = t[:i_fig] + t[i_ref:]
    figs_blk = t[i_fig:i_ref]

    hard: list[str] = []          # 判失败
    soft: list[str] = []          # 只提示

    # ------------------------------------------------- A/B：各族定义与引用
    for fam, defpat in FAMILIES.items():
        defs = {int(m.group(1)): m.start() for m in re.finditer(defpat, t)}
        refpat = REF_PATTERNS[fam]
        refs = []
        for m in re.finditer(refpat, prose):
            n = int(m.group(1))
            if defs.get(n) == m.start():      # 定义行本身
                continue
            refs.append((n, m.start()))
        # B 编号断档
        if defs:
            mx = max(defs)
            gap = [n for n in range(1, mx + 1) if n not in defs]
            if gap:
                hard.append(f"{fam} 编号断档：缺 {gap}（现有 {sorted(defs)}）")
        # A 悬空
        dangling = sorted({n for n, _ in refs if n not in defs})
        if dangling:
            hard.append(f"{fam} 悬空引用：{dangling}（全文未定义）")
        # E 前向。**带显式位置指针的**（"(Proposition 2, Section 4.3)" 这种，R20 之后的编号）
        # 是正常学术写法——读者被告知去哪看；没指针又提前用，才是 A2 那一类真缺陷。
        for n, pos in refs:
            if n in defs and pos < defs[n]:
                sec = section_of(pos, heads)
                tail = prose[pos:pos + 40]
                pointed = bool(re.search(r"Section \d+(\.\d+)?|Appendix A\.\d+", tail))
                msg = f"{fam} {n} 在 [{sec}] 被引，定义在 [{section_of(defs[n], heads)}]"
                if pointed:
                    soft.append("前向引用（带位置指针，正常）：" + msg)
                elif a.strict or sec not in BENIGN_SECTIONS:
                    hard.append("前向引用（无指针且在 §3 之后）：" + msg)
                else:
                    soft.append("前向引用（引言/相关工作，正常）：" + msg)

    # ------------------------------------------------- 表格
    tab_defs = {int(m.group(1)): m.start() for m in
                re.finditer(r"\*\*Table (\d+)\.\*\*", t)}
    tab_cap = {int(m.group(1)): m.group(2).strip() for m in
               re.finditer(r"\*\*Table (\d+)\.\*\*\s*(.+)", t)}
    if tab_defs:
        mx = max(tab_defs)
        gap = [n for n in range(1, mx + 1) if n not in tab_defs]
        if gap:
            hard.append(f"Table 编号断档：缺 {gap}")
    for m in re.finditer(r"\bTables? (\d+)", prose):
        n = int(m.group(1))
        if n not in tab_defs:
            hard.append(f"Table {n} 悬空引用（无此题注）")

    # ------------------------------------------------- 图
    fig_defs = {int(m.group(1)): m.start()
                for m in re.finditer(r"\*\*Fig\. (\d+)\.\*\*", figs_blk)}
    firsts: list[int] = []
    for m in re.finditer(r"\bFigs?\. ?(\d+)", prose):
        n = int(m.group(1))
        if n not in firsts:
            firsts.append(n)
        if n not in fig_defs:
            hard.append(f"Fig. {n} 悬空引用（图清单无此题注）")
    if fig_defs:
        uncited = sorted(set(fig_defs) - set(firsts))
        if uncited:
            hard.append(f"图未被正文引用：{uncited}")
        if firsts != sorted(firsts):
            hard.append(f"图号与首次引用次序不一致：{firsts}")
        else:
            soft.append(f"图号按首次引用顺序：{firsts}")

    # ------------------------------------------------- 节号 / 附录号
    have_sec = {m.group(1) for m in re.finditer(r"(?m)^## (\d+)\. ", t)}
    have_sub = {m.group(1) for m in re.finditer(r"(?m)^### (\d+\.\d+) ", t)}
    have_A = {m.group(1) for m in re.finditer(r"(?m)^### A\.(\d+) ", t)}

    # 重复节号：同一编号出现两个标题，是"搬走一节却忘了给后面的重新编号"的典型症状
    # （R14c 曾让过渡期的手稿里同时有两个 ### 8.6）。
    for pat, lab, have in ((r"(?m)^## (\d+)\. ", "节", have_sec),
                           (r"(?m)^### (\d+\.\d+) ", "小节", have_sub),
                           (r"(?m)^### A\.(\d+) ", "附录小节", have_A)):
        seen = [m.group(1) for m in re.finditer(pat, t)]
        dup = sorted({x for x in seen if seen.count(x) > 1})
        for d in dup:
            hard.append(f"{lab}号 {d} 有 {seen.count(d)} 个标题（重复编号）")

    def _sec_refs(text):
        """Section N.M / Section N，含 "Sections A and B" / "Sections A, B and C" 枚举。"""
        out = []
        for m in re.finditer(r"\bSections?\s+(\d+(?:\.\d+)?)", text):
            out.append(m.group(1))
            # 枚举尾巴：", 4.2 and 4.3" / " and 8.8"
            tail = text[m.end():m.end() + 60]
            for mm in re.finditer(r"(?:,\s*|\s+and\s+)(\d+(?:\.\d+)?)", tail):
                out.append(mm.group(1))
        return out

    for ref in _sec_refs(prose):
        if "." in ref:
            if ref not in have_sub:
                hard.append(f"Section {ref} 悬空（无此小节标题）")
        elif ref not in have_sec:
            hard.append(f"Section {ref} 悬空（无此节标题）")
    for m in re.finditer(r"\bAppendix A\.(\d+)", prose):
        if m.group(1) not in have_A:
            hard.append(f"Appendix A.{m.group(1)} 悬空（无此附录小节）")

    # ------------------------------------------------- F 表格引用语义
    for m in re.finditer(r"\bTables?\s+(\d+)\b", prose):
        n = int(m.group(1))
        if n not in tab_defs:
            continue
        cw = set(sorted(_content_words(tab_cap.get(n, "")))[:10])
        if not cw:
            continue
        lo, hi = max(0, m.start() - 300), min(len(prose), m.end() + 300)
        ctx = set(re.sub(r"[^a-z]+", " ", prose[lo:hi].lower()).split())
        hit = cw & ctx
        if len(hit) < 1:
            msg = (f"Table {n} 的引用处与表题共享的内容词不足（表题内容词 {sorted(cw)[:6]}，"
                   f"命中 {sorted(hit)}）——编号连续性查不出「旧编号指向了内容不同的新表」")
            (hard if a.semantic else soft).append(msg)

    # ------------------------------------------------- 报告
    print(f"手稿：{p}")
    print(f"  命题 {sorted(int(m.group(1)) for m in re.finditer(FAMILIES['Proposition'], t))}"
          f" | 推论 {sorted(int(m.group(1)) for m in re.finditer(FAMILIES['Corollary'], t))}"
          f" | 假设 {sorted(int(m.group(1)) for m in re.finditer(FAMILIES['Assumption'], t))}"
          f" | 表 {len(tab_defs)} | 图 {len(fig_defs)}"
          f" | 正文小节 {len(have_sub)} | 附录小节 {len(have_A)}")
    for s in soft:
        print(f"  · {s}")
    if hard:
        print(f"\n✗ {len(hard)} 项硬伤：")
        for h in hard:
            print("   ·", h)
        return 1
    print("\n✓ check_refs 全部通过：无悬空引用、无编号断档、图号有序")
    return 0


if __name__ == "__main__":
    sys.exit(main())
