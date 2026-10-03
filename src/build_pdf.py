#!/usr/bin/env python3
"""把 paper/manuscript.md 编译成单栏投稿版 PDF（xelatex + STIX Two）。

设计要点（都是实测过的，不是想当然）：
  * `°` 共 91 处，**全部落在数学环境内**——必须转成 `^\\circ`，否则报错或渲染空白。
  * 全稿只有 6 种非 ASCII 字符（° – — Ö ï é），且**没有任何 `\\begin{...}` 环境**，
    数学命令共 61 种，全部是 unicode-math 能直接提供的标准命令。
  * 引擎选 xelatex 而非 pdflatex：TeX Live 装的是 **basic** 方案，
    `newtxtext` 缺 xstring/mweights、`mathptmx` 缺 rsfs10、URW Times 的 TFM 也不全，
    pdflatex 路线全军覆没；xelatex + 系统字体不依赖这些 TeX 字体包。
  * 图号在源稿里**显式写出**（不靠 LaTeX 自动编号），插图位置由"首次引用"定位，
    所以"号"必须与"首次引用顺序"一致，否则印出来就是乱序——第五至七轮印出的
    顺序曾是 1,6,2,11,3,4,5,9,7,8,10。第七轮补遗用
    `src/apply_r7b_fig_renumber.py` 把号重排成按首次引用递增，并在下面第 3 项
    成品校验里**断言落页顺序递增**（原先只查集合齐全，故这类错误一直是静默的）。
  * 表格按源序 1..18 单调，交给 longtable 自动编号；唯 Table 6 是"一题两表"，
    第二块不给 caption（否则会多占一个号、把 Table 7..18 全推错一位），改用斜体引导语。
  * Algorithm 1 原是代码块，verbatim 不换行会溢出；实测最长行 85 字符，
    在 \\footnotesize 的等宽字体下约 14.1cm < 16cm 正文宽，故装箱并降字号；超限直接报错。

用法：
    python3 src/build_pdf.py                    # -> build/manuscript_submission.pdf
    python3 src/build_pdf.py --spacing single   # 单倍行距版本
    python3 src/build_pdf.py --font "Latin Modern Roman"
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = ROOT / "build"
MS = PAPER / "manuscript.md"

# 图文件目录与显示宽度。
#
# **号 -> 文件 的映射不再写在这里**：唯一真值来源是手稿 `## Figures` 里每条题注尾部的
# (`fig_x.pdf`) 标记，由 split_manuscript 推导并顺带核对文件存在。早先这里硬编码了
# 一张 号->文件 表，那么"稿里改了号、忘了改这张表"就会把题注静默接到别的图上——
# 与"把期望表数写死成 range(1,40)"是同一类错误。
FIGDIR = ROOT / "results"
# 稿中在用的两套图目录（推导号->文件时只在这里找）
FIGSEARCH = ("figures_nfr", "figures_real")
#
# 展示宽度（正文宽的倍数）。**按图文件名索引，不按图号**：图号会随"按首次引用顺序
# 重排"而变，文件名不变。第七轮重编号时 `{4: 0.62}` 就差点静默落到另一张图上
# （4 原本是 fig_snr，重编号后 4 变成 fig_sparsity）。fig_snr 宽高比只有 1.35，
# 拉满行宽会过高。
FIGWIDTH = {"fig_snr.pdf": 0.62}

# 正文宽：A4(21cm=597.5pt) - 2×2.5cm 页边(141.7pt) = 455.8pt
TEXTWIDTH_PT = 597.5 - 2 * (2.5 * 28.3465)
# 等宽字体每字符宽 0.525em
MONO_EM = 0.525
# (LaTeX 字号命令, pt)
MONO_SIZES = [("\\small", 10.0), ("\\footnotesize", 9.0),
              ("\\scriptsize", 8.0), ("\\tiny", 7.0)]


def mono_size_for(longest: int) -> tuple[str, float]:
    """按最长行挑等宽字号，留 12% 余量。返回 (字号命令, 实际占宽比)。"""
    for cmd, pt in MONO_SIZES:
        width = longest * MONO_EM * pt
        if width <= 0.88 * TEXTWIDTH_PT:
            return cmd, width / TEXTWIDTH_PT
    die(f"算法块最长行 {longest} 字符，即使 \\tiny 也放不下")


def die(msg: str) -> None:
    print(f"错误：{msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- 文本变换
def fix_degree(text: str) -> str:
    """数学环境内的 `°` -> `^\\circ`；文本内的保留原字符（xelatex 下可直接渲染）。"""
    out, in_math = [], False
    for ch in text:
        if ch == "$":
            in_math = not in_math
            out.append(ch)
        elif ch == "\u00b0":
            out.append("^\\circ" if in_math else "\u00b0")
        else:
            out.append(ch)
    return "".join(out)


def raw_latex(body: str) -> str:
    """包一个 pandoc 直通的 raw LaTeX 块。"""
    return "```{=latex}\n" + body.strip("\n") + "\n```\n"


# --------------------------------------------------------------------------- 解析
def split_manuscript(src: str) -> dict:
    """按标记切分：标题块 / 摘要 / 正文 / 图清单 / 参考文献。"""
    for k in ("## Abstract", "## 1. Introduction", "## Figures", "## References"):
        if k not in src:
            die(f"手稿缺少标记 {k!r}")

    title = re.search(r"^# (.+)$", src, re.M)
    if not title:
        die("找不到标题行（`# ...`）")

    i_abs, i_intro = src.index("## Abstract"), src.index("## 1. Introduction")
    i_fig, i_ref = src.index("## Figures"), src.index("## References")
    head, abs_block = src[:i_abs], src[i_abs:i_intro]
    body, fig_block, ref_block = src[i_intro:i_fig], src[i_fig:i_ref], src[i_ref:]

    # 注意 `[^\n]+`：整体用了 re.S，若写成 `(.+)` 会跨行贪婪捕获，
    # 把关键词之后的 `---` 分隔线一起吞进来（实测在成品里印出一个游离的破折号）。
    m = re.search(r"## Abstract\s*\n+(.*?)\n+\s*\*\*Keywords:\*\*\s*([^\n]+)", abs_block, re.S)
    if not m:
        die("解析摘要/关键词失败")
    abstract, keywords = m.group(1).strip(), m.group(2).strip()

    authors = re.search(r"^\*\*Authors:\*\*\s*(.+)$", head, re.M)
    affil = re.search(r"^\*\*Affiliation:\*\*\s*(.+)$", head, re.M)
    corr = re.search(r"^\*\*Corresponding author:\*\*\s*(.+)$", head, re.M)
    if not (authors and affil):
        die("解析作者/单位失败")

    figs: dict[int, str] = {}
    figfiles: dict[int, str] = {}
    for m in re.finditer(r"^\*\*Fig\. (\d+)\.\*\*(.+)$", fig_block, re.M):
        n, cap = int(m.group(1)), m.group(2).strip()
        # caption 以 `( `fig_x.pdf` )` 结尾（括号和反引号的顺序两种都要认）
        ref = re.search(r"\(?\s*`([\w./]+\.pdf)`\s*\)?\s*\.?\s*$", cap)
        if not ref:
            die(f"Fig. {n} 的 caption 结尾没有 (file.pdf) 标记，无法定位插图")
        rel = ref.group(1)
        # 题注只写文件名（如 fig_scale.pdf），所在子目录由这里唯一解析并写进映射。
        # 只在前两套目录里找：results/figures_v2/ 是早前 make_figures_v2.py 留下的
        # 另一套同名文件（fig_architecture.pdf 与稿中在用的那份不是同一个），
        # 一并纳入搜索会因为重名把这里逼成"猜"。
        cand = [FIGDIR / d / Path(rel).name for d in FIGSEARCH
                if (FIGDIR / d / Path(rel).name).exists()]
        if len(cand) != 1:
            die(f"Fig. {n} 指向 {rel}，在 {FIGSEARCH} 下找到 {len(cand)} 个："
                f"{[str(c) for c in cand]}")
        figs[n] = cap[:ref.start()].strip()
        figfiles[n] = str(cand[0].relative_to(FIGDIR))
    if sorted(figs) != list(range(1, len(figs) + 1)):
        die(f"图号不连续：{sorted(figs)}")

    refs: dict[int, str] = {}
    for m in re.finditer(r"^\[(\d+)\]\s*(.+)$", ref_block, re.M):
        refs[int(m.group(1))] = m.group(2).strip()
    if sorted(refs) != list(range(1, len(refs) + 1)):
        die(f"参考文献编号不连续：{sorted(refs)}")

    # 小节编号（用于和 .aux 里 LaTeX 实际编出的号核对）
    subsec = re.findall(r"^### (\d+\.\d+)\s", body, re.M)
    if not subsec:
        die("正文里找不到 `### N.M` 形式的小节标题")

    return dict(title=title.group(1).strip(), authors=authors.group(1).strip(),
                affil=affil.group(1).strip(),
                corr=(corr.group(1).strip() if corr else ""),
                abstract=abstract, keywords=keywords, body=body, figs=figs, refs=refs,
                figfiles=figfiles,
                subsec=sorted(subsec, key=lambda x: [int(t) for t in x.split(".")]))


# --------------------------------------------------------------------------- 正文变换
def normalise_headings(body: str) -> str:
    """去掉手工编号，并把层级**整体提升一级**。

    稿中 `## 3. Problem Formulation` 是顶层、`### 3.1 Mixture model` 是次层。
    但 pandoc 的 `--top-level-division=section` 把 `#` 映射到 \\section、`##` 到 \\subsection，
    若原样交给它，顶层会变成 \\subsection，节号会被编成 "0.1"（实测踩过）。
    提升一级后：`#`→\\section 得 "1"、`##`→\\subsection 得 "1.1"，与稿中自己的编号一致。
    """
    def rep(m):
        return "#" * (len(m.group(1)) - 1) + " " + m.group(2)
    # 编号后**点号可有可无**：`## 3. Problem Formulation` 有点，`### 3.1 Mixture model` 没有。
    # 早先写成 `\d+(?:\.\d+)?\.` 就漏掉了后者，于是它被当成 \subsubsection 编成 "3.0.1"（实测踩过）。
    return re.sub(r"^(#{2,3}) \d+(?:\.\d+)*\.?\s+(.*)$", rep, body, flags=re.M)


def star_appendix(tex: str) -> str:
    r"""把附录的章节标题改为**不编号**形式。

    附录里的表号从 19 继续（人工维持），但节号不能让 LaTeX 自己编，否则附录会被编成
    "10 Appendix A."，而正文章节号也要跟着重排。用 `\section*` / `\subsection*`
    并保留稿中显式写好的 "A.1" 前缀，编号完全由文本决定。
    """
    # 注意：附录标题没有数字前缀，normalise_headings 的"提升一级"规则（要求 `## <数字>.`）
    # 不匹配它，于是 pandoc 把它排成 \subsection、把 `### A.n` 排成 \subsubsection。
    # 两种形态都要处理（实测：只处理 \section 时命中数为 0，附录被编成 "11.1"）。
    #
    # ⚠️ 标题里含内联数学时（`### A.3 ... $\eta$`），pandoc 会包一层
    #    `\subsubsection{\texorpdfstring{A.3 ...}{...}}`；只认裸 `\subsubsection{A.` 的旧正则
    #    会漏掉 A.3 / A.4，使它们被 LaTeX 编成 "12.0.1"/"12.0.2"（**长期静默**，因为
    #    \subsubsection 的号不进 subsection 的 contentsline，直到 R10e 的 Declarations
    #    让 aux 多出 12.1 才被成品校验的 want_sub 抓到）。正则必须容忍 `\texorpdfstring{`。
    #
    # 另：R10e 的 `## Declarations` 也不带数字、同样不被提升，会被排成 \subsection 挂在
    #     §12 之下成为 "12.1" —— 一并改为不编号的顶层节。
    n_s = len(re.findall(r"\\subsection\{Appendix A\.", tex)) + \
          len(re.findall(r"\\section\{Appendix A\.", tex))
    n_b = len(re.findall(r"\\subsubsection\{(?:\\texorpdfstring\{)?A\.\d", tex))
    n_d = len(re.findall(r"\\subsection\{Declarations\}", tex))
    tex = re.sub(r"\\subsection\{Appendix A\.", r"\\section*{Appendix A.", tex)
    tex = re.sub(r"\\section\{Appendix A\.", r"\\section*{Appendix A.", tex)
    tex = re.sub(r"\\subsubsection\{(\\texorpdfstring\{)?A\.(\d)",
                 r"\\subsection*{\g<1>A.\g<2>", tex)
    tex = re.sub(r"\\subsection\{Declarations\}", r"\\section*{Declarations}", tex)
    print(f"附录：{n_s} 个节标题、{n_b} 个小节标题改为不编号形式"
          + (f"；Declarations {n_d} 处" if n_d else ""))
    return tex


def table_captions(body: str) -> tuple[str, list[int], list[list[float]]]:
    """`**Table N.** cap` -> pandoc 的 `: cap` 语法。

    返回 (正文, 表格编号序列, 每张表的列宽权重)。列宽由 Markdown 内容自己算，
    因为 pandoc 对 pipe 表**一律给等宽**（实测 6 列各 0.1667），标签列会被挤到换行。
    """
    lines = body.split("\n")
    out: list[str] = []
    nums: list[int] = []
    specs: list[list[float]] = []
    i = 0
    while i < len(lines):
        m = re.match(r"^\*\*Table (\d+)\.\*\*\s*(.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        n, cap = int(m.group(1)), m.group(2).strip()
        j, blocks = i + 1, []
        while j < len(lines):
            l = lines[j]
            if l.strip() == "":
                j += 1
                continue
            if l.startswith("|"):
                k2 = j
                while k2 < len(lines) and lines[k2].startswith("|"):
                    k2 += 1
                blocks.append((j, k2))
                j = k2
                continue
            if re.match(r"^\*\*Table \d+\.\*\*", l) or l.startswith("#"):
                break
            # 表题与表格块之间允许有正文，同一表题下的后续表格块仍归它。
            # 原先遇到正文即 break，会把后面的块漏出 specs，报"longtable 比预期的多"（实测踩过）。
            j += 1
        if not blocks:
            die(f"Table {n} 的 caption 后面找不到表格")
        nums.append(n)
        piece = [x.rstrip() for x in lines[i + 1:j]]
        while piece and not piece[-1].strip():
            piece.pop()
        # 第一块与第二块之间原文没有正文时，补一句块间说明（Table 6 的第二个块依赖此句）
        if len(blocks) > 1:
            gap = [x for x in lines[blocks[0][1]:blocks[1][0]] if x.strip()]
            if not gap:
                rel = blocks[1][0] - (i + 1)
                piece.insert(rel, "*SDR (dB), the same runs.*")
                piece.insert(rel + 1, "")
        out += [f": {cap}", ""] + piece + [""]
        for bi, (s_, e_) in enumerate(blocks):
            specs.append(col_weights(lines[s_:e_], f"Table {n}" + ("（后半块）" if bi else "")))
        i = j
    return "\n".join(out), nums, specs


def _visual_len(cell: str) -> int:
    """估算单元格的渲染宽度（字符数）：数学按半宽计，去掉强调标记。"""
    def shrink(m):
        return "x" * max(1, round(len(m.group(0)) * 0.45))
    txt = re.sub(r"\$[^$]*\$", shrink, cell)
    txt = txt.replace("**", "").replace("*", "")
    return len(txt)


def col_weights(block: list[str], label: str) -> list[float]:
    """由表格内容算列宽权重（和为 1）；顺带把关表格是否良构。"""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in block]
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]   # 去掉对齐行
    if not rows:
        die(f"{label} 的表格体为空")
    ncols = len(rows[0])
    for r in rows:
        if len(r) != ncols:
            die(f"{label} 列数不一致：首行 {ncols} 列，但某行 {len(r)} 列 —— "
                f"常见原因是单元格的数学里写了未转义的 `|`（应写 \\vert）")
    # 每个单元格的 $ 必须成对——否则说明数学跨了列，等价于上面那种缺陷的隐蔽形式
    for r in rows:
        for c in r:
            if c.count("$") % 2:
                die(f"{label} 有单元格数学未闭合（$ 数为奇数）：{c[:60]!r}")
    longest = [max(_visual_len(r[j]) for r in rows) for j in range(ncols)]
    # 每列最长**不可断词**（按空白切分）：既作为列宽下限（保证不可断词排得下），
    # 也用来判断整表能否装进版心，装不下就自动降一档字号（见 rewrite_tables.pick_size）
    toks = [max((_visual_len(t) for r in rows for t in r[j].split()), default=1)
            for j in range(ncols)]
    w = [max(max(l, 4) ** 0.75, t * 1.15)           # 指数压缩极差 + 不可断词下限
         for l, t in zip(longest, toks)]
    s = sum(w)
    return [x / s for x in w], toks


def rewrite_tables(tex: str, specs: list[list[float]]) -> str:
    """把 pandoc 的 longtable 列规格换成按内容算的宽度，并拆掉逐单元格的 minipage。

    pandoc 的原始规格是 `p{(\\linewidth - k\\tabcolsep) * \\real{0.1667}}`（等宽），
    且每个单元格包一层 `\\linewidth` 宽的 minipage —— 在窄列里必然溢出。
    """
    it = iter(specs)
    seen = 0

    TEXTWIDTH_PT = 16.0 * 72 / 2.54          # a4paper，边距 2.5cm
    SIZES = [(r"\footnotesize", 4.9), (r"\scriptsize", 4.3), (r"\tiny", 3.3)]

    def pick_size(w: list[float], toks: list[int]) -> str:
        """选一个能装下的字号：要求每一列的最长不可断词都排得进该列。"""
        avail = TEXTWIDTH_PT - 2 * (len(toks) - 1) * 6.0     # 去掉列间 \tabcolsep
        for cmd, cw in SIZES:
            if all(t * cw <= w[j] * avail for j, t in enumerate(toks)):
                return cmd
        return SIZES[-1][0]

    def rep(m):
        nonlocal seen
        try:
            w, toks = next(it)
        except StopIteration:
            die("body.tex 的 longtable 比预期的多")
        seen += 1
        gap = 2 * (len(w) - 1)          # 列间 \tabcolsep 的个数
        size = pick_size(w, toks)
        spec = "".join(
            ">{" + size + r"\raggedright\arraybackslash}p{(\linewidth - "
            rf"{gap}\tabcolsep) * \real{{{x:.4f}}}}}"
            for x in w)
        return r"\begin{longtable}[]{@{}" + spec + "@{}}"

    tex = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}", rep, tex, flags=re.S)
    if seen != len(specs):
        die(f"longtable 数 {seen} 与 Markdown 表格块数 {len(specs)} 不一致")

    # 拆掉 pandoc 的逐单元格 minipage（列已是 p{} 定宽，再套 \linewidth 的 minipage 会溢出）。
    # 必须**成对**匹配：全局删 `\end{minipage}` 会连算法框自己的 minipage 一起拆掉（实测踩过）。
    tex, n_sub = re.subn(
        r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\n(.*?)\\end\{minipage\}",
        lambda m: m.group(1), tex, flags=re.S)
    print(f"表格：{seen} 张，已按内容重算列宽；拆掉 {n_sub} 个单元格 minipage")
    return tex


def algorithm_block(body: str) -> str:
    """Algorithm 1 的代码块 -> 带框浮动体（verbatim 不换行，必须装箱降字号）。"""
    m = re.search(r"^\*\*Algorithm (\d+)\*\*\s*(.+?)\n\n```\n(.*?)\n```\n", body, re.S | re.M)
    if not m:
        return body
    num, title, code = m.group(1), m.group(2).strip(), m.group(3)
    if "\\end{verbatim}" in code:
        die("算法块含 \\end{verbatim}，需换实现")
    longest = max(len(l) for l in code.split("\n"))
    size_cmd, frac = mono_size_for(longest)
    block = raw_latex(rf"""
\begin{{figure}}[!ht]
\centering
\begin{{minipage}}{{\textwidth}}
\hrule height 0.8pt
\vspace{{4pt}}
{{\small\textbf{{Algorithm {num}}}\enspace {title}}}
\vspace{{3pt}}
\hrule height 0.4pt
\vspace{{4pt}}
{{{size_cmd}
\begin{{verbatim}}
{code}
\end{{verbatim}}
}}
\vspace{{2pt}}
\hrule height 0.4pt
\end{{minipage}}
\end{{figure}}
""")
    print(f"算法块：最长行 {longest} 字符 -> {size_cmd}（占正文宽 {frac:.0%}）")
    return body[:m.start()] + block + body[m.end():]


def insert_figures(body: str, figs: dict[int, str], figfiles: dict[int, str]) -> str:
    """在每张图**首次被引用**的那一段之后插入图片块。"""
    ins: list[tuple[int, int]] = []
    for n in sorted(figs):
        m = re.search(rf"Fig\.\s*{n}(?![0-9])", body)
        if not m:
            die(f"正文未引用 Fig. {n}，无法定位插图位置")
        end = body.find("\n\n", m.end())
        ins.append((end if end > 0 else len(body), n))

    for pos, n in sorted(ins, reverse=True):
        w = FIGWIDTH.get(Path(figfiles[n]).name, 0.98)
        # **不用浮动体**。用 figure 环境时，一旦这一页剩余空间不足，LaTeX 仍会把图放上去，
        # 结果图元画出来了、**题注被挤出 \textheight 而整条消失**（日志只留一句
        # "Overfull \vbox ... too high"，成品里那句题注根本不存在）。实测在第六轮改稿
        # 引起分页移动后，图 3 的题注就是这样丢的，只有成品校验的图号检查能发现。
        # 改成正常竖直列表里的居中块后，LaTeX 会自然分页，题注不可能离开页面；
        # 高度的上限保证「图 + 题注」整体永远装得下一页。
        block = raw_latex(rf"""
\par\medskip
\begin{{center}}
\includegraphics[width={w}\linewidth,height=0.68\textheight,keepaspectratio]{{{figfiles[n]}}}\\[6pt]
\parbox{{0.92\linewidth}}{{\centering\small\textbf{{Fig. {n}.}} {figs[n]}}}
\end{{center}}
\par\medskip
""")
        body = body[:pos] + "\n\n" + block + body[pos:]
    return body


def build_body_md(d: dict) -> tuple[str, list[int], list[list[float]]]:
    body = d["body"]
    body = re.sub(r"^\s*---\s*$", "", body, flags=re.M)
    body = normalise_headings(body)
    body = algorithm_block(body)
    body, tnums, specs = table_captions(body)
    body = insert_figures(body, d["figs"], d["figfiles"])
    body = re.sub(r"\n{3,}", "\n\n", body)
    return fix_degree(body), tnums, specs


# --------------------------------------------------------------------------- 排版
# 字体回退链：(正文字体, 数学字体)。空串表示用 XeLaTeX/unicode-math 的默认。
# STIX Two Text 虽是 macOS 自带、观感最好，但它是**可变字体**，XeTeX 选不到 Bold/Italic
# 命名实例（实测缺字形），故只用作数学字体（STIXTwoMath.otf 是单面静态字体，正常）。
FONTSETS = [
    ("Times New Roman", "STIX Two Math"),
    ("Palatino", "STIX Two Math"),
    ("Times New Roman", ""),
    ("", ""),
]


def preamble(textfont: str, mathfont: str) -> str:
    # 注意顺序：\setmainfont 必须在 fontspec 之后，\setmathfont 必须在 unicode-math 之后
    text_line = (r"\setmainfont{%s}" % textfont) if textfont else "% (默认正文字体)"
    math_line = (r"\setmathfont{%s}" % mathfont) if mathfont else "% (默认数学字体)"
    # 编译在 build/ 下进行，图在 results/ 下 —— 用绝对路径的 graphicspath，避免相对路径漂移
    graphicspath = str(ROOT / "results") + "/"
    return r"""
\documentclass[11pt,a4paper]{article}

\usepackage{fontspec}
%(textline)s

\usepackage{amsmath}
\usepackage{unicode-math}
%(mathline)s
%% unicode-math 不提供 \square（稿中 6 处证毕符号用它），映射到它的正式名
\providecommand{\square}{\mdlgwhtsquare}

\usepackage{graphicx}
\graphicspath{{%(graphicspath)s}}
\usepackage{calc}          %% pandoc 的列宽表达式里用到 calc 算术
\providecommand{\real}[1]{#1}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage[margin=2.5cm]{geometry}
\usepackage{setspace}
\usepackage{lineno}
\usepackage[font=small,labelfont=bf,labelsep=period,%%
            justification=raggedright,singlelinecheck=false]{caption}
\usepackage{etoolbox}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}

%% pandoc 会给"没有 caption 的续接表"发 {\def\LTcaptype{none} ...}（表示不递增编号），
%% 而 longtable(TeX Live 2026) 在 \begin{longtable} 当口就 \refstepcounter{\LTcaptype}，
%% 于是报 "No counter 'none' defined"。登记一个同名计数器即可（该表本就无 caption）。
\newcounter{none}

%% 表格统一小字号，longtable 的 caption 占满正文宽
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{4pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.08}

%% 抑制浮动体漂移
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{6}
""" % dict(textline=text_line, mathline=math_line, graphicspath=graphicspath)


def hypersetup(d: dict) -> str:
    """写 PDF 元数据（投稿系统的预填/检索会用到）。"""
    def esc(s: str) -> str:
        for a, b in (("\\", ""), ("{", ""), ("}", ""), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
            s = s.replace(a, b)
        return s
    kw = re.sub(r"\s*;\s*", ", ", d["keywords"])
    return (r"\hypersetup{pdftitle={%s}, pdfauthor={%s}, pdfsubject={%s}, pdfkeywords={%s}}"
            % (esc(d["title"]), esc(d["authors"]),
               "Underdetermined blind source separation (UBSS)", esc(kw)))


def corr_tex(corr: str) -> str:
    """把通讯作者行的 markdown 转成 LaTeX：反引号包住的邮箱 -> \\texttt，并转义特殊字符。"""
    if not corr:
        return ""

    def code(m):
        t = m.group(1)
        for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                     ("&", r"\&"), ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}")):
            t = t.replace(a, b)
        return r"\texttt{%s}" % t

    return re.sub(r"`([^`]+)`", code, corr)


def author_block(d: dict) -> str:
    """标题页的作者块：姓名 / 单位 / 通讯作者（含邮箱）。"""
    lines = [re.sub(r"[.,]\s*$", "", d["authors"]), r"\small " + d["affil"]]
    if d["corr"]:
        lines.append(r"\small Corresponding author: " + corr_tex(d["corr"]))
    return (r"\\[2pt]" + "\n").join(lines)


def write_tex(d: dict, body_tex: str, bibitems: str, abstract_tex: str,
              fonts: tuple[str, str], half: bool) -> str:
    return rf"""{preamble(*fonts)}

{hypersetup(d)}

\title{{\large\bfseries {d['title']}}}
\author{{{author_block(d)}}}
\date{{}}

\begin{{document}}
{'\\onehalfspacing' if half else '\\singlespacing'}
\linenumbers

\maketitle
\thispagestyle{{plain}}

\begin{{abstract}}
\noindent {abstract_tex}
\end{{abstract}}

\noindent\textbf{{Keywords:}} {d['keywords']}

{body_tex}

\begin{{thebibliography}}{{99}}
{bibitems}
\end{{thebibliography}}

\end{{document}}
"""


def verify_pdf(pdf: Path, d: dict) -> list[str]:
    """从**成品 PDF** 反向校验：编译通过不等于内容正确（实测漏过两处）。

    返回问题列表；空列表表示通过。
    """
    r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                       capture_output=True, text=True)
    if r.returncode:
        return ["pdftotext 失败，无法校验"]
    txt = r.stdout
    flat = re.sub(r"\s+", " ", txt)                    # 行号与断行会切开词组，先归一化空白
    # lineno 的行号是独立 token，抹掉它才能查词组；用前后都是空白的 `\d+` 判断，避免吃掉节号
    flat_num = re.sub(r"(?<=\s)\d{1,3}(?=\s)", "", flat)
    flat_num = re.sub(r"\s+", " ", flat_num)
    bad: list[str] = []

    # 1) 占位标记必须全部消失
    for pat, what in ((r"%%%R\d+%%%", "%%%R#%%% 标记"),
                      (r"ZZBIB\d+ZZ", "ZZBIB#ZZ 标记"),
                      (r"\@\@", "@@ 占位符")):
        if re.search(pat, flat):
            bad.append(f"成品里残留 {what}")

    # 1b) 投稿前的草稿痕迹：这三类一旦留在成品里，就是"unfinished draft"的证据
    for pat, what in ((r"repository to be inserted", "仓库地址占位符 `[repository to be inserted]`"),
                      (r"funding information to be inserted",
                       "基金信息占位符 `[funding information to be inserted]`"),
                      (r"add campus/city", "单位占位符 `[add campus/city if required]`"),
                      (r"To be filled", "作者/单位占位符 `[To be filled]`"),
                      (r"\bTODO\b", "TODO 标记")):
        if re.search(pat, flat, re.I):
            bad.append(f"投稿前必须处理：{what}")

    # 2) markdown 语法不该漏成字面量（算法块里的 F*T 之类是合法星号，故只认"字母/空格/连字符"夹心）
    if "**" in flat:
        bad.append("成品里有字面量 `**`（markdown 粗体未转换）")
    stray = re.findall(r"\*[A-Za-z][A-Za-z \-]{2,40}\*", flat_num)
    if stray:
        bad.append(f"成品里有字面量 `*强调*`：{stray[:3]}")

    # 3) 表号、图号必须与稿中一致且齐全，**且落页顺序递增**
    #
    #    判据从**手稿**推导，不写死张数与顺序——原先把表数硬编码为 39，第七轮新增
    #    Tables 40--41 后立刻误报"表号异常"，是"把数字写死在脚本里"的同一类错误的又一例。
    #
    #    更要紧的是这里原先**只查集合齐全、不查顺序**，于是"图题注按 1,6,2,11,3,... 的
    #    次序落在纸上"这整类错误一直是静默的：集合是 {1..11}，检查通过，而读者翻到的
    #    是图 1、图 6、图 2、图 11、图 3……。集合与顺序必须分开断言。
    #    行号是独立 token，先抹掉；题注认得的是**行首**的 `Fig. N.` / `Table N.`
    #    （正文里的 "Fig. 4b"、"Fig. 11 shows" 不匹配，因为号后面紧跟的不是句点）。
    tabs = sorted({int(x) for x in re.findall(r"Table (\d+)[.:]", flat)})
    figs = sorted({int(x) for x in re.findall(r"Fig\. (\d+)\.", flat)})
    n_tab = len(re.findall(r"^\*\*Table (\d+)\.\*\*", d["body"], re.M))
    n_fig = len(d["figs"])
    if tabs != list(range(1, n_tab + 1)):
        bad.append(f"表号异常：稿中 {n_tab} 张，成品里 {tabs}")
    if figs != list(range(1, n_fig + 1)):
        bad.append(f"图号异常：稿中 {n_fig} 张，成品里 {figs}")

    def caption_page_order(pat: str) -> list[int]:
        """按 pdftotext 的分页符逐页取**题注**，返回落页顺序（去相邻重复）。

        逐行处理，不整页做正则替换：题注是缩进的，且行首行号是独立 token，
        用 `^\\s*\\d{1,3}\\s+` 整页替换会跨行吞掉换行，把相邻行粘起来
        （实测这样写会漏掉 Table 全表和 Fig. 11）。
        题注认得的是**行首**的 `Fig. N.` / `Table N.`：正文里的 "Fig. 4b"、
        "Fig. 11 shows" 都不匹配，因为号后面紧跟的不是句点。
        """
        seen: list[int] = []
        for page in txt.split("\f"):
            for line in page.split("\n"):
                # 行号上限给到 4 位：稿子已过 1700 行，`\d{1,3}` 会漏掉 4 位的行号，
                # 而那会让题注行以 "1141 Table 28." 开头、进而被这条校验漏检。
                s = re.sub(r"^\s*\d{1,4}(?:\s+|$)", "", line).strip()
                m = re.match(pat, s)
                if m:
                    x = int(m.group(1))
                    if not seen or seen[-1] != x:
                        seen.append(x)
        return seen

    for kind, pat, n in (("表", r"Table (\d+)[.:]", n_tab),
                         ("图", r"Fig\. (\d+)\.", n_fig)):
        order = caption_page_order(pat)
        if order != list(range(1, n + 1)):
            bad.append(f"{kind} 题注落页顺序不是 1..{n}：{order}")

    # 4) 结构锚点
    for k in ("Abstract", "Keywords", "References", "Algorithm 1"):
        if k not in flat_num:
            bad.append(f"缺少结构锚点 {k!r}")

    # 5) 章节编号：从 .aux 核对（成品文本里行号与节号混在一起，分不干净；
    #    `\numberline {N}` 是 LaTeX 自己写下的权威值）
    aux = BUILD / "main.aux"
    if aux.exists():
        a = aux.read_text(errors="replace")
        n_sec = len(re.findall(r"\\contentsline \{section\}", a))
        # 章节数与首尾节名一律从**手稿**推，不写死。
        # 第七轮把期望表数写死成 range(1,40)、本轮把章节数写死成 11 与 "11 Conclusion"——
        # 同类错误各犯一次，已第二次改成推导。
        want_sec = [int(x) for x in re.findall(r"(?m)^## (\d+)\. ", d["body"])]
        if not want_sec:
            bad.append("手稿里找不到 '## N. ' 形式的编号节")
        else:
            if n_sec != len(want_sec):
                bad.append(f"\\section 数为 {n_sec}，手稿有 {len(want_sec)} 个编号节")
            for num in (want_sec[0], want_sec[-1]):
                m = re.search(rf"(?m)^## {num}\. (.+)$", d["body"])
                name = m.group(1).strip() if m else "?"
                if f"\\numberline {{{num}}}{name}" not in a:
                    bad.append(f"章节编号异常：aux 里找不到 {num} {name}")
        got_sub = sorted({m for m in re.findall(r"\\contentsline \{subsection\}\{\\numberline \{(\d+\.\d+)\}", a)},
                         key=lambda x: [int(t) for t in x.split(".")])
        if got_sub != d["subsec"]:
            bad.append(f"小节编号与稿中不符：aux {got_sub} vs 稿 {d['subsec']}")
    else:
        bad.append("找不到 main.aux，无法核对章节编号")

    # 6) 摘要的关键论断（含必须保留的负面结论）
    for k in ("noise-only points", "retention self-check", "marginally more accurate",
              "order-of-magnitude penalty", "convolutive mixtures are out of reach"):
        # LaTeX 会在行末断词，而 pdftotext 会把行末那个连字符**吃掉**：
        # 实测 "order-of-" + "magnitude" 被提取成 "orderof-magnitude"。所以每个连字符都放宽成
        # "-?\\s*"（可缺、后可跟空白），否则**改一次摘要的换行就会误报**（本轮压到 250 词就触发了）。
        pat = re.compile(r"-?\s*".join(re.escape(w) for w in k.split("-")))
        if not pat.search(flat_num):
            bad.append(f"摘要缺关键句 {k!r}")

    # 7) 参考文献条目数
    n_ref = len(re.findall(r"\[\d+\]\s*[A-ZÖ]", flat_num))
    if n_ref < len(d["refs"]):
        bad.append(f"参考文献条目只找到 {n_ref} / {len(d['refs'])} 条")

    # 8) 标题页必须带上作者、单位和通讯作者邮箱
    #    （曾静默漏过：脚本解析了 corr 却没写进标题块，编译毫无报错）
    for name in re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", d["authors"]):
        if name not in flat_num:
            bad.append(f"标题页缺作者 {name!r}")
    for e in re.findall(r"[\w.+-]+@[\w.-]+\.\w+", d["corr"]):
        if e not in flat_num:
            bad.append(f"标题页缺通讯作者邮箱 {e!r}")
    return bad


def pandoc(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["pandoc", *args], capture_output=True, text=True, **kw)


def compile_pdf(fonts: tuple[str, str], d: dict, body_tex: str, bibitems: str,
                abstract_tex: str, half: bool) -> str | None:
    """用给定字体组合编译；成功返回日志，失败返回 None。"""
    (BUILD / "main.tex").write_text(
        write_tex(d, body_tex, bibitems, abstract_tex, fonts, half))
    for _ in range(3):
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "main.tex"],
            cwd=BUILD, capture_output=True, text=True)
        if r.returncode:
            return None
    if not (BUILD / "main.pdf").exists():
        return None
    return (BUILD / "main.log").read_text(errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留中间文件")
    ap.add_argument("--spacing", choices=["half", "single"], default="half")
    ap.add_argument("--font", default=None,
                    help="正文字体；默认按 Times New Roman → Palatino → 引擎默认 依次尝试")
    ap.add_argument("--mathfont", default=None, help="数学字体；默认 STIX Two Math")
    ap.add_argument("--repo", default=None,
                    help="代码/数据仓库地址（Zenodo DOI、GitHub 等）。不给则成品里保留 "
                         "`[repository to be inserted]` 并在校验中报为投稿前必须处理项")
    ap.add_argument("--campus", default=None,
                    help="单位补充信息（校区/城市/院系），追加到 Affiliation 之后")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not MS.exists():
        die(f"找不到 {MS}")
    if not BUILD.exists():          # 不用 mkdir(exist_ok=True)：本机沙箱 shim 会误报 EEXIST
        BUILD.mkdir(parents=True)

    d = split_manuscript(MS.read_text())

    # 投稿版不应带占位符：把地址/单位这两项做成构建参数，而不是让人去改手稿。
    if a.repo:
        d["body"] = d["body"].replace("[repository to be inserted]", a.repo)
    if a.campus:
        d["affil"] = f"{d['affil']}, {a.campus}"

    body_md, tnums, specs = build_body_md(d)
    if tnums != list(range(1, len(tnums) + 1)):
        die(f"表格编号不连续：{tnums}")

    (BUILD / "body.md").write_text(body_md)

    p = pandoc(["body.md", "-f", "markdown+table_captions+raw_attribute",
                "-t", "latex", "--top-level-division=section"], cwd=BUILD)
    if p.returncode:
        die(f"pandoc 正文失败：\n{p.stderr}")
    body_tex = star_appendix(rewrite_tables(p.stdout, specs))

    # 摘要与参考文献都要过 pandoc：它们是 markdown（*强调*、$数学$），
    # 直接塞进 LaTeX 会把 `*...*` 的星号原样印出来（实测踩过）。
    p = pandoc(["-f", "markdown", "-t", "latex"], input=fix_degree(d["abstract"]))
    if p.returncode:
        die(f"pandoc 摘要失败：\n{p.stderr}")
    abstract_tex = p.stdout.strip()

    # 参考文献：标记用纯字母数字，因为 pandoc 会把 % 转义成 \%（原来用 %%%R1%%% 就因此失效）
    refs_in = "\n\n".join(f"ZZBIB{n}ZZ {fix_degree(t)}" for n, t in sorted(d["refs"].items()))
    p = pandoc(["-f", "markdown", "-t", "latex"], input=refs_in)
    if p.returncode:
        die(f"pandoc 参考文献失败：\n{p.stderr}")
    bibitems, n_sub = re.subn(r"ZZBIB(\d+)ZZ\s*", r"\\bibitem{r\1} ", p.stdout)
    if n_sub != len(d["refs"]):
        die(f"参考文献标记只替换了 {n_sub}/{len(d['refs'])} 条，说明 pandoc 改了标记形式")

    if a.font is not None:
        fontsets = [(a.font, a.mathfont if a.mathfont is not None else "STIX Two Math")]
    else:
        fontsets = FONTSETS
    log, used = None, None
    for fs in fontsets:
        log = compile_pdf(fs, d, body_tex, bibitems, abstract_tex, a.spacing == "half")
        if log:
            used = fs
            break
    if not log:
        die(f"字体组合 {fontsets} 均编译失败；见 build/main.log")

    out = Path(a.out) if a.out else PAPER / (
        "manuscript_submission.pdf" if a.spacing == "half" else "manuscript_single.pdf")
    shutil.copy(BUILD / "main.pdf", out)
    if not a.keep:
        for f in ("body.md",):
            if (BUILD / f).exists():
                (BUILD / f).unlink()

    over = [l for l in log.splitlines()
            if l.startswith("Overfull \\hbox") and float(re.search(r"\((\d+\.\d+)pt", l).group(1)) > 20]
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    print(f"输出  ：{out}")
    print(f"字体  ：{used[0] or '引擎默认'} / 数学 {used[1] or '引擎默认'}"
          f"     引擎：xelatex    行距：{'1.5 倍' if a.spacing == 'half' else '单倍'}")
    print(f"页数  ：{pages.group(1) if pages else '?'}     "
          f"表 {len(tnums)} 张（{tnums[0]}–{tnums[-1]}）   图 {len(d['figs'])} 张   文献 {len(d['refs'])} 条")
    if over:
        print(f"警告  ：{len(over)} 处水平溢出 > 20pt")
        for l in over[:10]:
            print("   " + l)
    else:
        print("排版  ：无水平溢出 > 20pt")

    problems = verify_pdf(out, d)
    if problems:
        print(f"成品校验：发现 {len(problems)} 个问题")
        for p_ in problems:
            print("   ✗ " + p_)
        must = [p_ for p_ in problems if p_.startswith("投稿前必须处理")]
        if must:
            print(f"\n   ⚠ 其中 {len(must)} 项属于**投稿前必须处理**："
                  f"补上 `--repo <URL>`（以及在需要时 `--campus <校区>`）后重新构建即可。")
            # PDF 保留（供审阅），但**以非零码退出**：只打印警告、仍然退出 0 的构建，
            # 在批处理里与"构建成功"无法区分（实测踩过：答复文档里就把它写成了"会报错"）。
            print("   ⚠ 该 PDF 已写出但不具备投稿条件，脚本以非零码退出。")
            raise SystemExit(2)
    else:
        print("成品校验：表号/图号/锚点/摘要关键句/文献条目 全部通过，无残留标记")

    if not a.keep:
        for f in ("body.md",):
            if (BUILD / f).exists():
                (BUILD / f).unlink()


if __name__ == "__main__":
    main()
