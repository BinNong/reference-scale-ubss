r"""sn-jnl 版投稿 PDF：走 Springer 官方模板类，与既有构建链**并行**。

为什么单独一个脚本
==================
`build_pdf.py` / `build_pdf_cssp.py` 用的是自建 `article + xelatex` 骨架，原因是本机
TeX Live 是 basic 方案、`sn-jnl.cls` 依赖的三个包装不上（详见那个脚本的说明）。现在
`cuted / breakurl / wrapfig` 等已装入用户树（`~/Library/texmf`，不需 sudo），官方类可用。

本脚本**不改造既有链**：那两条链上挂着 `number_equations()`、成品反查与 8 个 `verify_*`
的探针，动它们风险大于收益。所以这里是第三条并行的构建，产出
`paper/manuscript_sn_jnl.pdf`，用来判断"换到官方类后稿子是否依然成立"。

复用什么 / 换什么
==================
复用 `build_pdf` 的：`split_manuscript`、`normalise_headings`、`star_appendix`、
`number_equations`、`table_captions` / `rewrite_tables`、`algorithm_block`、
`insert_figures`、`build_body_md`、`corr_tex`、`verify_pdf`、`pandoc`。
正文变换一概不动 —— 那句话是对的：**版式换了，内容不该换**。

换掉的只有四处：
  1. 文档类  `article[11pt,a4paper]`  ->  `[referee,lineno,sn-basic]{sn-jnl}`
  2. 引擎    xelatex + fontspec + unicode-math  ->  pdflatex + T1 + utf8
             （`sn-jnl.cls` 内没有任何 xelatex/fontspec 分支，只能走 pdflatex）
  3. 前置区  `\maketitle` + `\begin{abstract}`  ->  sn-jnl 的
             `\title[] / \author*[]{\fnm{}\sur{}}\email{} / \affil[]{\orgdiv{}\orgname{}
             \orgaddress{\city{}\state{}\country{}}} / \abstract{} / \keywords{}`
  4. 参考文献形式见下。

行距
====
`[referee]` 的含义就是双倍行距。官方模板手册 §2.1 原文：
    "For the peer review and editorial stages you are requested to enable
     double-line spacing by using the [referee] option."
这是官方要求，故本脚本**默认双倍**；`--spacing half|single` 可回到 1.5 / 单倍
（供对比页数）。实测同一份稿子：1.5 倍 62 页、双倍约 78 页。

栏宽（第五处，纯排版）
====================
sn-jnl 是单栏，正文宽只有 **372pt**，比自建骨架的 455.8pt 窄 18%。窄栏下有两处
不换行就会冲出纸面，都在本脚本里处理（**都不改内容**）：

  * Algorithm 1 的等宽块：字号改按 372pt 算（`build_pdf.mono_size_for` 加了
    正文宽参数，article 链不传、行为逐字节不变）。此前当过"缩放整个 minipage"的
    备选，verbatim 进 `lrbox` 会 "You've lost some text"，已不走那条路。
  * 超宽的显示公式：`break_wide_equations()` 按手稿自己的 `\qquad` 续接点拆成
    `split` 两行，编号仍是一个。实测 §4.2 那条溢出 51.9pt，拆后归零。

剩下的 11pt 级溢出是行内数学，属正常排印公差，main() 只列 > 20pt 的。

参考文献
========
手册 §8.3 说 "BibTeX is the preferred format for references"，但本脚本**仍用
`thebibliography` 内嵌**，原因有两个：

  * 稿中 `## References` 是已按 Springer 格式排好的整句文本（`build_pdf_cssp.py` 做的
    IEEE→Springer 转换，且 `verify_*` 逐条对过）。要转 BibTeX 就得把这些句子解析成
    author/title/journal/year/volume/pages 字段——**解析错一处就是改内容**，而本脚本的
    目的是"只换版式不换内容"。
  * `thebibliography` 是核心 LaTeX 环境，`sn-jnl` 内完全合法；"preferred" 不是 "required"。

若要改成 BibTeX，应在解析层之外单独做，并配一条"转出来的字段与原句一致"的校验。

`paper/springer_template/` 在 `.gitignore` 里（第三方材料、版权归 Springer），故本脚本
**不把它当依赖**：只需其中 `sn-jnl.cls` 一个文件，每次构建复制进 `build/sn/`。缺文件时
明确报错并说明取法，而不是静默回落到自建骨架。

用法：
    python3 build_pdf_sn.py --repo https://github.com/BinNong/reference-scale-ubss
    python3 build_pdf_sn.py --spacing half        # 1.5 倍，用于和现有成品比页数
    python3 build_pdf_sn.py --keep                # 留中间文件
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import build_pdf as B

ROOT = B.ROOT
PAPER = B.PAPER
SN = PAPER / "springer_template"
BUILD = ROOT / "build" / "sn"
MS = B.MS
OUT = PAPER / "manuscript_sn_jnl.pdf"

CLASS = "sn-jnl.cls"
# sn-jnl 的正文宽。等宽块（Algorithm 1）不换行，字号必须按这个宽算：article 骨架是
# 455.8pt，sn-jnl 只有 372pt（窄 18%），拿前者的字号结论排后者会真的冲出纸面
# （实测 6 处 Overfull hbox 全在 verbatim 里）。
TEXTWIDTH_PT = 372.0
# 官方样例 sn-article.tex 用的标准包；只放 sn-jnl 自己不加载的那些
# （geometry / setspace / fontenc 由 sn-jnl 加载，重复加载会撞选项）
PACKAGES = [
    r"\usepackage[utf8]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    # lmodern = 可缩放的 Computer Modern。basic 方案没有 T1 的 Type1 CM（ec-*.pfb），
    # 只有位图 Metafont，microtype 的字体伸缩遇到位图会直接
    # "auto expansion is only possible with scalable fonts"  fatal。手册 §3.2 要求
    # "refrain from using custom fonts"、用 CM —— lmodern 正是 CM 的可缩放版本，符合要求。
    r"\usepackage{lmodern}",
    r"\usepackage{textcomp}",            # ° § – — ’ 在 T1 下可直排
    r"\usepackage{amsmath,amssymb,amsfonts}",
    r"\usepackage{amsthm}",
    r"\usepackage{mathrsfs}",
    r"\usepackage{booktabs}",
    r"\usepackage{longtable}",
    r"\usepackage{array}",
    r"\usepackage{float}",
    r"\usepackage{multirow}",
    r"\usepackage{threeparttable}",
    r"\usepackage{etoolbox}",
    r"\usepackage{microtype}",
    # xcolor **必须由作者加载**：官方样例 sn-article.tex 就加载它。sn-jnl 的
    # \begin{document} 钩子无条件调用 \XC@ifxc、\email{} 用 \textcolor{blue}{…}，
    # 不加载这两处都会 "Undefined control sequence"（实测过两个方向）。
    r"\usepackage{xcolor}",
    # hyperref 由 sn-jnl 自己加载（带空选项），这里再加载会 Option clash。
    # 链接样式改为在类加载之后用 \hypersetup 设，见 PREAMBLE_EXTRA。
]

# 表题注沿用 sn-jnl 的 caption 机制；这里只放正文已有的两处 pandoc 续接表所需
PREAMBLE_EXTRA = r"""
% sn-jnl 已加载 hyperref；链接框样式只能在此之后设
\hypersetup{hidelinks}

% natbib 由 sn-jnl 以无选项加载，默认 author-year；sn-basic 是编号制。不一致时 natbib
% 在 \AtEndDocument 往 aux 写一行 `\providecommand\NAT@force@numbers{}\NAT@force@numbers`，
% 下一轮 \begin{document} 读 aux 执行到它就报
% "Bibliography not compatible with author-year citations" —— 报错位置是 `./main.aux:N`，
% 看起来像正文的错，实际是上一轮留下的。用 natbib 自己的命令切到编号制。
\setcitestyle{numbers}

% pandoc 算 longtable 列宽时输出 `(\linewidth - 10\tabcolsep) * \real{0.1163}`。
% `\real` 是 pandoc 的 calc 辅助宏，**pandoc 自己不负定义它**；不定义则它在列规格里
% 炸开，级联出一串 "Missing number / Illegal unit of measure / Misplaced \noalign"
% （实测第一条报在表格头单元的 `\(p\)` 上，离真正的原因隔了七八条错误）。
\providecommand{\real}[1]{#1}

% pandoc 会给"没有 caption 的续接表"发 {\def\LTcaptype{none} ...}，而 longtable 在
% \begin{longtable} 当口就 \refstepcounter{\LTcaptype}，于是报 "No counter 'none'"
% （与 build_pdf.py 同一处登记）。
\newcounter{none}

% 表格统一小字号；longtable 的 caption 占满正文宽
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{4pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\arraystretch}{1.08}
\raggedbottom
"""


def die(msg: str) -> None:
    B.die(msg)


# --------------------------------------------------------------------------- 前置区
_ORGDIV = ""          # 本稿无院系一级，留空由 sn-jnl 自己处理


def _split_names(authors: str) -> list[str]:
    return [a.strip() for a in re.split(r"\s+and\s+", authors) if a.strip()]


def _org_parts(affil: str) -> dict[str, str]:
    """把一条自然语言单位拆成 sn-jnl 的 orgdiv/orgname/city/state/country。

    "Tianfu College of Southwestern University of Finance and Economics, Mianyang, China"
      -> orgname=Tianfu College …, city=Mianyang, country=China
    "School of Telecommunications Engineering, Xidian University, Xi'an, Shaanxi, China"
      -> orgname=School … Xidian University, city=Xi'an, state=Shaanxi, country=China

    逗号个数不一时按尾部倒推：末位是国家、次末位是省/州（有则填）、再往前是城市。
    拆不出合理结构时整句退到 `orgname`，**不丢文本**。
    """
    parts = [p.strip() for p in affil.split(",") if p.strip()]
    # 句法带来的冠词要剥掉：Affiliation 句是 "… is with the School of …"，
    # 剥掉 "… is with " 后剩下的 "the" 是句子的冠词、不是机构名的一部分。
    for i, p in enumerate(parts):
        if re.match(r"^(the|The)\s+", p) and i == 0:
            parts[i] = re.sub(r"^(the|The)\s+", "", p)
    if len(parts) >= 4:
        country, state, city = parts[-1], parts[-2], parts[-3]
        org = ", ".join(parts[:-3])
    elif len(parts) == 3:
        country, city, org = parts[-1], parts[-2], ", ".join(parts[:-2])
        state = ""
    elif len(parts) == 2:
        country, org = parts[-1], parts[0]
        city = state = ""
    else:
        return {"orgdiv": "", "orgname": affil, "city": "", "state": "",
                "country": ""}
    return {"orgdiv": _ORGDIV, "orgname": org, "city": city, "state": state,
            "country": country}


def sn_front(d: dict) -> str:
    """sn-jnl 的标题/作者/单位/摘要/关键词。

    姓名按 "First ... Last" 拆成 \\fnm{First} \\sur{Last}：两个词的取 (首, 末)，
    单词的整句进 \\sur。单位按作者姓名在 Affiliation 里切分，切不开时退回"共用第一条"。
    """
    names = _split_names(d["authors"])
    if len(names) < 1:
        die("解析不出作者姓名")

    # 按 "<name> is with " 把 Affiliation 切成逐人单位
    affil_txt = d["affil"]
    marks = []
    for n in names:
        m = re.search(re.escape(n) + r"\s+is with\s+", affil_txt)
        if m:
            marks.append((m.start(), m.end()))
    affils: list[str] = []
    if len(marks) == len(names):
        for i, (_s, e) in enumerate(marks):
            stop = marks[i + 1][0] if i + 1 < len(marks) else len(affil_txt)
            affils.append(affil_txt[e:stop].strip().rstrip(".").strip())
    else:
        # 切不开：整句作为共用单位，宁可不拆也不要错配到人
        affils = [affil_txt.strip().rstrip(".")] * len(names)
        print(f"⚠ Affiliation 未按姓名切开（标记 {len(marks)}/{len(names)}），"
              f"整句作共用单位")

    # 通讯作者邮箱
    email = ""
    if d["corr"]:
        m = re.search(r"`([^`]+@[^`]+)`", d["corr"])
        if m:
            email = m.group(1)

    lines = [r"\title[short]{%s}" % d["title"].replace("[short]", "")]
    lines[0] = r"\title{%s}" % d["title"]
    for i, n in enumerate(names):
        bits = n.split()
        fnm, sur = (" ".join(bits[:-1]), bits[-1]) if len(bits) >= 2 else ("", n)
        tag = r"\author*" if i == 0 else r"\author"
        aff = f"[{i + 1}]" if len(affils) > 1 else ""
        blk = rf"{tag}{aff}{{\fnm{{{fnm}}} \sur{{{sur}}}}}"
        if email and i == 0:
            blk += rf"\email{{{email}}}"
        lines.append(blk)
    for i, a in enumerate(affils):
        o = _org_parts(a)
        tag = r"\affil*" if i == 0 else r"\affil"
        inner = []
        if o["orgdiv"]:
            inner.append(r"\orgdiv{%s}" % o["orgdiv"])
        inner.append(r"\orgname{%s}" % o["orgname"])
        addr = []
        if o["city"]:
            addr.append(r"\city{%s}" % o["city"])
        if o["state"]:
            addr.append(r"\state{%s}" % o["state"])
        if o["country"]:
            addr.append(r"\country{%s}" % o["country"])
        if addr:
            inner.append(r"\orgaddress{%s}" % ", ".join(addr))
        lines.append(rf"{tag}[{i + 1}]{{{', '.join(inner)}}}")
    return "\n".join(lines)


def preamble() -> str:
    # 图以 `figures_nfr/fig_x.pdf` 这类相对路径写进 tex，构建目录在 build/sn/，
    # 故显式给 graphicspath 指向 results/（与 build_pdf.py 同一处理）。
    gp = str(B.FIGDIR.resolve())
    # natbib 由 sn-jnl 以**无选项**加载，其默认是 author-year；而 sn-basic 是编号制。
    # 于是编到 thebibliography 时报 "Bibliography not compatible with author-year
    # citations"。选项只能在包被加载前传，故写在 \documentclass 之前。
    return "\n".join([
        r"\PassOptionsToPackage{numbers}{natbib}",
        r"\documentclass[referee,lineno,sn-basic]{sn-jnl}",
        r"\usepackage{graphicx}",
        r"\usepackage{calc}",                 # 见下方 PREAMBLE_EXTRA 里 \real 的说明
        "\\graphicspath{{" + gp + "/}}",
        *PACKAGES,
    ]) + PREAMBLE_EXTRA


def write_tex(d: dict, body_tex: str, abstract_tex: str, bibitems: str,
              spacing: str) -> str:
    """按官方样例 sn-article.tex 的顺序：\\begin{document} → title/author/affil
    → abstract → keywords → \\maketitle。顺序颠倒会让 \\maketitle 拿不到题录。"""
    sp = {"double": r"\doublespacing", "half": r"\onehalfspacing",
          "single": r"\singlespacing"}[spacing]
    return rf"""{preamble()}

\begin{{document}}

{sn_front(d)}

\abstract{{{abstract_tex}}}

\keywords{{{d['keywords']}}}

\maketitle

{sp}

{body_tex}

\begin{{thebibliography}}{{99}}
{bibitems}
\end{{thebibliography}}

\end{{document}}
"""


# --------------------------------------------------------------------------- 正文微排版
def break_wide_equations(tex: str) -> str:
    """把"一行放不下的显示公式"拆成两行。**只拆行，不动任何符号、措辞与编号。**

    sn-jnl 是**单栏 372pt**，而显示公式不换行——超宽就整条冲出纸面。实测
    §4.2 的半解析形式（`F_J(x) \\simeq … , \\qquad  m_k(x) = …`）溢出 **51.9pt**；
    同样的 tex 在自建骨架 455.8pt 下是一条线放得下的，所以这条**不是内容问题，
    是栏宽问题**，两个骨架构建不必跟着改。

    断点用手稿自己的结构：`equation` 体内有一行以 `\\qquad` 结尾 —— 那正是
    markdown 里把一条公式写成两行的续接处（本稿只有这一处）。改成 `split` 后两半
    各占一行、各自对齐到自己的谓词，外层仍是 `equation`，**编号还是一个**，
    `verify_pdf` 的"公式号恰为 1..N 且按序"判据不受影响。

    只认"两行、且第一行以 \\qquad 续接"这一种形态；认不出就原样放过并**打印** ——
    静默不拆会让溢出重新变成没人看见的状态。真正的兜底是 main() 里的
    "水平溢出 > 20pt" 清单：这条从编译日志里数，拆行失效会直接出现在那儿。
    """
    out, n_break = [], 0
    lines = tex.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip() != r"\begin{equation}":
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        body: list[str] = []
        while j < len(lines) and lines[j].strip() != r"\end{equation}":
            body.append(lines[j])
            j += 1
        if j >= len(lines):
            die("equation 环境没有配对的 \\end{equation}")
        cont = [k for k, l in enumerate(body) if l.rstrip().endswith(r"\qquad")]
        if len(body) != 2 or len(cont) != 1:
            out.append(lines[i])
            out.extend(body)
            out.append(lines[j])
            i = j + 1
            continue
        first = body[cont[0]].rstrip()[: -len(r"\qquad")].rstrip().rstrip(",")
        second = body[1 - cont[0]].lstrip().rstrip().rstrip(",")
        # 各自对齐到自己的谓词：第一半的 \simeq，第二半的第一个 "="
        if not re.search(r"\\simeq", first) or "=" not in second:
            out.append(lines[i])
            out.extend(body)
            out.append(lines[j])
            i = j + 1
            continue
        out.extend([
            r"\begin{equation}",
            r"\begin{split}",
            re.sub(r"\\simeq", r"&\\simeq", first, count=1) + r",\\",
            re.sub(r"=", r"&=", second, count=1),
            r"\end{split}",
            r"\end{equation}",
        ])
        n_break += 1
        i = j + 1
    if n_break == 0:
        print("拆行  ：没有 \\qquad 续接的长公式（若编译日志报溢出 > 20pt，"
              "手动拆那条公式）")
    else:
        print(f"拆行  ：{n_break} 条长公式改为 split 两行（sn-jnl 单栏 372pt）")
    return "\n".join(out)


# --------------------------------------------------------------------------- 编译
def compile_sn(spacing: str) -> str | None:
    """pdflatex 三遍（无 BibTeX，两遍够，第三遍兜住 longtable 与交叉引用）。

    开头必须清掉 `.aux` / `.out` / `.log`：natbib 在 author-year 模式下会往 aux 写
    `\\NAT@force@numbers{}`，而本脚本用 `[numbers]` 模式、该宏不存在——**陈旧 aux 会让
    第一遍在读 aux 阶段就 fatal**（报错位置在 `./main.aux:N`，看起来像正文的错）。
    换过任何包选项之后重编也必须清。
    """
    for f in ("main.aux", "main.out", "main.log", "main.pdf", "main.nav"):
        p = BUILD / f
        if p.exists():
            p.unlink()
    (BUILD / "main.tex").write_text(
        write_tex(CURRENT["d"], CURRENT["body_tex"], CURRENT["abstract_tex"],
                  CURRENT["bibitems"], spacing))
    for i in range(3):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "main.tex"],
            cwd=BUILD, capture_output=True, text=True)
        if r.returncode and i == 2:
            return None
    if not (BUILD / "main.pdf").exists():
        return None
    return (BUILD / "main.log").read_text(errors="replace")


# 编译期要用的上下文；compile_sn 是同步单线程，用模块级变量传递比改签名干净
CURRENT: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", choices=["double", "half", "single"],
                    default="double",
                    help="行距；double = 官方 [referee] 要求（默认）")
    ap.add_argument("--repo", default=None,
                    help="仓库地址；不给则成品里保留 [repository to be inserted]")
    ap.add_argument("--campus", default=None, help="追加到单位后的校区/城市")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep", action="store_true", help="保留中间文件")
    a = ap.parse_args()

    if not MS.exists():
        die(f"找不到 {MS}")
    if not (SN / CLASS).exists():
        die(f"缺 {SN / CLASS}\n"
            f"`paper/springer_template/` 在 .gitignore 里（第三方材料，版权归 Springer），"
            f"clone 出来的仓库没有它。取法：从 Springer 官方模板包拿 sn-jnl.cls，"
            f"或 `tlmgr` 体系内对应文件，放到该目录即可。")

    if not BUILD.exists():
        BUILD.mkdir(parents=True)
    # sn-jnl.cls 复制进构建目录，构建不依赖 paper/springer_template/
    shutil.copy2(SN / CLASS, BUILD / CLASS)

    # ---------------------------------------------------------------- 解析与正文
    # 与 build_pdf.main() 完全相同的顺序：版式换了，内容不该换
    d = B.split_manuscript(MS.read_text())
    if a.repo:
        d["body"] = d["body"].replace("[repository to be inserted]", a.repo)
    if a.campus:
        d["affil"] = f"{d['affil']}, {a.campus}"

    body_md, tnums, specs = B.build_body_md(d, TEXTWIDTH_PT)
    if tnums != list(range(1, len(tnums) + 1)):
        die(f"表格编号不连续：{tnums}")

    (BUILD / "body.md").write_text(body_md)
    p = B.pandoc(["body.md", "-f", "markdown+table_captions+raw_attribute",
                  "-t", "latex", "--top-level-division=section"], cwd=BUILD)
    if p.returncode:
        die(f"pandoc 正文失败：\n{p.stderr}")
    body_tex = break_wide_equations(
        B.finish_body_tex(p.stdout, specs, d["body"]))

    p = B.pandoc(["-f", "markdown", "-t", "latex"], input=B.fix_degree(d["abstract"]))
    if p.returncode:
        die(f"pandoc 摘要失败：\n{p.stderr}")
    abstract_tex = p.stdout.strip()

    refs_in = "\n\n".join(f"ZZBIB{n}ZZ {B.fix_degree(t)}"
                          for n, t in sorted(d["refs"].items()))
    p = B.pandoc(["-f", "markdown", "-t", "latex"], input=refs_in)
    if p.returncode:
        die(f"pandoc 参考文献失败：\n{p.stderr}")
    bibitems, n_sub = re.subn(r"ZZBIB(\d+)ZZ\s*", r"\\bibitem{r\1} ", p.stdout)
    if n_sub != len(d["refs"]):
        die(f"参考文献标记只替换了 {n_sub}/{len(d['refs'])} 条")

    CURRENT.update(d=d, body_tex=body_tex, abstract_tex=abstract_tex,
                   bibitems=bibitems)
    log = compile_sn(a.spacing)
    if not log:
        die(f"sn-jnl 编译失败；见 {BUILD / 'main.log'}")

    out = Path(a.out) if a.out else OUT
    shutil.copy(BUILD / "main.pdf", out)
    if not a.keep and (BUILD / "body.md").exists():
        (BUILD / "body.md").unlink()

    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    over = [l for l in log.splitlines()
            if l.startswith("Overfull \\hbox")
            and float(re.search(r"\((\d+\.\d+)pt", l).group(1)) > 20]
    print(f"输出  ：{out}")
    print(f"类    ：sn-jnl [referee,lineno,sn-basic]    引擎：pdflatex"
          f"    行距：{ {'double': '双倍', 'half': '1.5 倍', 'single': '单倍'}[a.spacing] }")
    print(f"页数  ：{pages.group(1) if pages else '?'}     "
          f"表 {len(tnums)} 张（{tnums[0]}–{tnums[-1]}）    "
          f"图 {len(d['figs'])} 张   文献 {len(d['refs'])} 条")
    if over:
        print(f"警告  ：{len(over)} 处水平溢出 > 20pt")
        for l in over[:10]:
            print("   " + l)
    else:
        print("排版  ：无水平溢出 > 20pt")

    # ---------------------------------------------------------------- 成品反查
    # 直接用 build_pdf.verify_pdf：判据是"内容与手稿一致"，与用哪个类无关。
    # strip_lineno=True：sn-jnl 的 [lineno] 用 vruler 排行号，pdftotext 会把行号贴到
    # 公式号后面（(5)   480），切断"公式号在行尾"的判据；编号本身一个不少。
    problems = B.verify_pdf(out, d, strip_lineno=True)
    if problems:
        print(f"成品校验：发现 {len(problems)} 个问题")
        for x in problems:
            print("   ✗ " + x)
        raise SystemExit(2)
    print("成品校验：表号/图号/锚点/摘要关键句/文献条目 全部通过，无残留标记")


if __name__ == "__main__":
    main()
