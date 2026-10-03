#!/usr/bin/env python3
"""把 paper/manuscript.md 编译成 CSSP 投稿版 PDF（Springer）。

CSSP = Circuits, Systems and Signal Processing（Springer US，ISSN 0278-081X）。

**评审模式：单盲（single-anonymous）** —— 2026-10-03 由 Springer 官方两页确证：
`journal/34/ethics-and-disclosures` 明写 "peer reviewed (single-anonymous)"；
`journal/34/submission-guidelines` 要求 **title page 含作者姓名/单位/通讯邮箱**。
故**默认不匿名**，作者信息照常排在首页；`--anon` 保留给将来投双盲刊时使用
（那时才清空作者块、并把仓库地址换成中性表述）。

与 build_pdf.py 的分工：**复用**它的解析 / 表格改写 / pandoc / xelatex 编译 / 成品
反向校验（`import build_pdf as B`），只覆盖 CSSP 特有的两处：

  1. **匿名化（仅 `--anon`）** —— CSSP 是**单盲**（见上），默认**保留**作者信息；
     只有将来投双盲刊时才把作者姓名 / 单位 / 通讯邮箱从正文清掉、
     改由单独的 title page（`paper/cssp_title_page.md`）承载。
  2. **参考文献 IEEE → Springer 风格** —— 原稿为 IEEE（标题带引号、vol./no./pp.）；
     Springer Basic 为 `Author A, Author B (YEAR) Title. Journal V(N):P–Q`。

已逐条核对**无需改动**的三项（依据见 .workbuddy/memory/2026-09-26.md）：
  * 正文引用已是方括号编号 `[1,2]` —— CSSP 要求 `[1,2]` / `[1–4]`，一致；
  * 摘要 **248 词** ≤ 250 上限；
  * 关键词 **6 个**，落在 CSSP 要求的 4–6 之间。

**引擎说明**：Springer 官方模板 sn-jnl.cls 在本机 TeX Live **basic** 方案下缺依赖
（cuted / breakurl / wrapfig），而 `tlmgr` 对 /usr/local/texlive 无写权限，补不了包；
官方示例 sn-article.tex 实测编译失败。故继续用 build_pdf.py 已验证的
`article + xelatex + unicode-math` 骨架 —— Springer 对 sn-jnl 的定位本就是
"content first，不复制期刊版式"，最终版式由其生产部门完成。sn-jnl 模板包已归档在
`paper/springer_template/`，换到 TeX Live 装全的机器即可切换 documentclass。

**绝不修改 paper/manuscript.md**：前七轮 verify_*.py 与全部 applier 都依赖它。

用法：
    python3 src/build_pdf_cssp.py --refs      # 只打印 28 条参考文献的转换结果，供核对
    python3 src/build_pdf_cssp.py             # 构建 paper/manuscript_cssp.pdf
    python3 src/build_pdf_cssp.py --repo <URL>  # 指定仓库地址（去掉占位符）
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pdf as B  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MS = PAPER / "manuscript.md"
OUT = PAPER / "manuscript_cssp.pdf"


# --------------------------------------------------------------------- 参考文献转换
def flip_authors(a: str) -> str:
    """`P. Bofill, M. Zibulevsky, S. Araki et al.` -> `Bofill P, Zibulevsky M, Araki S et al.`

    IEEE 是「缩写 姓」，Springer Basic 是「姓 缩写」。`et al.` 要挂在最后一位作者之后，
    不能被当成姓氏切分（`S. Araki et al.` 里最后一个空白 token 是 `al.`）。
    """
    out = []
    for name in [x.strip() for x in a.split(",") if x.strip()]:
        tail = ""
        m = re.match(r"^(.*?)\s+et\s+al\.?$", name)
        if m:
            name, tail = m.group(1).strip(), " et al."
        parts = name.split()
        if len(parts) < 2:
            out.append(name + tail)          # 机构名或单字，原样保留
            continue
        # Springer Basic 的首字母**不带点**：`D. W. C. Ho` -> `Ho D W C`
        out.append(f"{parts[-1]} {re.sub(r'\.', '', ' '.join(parts[:-1]))}{tail}")
    return ", ".join(out)


# 期刊：`vol. 81, no. 11, pp. 2353–2362, 2001` / `vol. 36, pp. …` / `vol. 13, no. 9, 1677, 2021`
_J_VOLNO = re.compile(r"^vol\.\s*(?P<vol>[^,]+?)(?:,\s*no\.\s*(?P<no>[^,]+?))?,\s*"
                      r"(?:pp\.\s*(?P<pp>[^,]+?),\s*)?(?P<art>[^,]*?),?\s*(?P<year>\d{4})$")


def ieee_to_springer(ref: str) -> str:
    """单条 IEEE -> Springer Basic。无法识别时原样返回（调用方会核对）。"""
    t = ref.strip()
    note = ""
    m = re.search(r"\s*\((in Chinese|in [A-Za-z ]+)\)\s*\.?$", t)
    if m:
        note = f" ({m.group(1)})"          # 注记整块保留括号：`(in Chinese)`
        t = t[: m.start()].strip().rstrip(".")
    t = t.rstrip(".")

    # ⓪ 无作者的在线资源 / 数据集：`"标题," 出版方, 年. [Online]. Available: URL`
    #    SLR28 这类数据集没有自然作者，只有托管方与 URL，走不到下面的 ①。
    m0 = re.match(r'^"(?P<title>.+?),?"\s*(?P<rest>.+)$', t)
    if m0:
        r0 = m0.group("rest").strip()
        mo0 = re.search(r"\[Online\]\.?\s*Available:\s*(\S+)\s*$", r0)
        if mo0:
            ym = re.search(r"(\d{4})", r0)
            venue = re.sub(r"[,\s]*\d{4}\.?\s*$", "", r0[: mo0.start()]).strip().rstrip(".,")
            return (f"({ym.group(1) if ym else ''}) {m0.group('title').rstrip(',')}."
                    + (f" {venue}." if venue else "") + f" {mo0.group(1)}") + note

    # ① 作者, "标题," ...（期刊 / 会议 / 预印本）
    m = re.match(r'^(?P<auth>.+?),\s*"(?P<title>.+?),?"\s*(?P<rest>.+)$', t)
    if m:
        auth = flip_authors(m.group("auth"))
        title = m.group("title").rstrip(",")
        rest = m.group("rest").strip()

        # 有作者的在线资源 / 数据集：`出版方, 年. [Online]. Available: URL`
        mo = re.search(r"\[Online\]\.?\s*Available:\s*(\S+)\s*$", rest)
        if mo:
            ym = re.search(r"(\d{4})", rest)
            venue = re.sub(r"[,\s]*\d{4}\.?\s*$", "", rest[: mo.start()]).strip().rstrip(".,")
            return (f"{auth} ({ym.group(1) if ym else ''}) {title}."
                    + (f" {venue}." if venue else "") + f" {mo.group(1)}") + note

        # 带星号侧写：期刊 / 会议都长成 `*venue*, <tail>`
        mv = re.match(r'^\*(?P<venue>.+?)\*\s*,\s*(?P<tail>.+)$', rest)
        if mv:
            venue, tail = mv.group("venue"), mv.group("tail").strip()
            mm = _J_VOLNO.match(tail)
            if mm:                                    # 期刊
                vol, no, pp = mm.group("vol"), mm.group("no"), mm.group("pp")
                page = pp or (mm.group("art") or "").strip()
                out = f"{auth} ({mm.group('year')}) {title}. {venue} {vol}"
                if no:
                    out += f"({no})"
                return (out + (f":{page}" if page else "")) + note
            ym = re.search(r"(\d{4})", tail)          # 会议
            pp = re.search(r"pp\.\s*([^,]+)", tail)
            out = f"{auth} ({ym.group(1) if ym else ''}) {title}. In: {venue}"
            return (out + (f", pp {pp.group(1).strip()}" if pp else "")) + note

        ma = re.search(r"(arXiv:[\w.]+)", rest)       # 预印本
        if ma:
            ym = re.search(r"(\d{4})", rest)
            return f"{auth} ({ym.group(1) if ym else ''}) {title}. {ma.group(1)}{note}"

    # ② 书：`作者, *书名*. 出版社, 年.`
    m2 = re.match(r'^(?P<auth>.+?),\s*\*(?P<title>.+?)\*\s*\.?\s*(?P<rest>.*)$', t)
    if m2:
        ym = re.search(r"(\d{4})", m2.group("rest"))
        pub = re.sub(r",?\s*\d{4}\.?\s*$", "", m2.group("rest")).strip().rstrip(".")
        return (f"{flip_authors(m2.group('auth'))} ({ym.group(1) if ym else ''}) "
                f"{m2.group('title')}. {pub}{note}")

    return ref                                        # 认不出来 -> 原样（调用方列出待核对）


# --------------------------------------------------------------------- 匿名化
ANON = {"authors": "", "affil": "", "corr": ""}

# 双盲稿**不能**指向个人仓库：仓库首页通常带作者名与机构，等于自曝身份。
# 默认换成中性表述；老师给了匿名地址（anonymous.4open.science 之类）就用它。
ANON_REPO = "an anonymised repository (link withheld for double-blind review)"


def collapse_ranges(text: str) -> str:
    """`[3,4,5,6,7,8,9,10]` -> `[3–10]`；`[15,16,17,23]` -> `[15–17,23]`。

    CSSP 的多引用写法：两个仍写 `[1,2]`，三个及以上的连续段用 en dash（`[1–4]`）。
    只压缩**连续 ≥3** 的段，`[1,2]` 这类短引用保持原样。
    """
    def rep(m: re.Match) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if len(nums) < 3:
            return m.group(0)
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if j - i >= 2:
                out.append(f"{nums[i]}\u2013{nums[j]}")
            else:
                out.extend(str(x) for x in nums[i:j + 1])
            i = j + 1
        return "[" + ",".join(out) + "]"

    return re.sub(r"\[[\d,\s]+\]", rep, text)


def anonymise(d: dict) -> dict:
    """双盲：正文里不留任何身份线索，作者信息移交 title page。"""
    d = dict(d)
    d.update(ANON)
    return d


# --------------------------------------------------------------------- LaTeX 写出（匿名标题页）
def write_tex_cssp(d: dict, body_tex: str, bibitems: str, abstract_tex: str,
                   fonts: tuple[str, str], half: bool) -> str:
    """与 build_pdf.write_tex 相同；仅当作者信息为空（`--anon`）时才省略标题页的作者块。"""
    author = f"\n\\author{{{B.author_block(d)}}}" if d.get("authors") else ""
    return rf"""{B.preamble(*fonts)}

{B.hypersetup(d)}

\title{{\large\bfseries {d['title']}}}{author}
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


# --------------------------------------------------------------------- CSSP 专有校验
def verify_cssp(pdf: Path, d: dict, anon: bool = False) -> list[str]:
    """在 build_pdf 的成品校验之外，补两条 CSSP 专有判据。

    `anon=True`（即 `--anon`）时才回查"匿名版里不得出现作者信息"——
    单盲的默认版本本来就把作者信息排在第一页。
    """
    bad = list(B.verify_pdf(pdf, d))
    txt = B.subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                           capture_output=True, text=True).stdout
    flat = re.sub(r"\s+", " ", txt)

    # 1) 匿名性：**仅 `--anon` 时**回查（单盲默认版的第一页就有作者信息）
    if anon:
        # 模式从**手稿头部**动态构造：本仓库是公开的，不把真实姓名/邮箱/单位写进脚本。
        src = MS.read_text()
        au = re.search(r"(?m)^\*\*Authors:\*\*\s*(.+)$", src)
        af = re.search(r"(?m)^\*\*Affiliation:\*\*\s*(.+)$", src)
        co = re.search(r"(?m)^\*\*Corresponding author:\*\*\s*(.+)$", src)
        ident: list[tuple[str, str]] = []
        if au:
            for nm in re.split(r",|\s+and\s+", au.group(1)):
                if nm.strip():
                    ident.append((re.escape(nm.strip()), "作者姓名"))
        if af and af.group(1).strip():
            ident.append((re.escape(af.group(1).strip()), "单位"))
        if co:
            mm = re.search(r"[\w.+-]+@[\w.-]+\.\w+", co.group(1))
            if mm:
                ident.append((re.escape(mm.group(0)), "通讯邮箱"))
        for pat, what in ident:
            if re.search(pat, flat, re.I):
                bad.append(f"匿名性：正文里出现{what}（双盲稿不该有）")

    # 2) 参考文献格式：不该再有 IEEE 痕迹
    for pat, what in ((r"\bvol\.\s*\d", "IEEE 的 `vol.` 标记"),
                      (r"\bno\.\s*\d", "IEEE 的 `no.` 标记"),
                      (r"\bpp\.\s*\d", "IEEE 的 `pp.` 标记")):
        n = len(re.findall(pat, flat))
        if n:
            bad.append(f"参考文献格式：仍残留 {n} 处{what}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", action="store_true", help="只打印参考文献转换结果，供核对")
    ap.add_argument("--repo", default=None, help="代码/数据仓库地址；不给则保留占位符")
    ap.add_argument("--anon", action="store_true",
                    help="双盲专用：清空作者信息并把仓库地址换成中性表述"
                         "（CSSP 是单盲，默认不要开）")
    ap.add_argument("--spacing", choices=["half", "single"], default="half")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not MS.exists():
        B.die(f"找不到 {MS}")
    d = B.split_manuscript(MS.read_text())

    # 参考文献：逐条转换
    conv = {n: ieee_to_springer(t) for n, t in d["refs"].items()}
    if a.refs:
        for n in sorted(d["refs"]):
            print(f"[{n:>2}] 旧: {d['refs'][n]}")
            print(f"     新: {conv[n]}")
            print()
        left = [n for n in sorted(d["refs"]) if conv[n] == d["refs"][n]]
        print(f"共 {len(conv)} 条；未能自动转换需人工核对：{left if left else '无'}")
        return
    d["refs"] = conv

    # CSSP 是单盲：默认保留作者信息与真实仓库地址；只有 --anon 才做双盲处理。
    if a.repo:
        d["body"] = d["body"].replace("[repository to be inserted]", a.repo)
    elif a.anon:
        d["body"] = d["body"].replace("[repository to be inserted]", ANON_REPO)
    d["body"] = collapse_ranges(d["body"])

    if a.anon:
        d = anonymise(d)

    body_md, tnums, specs = B.build_body_md(d)
    if tnums != list(range(1, len(tnums) + 1)):
        B.die(f"表格编号不连续：{tnums}")

    B.BUILD = ROOT / "build" / "cssp"
    if not B.BUILD.exists():
        B.BUILD.mkdir(parents=True)
    B.write_tex = write_tex_cssp          # compile_pdf 内部按模块全局查找该函数
    (B.BUILD / "body.md").write_text(body_md)

    p = B.pandoc(["body.md", "-f", "markdown+table_captions+raw_attribute",
                  "-t", "latex", "--top-level-division=section"], cwd=B.BUILD)
    if p.returncode:
        B.die(f"pandoc 正文失败：\n{p.stderr}")
    body_tex = B.star_appendix(B.rewrite_tables(p.stdout, specs))

    p = B.pandoc(["-f", "markdown", "-t", "latex"], input=B.fix_degree(d["abstract"]))
    if p.returncode:
        B.die(f"pandoc 摘要失败：\n{p.stderr}")
    abstract_tex = p.stdout.strip()

    refs_in = "\n\n".join(f"ZZBIB{n}ZZ {B.fix_degree(t)}" for n, t in sorted(d["refs"].items()))
    p = B.pandoc(["-f", "markdown", "-t", "latex"], input=refs_in)
    if p.returncode:
        B.die(f"pandoc 参考文献失败：\n{p.stderr}")
    bibitems, n_sub = re.subn(r"ZZBIB(\d+)ZZ\s*", r"\\bibitem{r\1} ", p.stdout)
    if n_sub != len(d["refs"]):
        B.die(f"参考文献标记只替换了 {n_sub}/{len(d['refs'])} 条")

    log, used = None, None
    for fs in B.FONTSETS:
        log = B.compile_pdf(fs, d, body_tex, bibitems, abstract_tex, a.spacing == "half")
        if log:
            used = fs
            break
    if not log:
        B.die("字体组合均编译失败；见 build/cssp/main.log")

    out = Path(a.out) if a.out else OUT
    shutil.copy(B.BUILD / "main.pdf", out)

    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    n_words = len(d["abstract"].split())
    print(f"输出  ：{out}")
    print(f"字体  ：{used[0] or '引擎默认'} / 数学 {used[1] or '引擎默认'}    引擎：xelatex")
    print(f"页数  ：{pages.group(1) if pages else '?'}")
    print(f"稿源  ：表 {len(tnums)} 张（{tnums[0]}–{tnums[-1]}）   图 {len(d['figs'])} 张   "
          f"文献 {len(d['refs'])} 条   摘要 {n_words} 词   关键词 "
          f"{len(d['keywords'].split(';'))} 个")
    print("模板  ：article（本机 TeX Live basic 缺 sn-jnl 依赖；官方模板见 paper/springer_template/）")

    problems = verify_cssp(out, d, a.anon)
    if problems:
        print(f"成品校验：发现 {len(problems)} 个问题")
        for x in problems:
            print("   ✗ " + x)
        raise SystemExit(2)
    scope = "匿名性（--anon）" if a.anon else "单盲默认版，作者信息保留在首页"
    print(f"成品校验：表号/图号/锚点/文献条目 + {scope} + 非 IEEE 格式，全部通过")


if __name__ == "__main__":
    main()
