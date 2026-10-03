r"""一键生成/刷新投稿材料。

产出两处（都在 .gitignore 里，**不进公开仓库**）：

  paper/submission_latex/   LaTeX 源文件包（投稿的"可编辑源文件"）
      manuscript.tex           从 build/cssp/main.tex 派生；graphicspath 改相对
      Fig1.pdf … Fig11.pdf     11 张图，按稿中 Fig. 1–11 对应关系扁平化命名
      manuscript.pdf           在包内编译一遍（验证源文件自洽）

  paper/submission/         上传包（Editorial Manager 里逐项对应）
      manuscript.pdf           = paper/manuscript_cssp.pdf（Springer 文献格式、含作者）
      manuscript_source.zip    manuscript.tex + Fig*.pdf
      title_page.pdf           由 paper/cssp_title_page.md 转（纯英文投稿版）
      cover_letter.pdf         由 paper/cover_letter_cssp.md 转
      README.txt               上传对照表

前提（先跑这两条，确保正文是最新的）：
    python3 src/build_pdf.py      --repo <URL>
    python3 src/build_pdf_cssp.py --repo <URL>

用法：
    python3 src/pack_submission.py
"""

import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
LATEX = ROOT / "paper" / "submission_latex"
SUB = ROOT / "paper" / "submission"

# 稿中 Fig.n → 源文件（与 README 的 figures 表一致）
FIGMAP = [
    ("figures_nfr/fig_scale.pdf", "Fig1.pdf"),
    ("figures_nfr/fig_guard.pdf", "Fig2.pdf"),
    ("figures_nfr/fig_architecture.pdf", "Fig3.pdf"),
    ("figures_nfr/fig_sparsity.pdf", "Fig4.pdf"),
    ("figures_nfr/fig_snr.pdf", "Fig5.pdf"),
    ("figures_nfr/fig_nsources.pdf", "Fig6.pdf"),
    ("figures_nfr/fig_scaling.pdf", "Fig7.pdf"),
    ("figures_real/fig_real_scale.pdf", "Fig8.pdf"),
    ("figures_real/fig_real_main.pdf", "Fig9.pdf"),
    ("figures_real/fig_real_summary.pdf", "Fig10.pdf"),
    ("figures_nfr/fig_heatmap.pdf", "Fig11.pdf"),
]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def build_latex() -> None:
    src = ROOT / "build" / "cssp" / "main.tex"
    if not src.exists():
        die("缺 build/cssp/main.tex —— 先跑 python3 src/build_pdf_cssp.py --repo <URL>")
    tex = src.read_text()

    if LATEX.exists():
        shutil.rmtree(LATEX)
    LATEX.mkdir(parents=True)

    for rel, dst in FIGMAP:
        s = ROOT / "results" / rel
        if not s.exists():
            die(f"缺图：{s}")
        shutil.copy2(s, LATEX / dst)
        old = "{" + rel + "}"
        if tex.count(old) != 1:
            die(f"{rel} 在 tex 中出现 {tex.count(old)} 次（期望 1）")
        tex = tex.replace(old, "{" + dst + "}", 1)

    tex, n = re.subn(r"\\graphicspath\{\{[^}]*\}\}", r"\\graphicspath{{./}}", tex)
    if n != 1:
        die(f"graphicspath 替换命中 {n} 次")
    left = re.findall(r"/Users/\S+", tex)
    if left:
        die(f"仍残留绝对路径：{left[:3]}")

    (LATEX / "manuscript.tex").write_text(tex)
    print(f"  ✓ submission_latex/manuscript.tex（{len(tex):,} 字符）")
    print(f"  ✓ submission_latex/Fig1..Fig{len(FIGMAP)}.pdf")

    # 包内编译一遍：源文件必须自洽
    for i in (1, 2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "manuscript.tex"],
                           cwd=LATEX, capture_output=True, text=True)
    pdf = LATEX / "manuscript.pdf"
    if not pdf.exists():
        die("源文件包编译失败，见 paper/submission_latex/manuscript.log")
    nerr = len(re.findall(r"(?m)^!", (LATEX / "manuscript.log").read_text(errors="replace")))
    print(f"  ✓ 源文件包编译通过（{nerr} 个 LaTeX 错误）")
    for junk in ("manuscript.aux", "manuscript.log", "manuscript.out"):
        (LATEX / junk).unlink(missing_ok=True)


def build_bundle() -> None:
    for need in (ROOT / "paper" / "manuscript_cssp.pdf", ROOT / "paper" / "cssp_title_page.md",
                 ROOT / "paper" / "cover_letter_cssp.md"):
        if not need.exists():
            die(f"缺 {need}")

    if SUB.exists():
        shutil.rmtree(SUB)
    SUB.mkdir(parents=True)

    shutil.copy2(ROOT / "paper" / "manuscript_cssp.pdf", SUB / "manuscript.pdf")
    print("  ✓ submission/manuscript.pdf")

    with zipfile.ZipFile(SUB / "manuscript_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(LATEX / "manuscript.tex", "manuscript.tex")
        for p in sorted(LATEX.glob("Fig*.pdf")):
            z.write(p, p.name)
    print(f"  ✓ submission/manuscript_source.zip（1 tex + {len(list(LATEX.glob('Fig*.pdf')))} 图）")

    tmp = SUB / "_cover_letter.md"
    tmp.write_text((ROOT / "paper" / "cover_letter_cssp.md").read_text())
    for src, pdf in ((ROOT / "paper" / "cssp_title_page.md", "title_page.pdf"),
                     (tmp, "cover_letter.pdf")):
        r = subprocess.run(["pandoc", str(src), "-o", str(SUB / pdf), "--pdf-engine=xelatex",
                            "-V", "geometry:margin=2.5cm", "-V", "fontsize=11pt"],
                           capture_output=True, text=True)
        if r.returncode:
            die(f"{pdf} 转换失败：{r.stderr[:200]}")
        print(f"  ✓ submission/{pdf}")
    tmp.unlink()

    # 投稿件里不该有中文（内部注记只能待在 *_notes.md 里）
    for f in ("title_page.pdf", "cover_letter.pdf"):
        t = subprocess.run(["pdftotext", str(SUB / f), "-"], capture_output=True, text=True).stdout
        hz = re.findall(r"[\u4e00-\u9fff]", t)
        if hz:
            die(f"{f} 里仍有 {len(hz)} 个中文字符——内部注记混进投稿件了")
    print("  ✓ 投稿件无中文残留")

    write_readme()


def write_readme() -> None:
    """上传对照表。数字与仓库地址都**现取**（手稿 / 成品 PDF），不在脚本里写死——
    本仓库是公开的，脚本里不出现姓名、邮箱与地址。"""
    md = (ROOT / "paper" / "manuscript.md").read_text()
    cl = (ROOT / "paper" / "cover_letter_cssp.md").read_text()

    n_tab = len(re.findall(r"\*\*Table \d+\.\*\*", md))
    n_fig = len(re.findall(r"\*\*Fig\. \d+\.\*\*", md))
    n_ref = len(re.findall(r"(?m)^\[\d+\]", md))
    pages = subprocess.run(["pdfinfo", str(SUB / "manuscript.pdf")],
                           capture_output=True, text=True).stdout
    m = re.search(r"(?m)^Pages:\s*(\d+)", pages)
    n_pg = m.group(1) if m else "?"
    m = re.search(r"https?://\S*github\.com/[^\s>,]+", md)
    repo = m.group(0) if m else "(未在手稿里找到仓库地址)"
    m = re.search(r"(?m)^\*\*Date:\*\*\s*(.+)$", cl)
    date = m.group(1).strip() if m else "(未填日期)"

    text = f"""Upload checklist
Circuits, Systems, and Signal Processing (Springer US) -- Editorial Manager
https://www.editorialmanager.com/cssp/

Produced by `python3 src/pack_submission.py`. Do not edit by hand: rerun it.

  Item in Editorial Manager    File in this folder       Notes
  --------------------------   -----------------------   --------------------------------
  Cover Letter                 cover_letter.pdf          letter dated {date}
  Title Page                   title_page.pdf            authors, affiliations, statements
  Manuscript                   manuscript.pdf            Springer reference style,
                                                         author block on page 1
  Editable source files        manuscript_source.zip     manuscript.tex + Fig1..Fig{n_fig}

Manuscript: {n_pg} pages, {n_tab} tables, {n_fig} figures, {n_ref} references.

Notes
  * Editable source files are mandatory at this journal ("Failing to submit a complete
    set of editable source files will result in your article not being considered for
    review"); they are in manuscript_source.zip.
  * All figures are placed within the body of the manuscript PDF, so no separate figure
    upload is required. If the system asks for them separately, extract Fig1.pdf ...
    Fig{n_fig}.pdf from manuscript_source.zip -- the names already follow the required
    "Fig<number>" pattern.
  * Code, fixed configurations and every machine-readable record:
    {repo}

The manuscript and this bundle are not part of the public repository above; only code,
configurations, records and the README are.
"""
    (SUB / "README.txt").write_text(text)
    print("  ✓ submission/README.txt")


def main() -> None:
    print("=== 1/2 LaTeX 源文件包 ===")
    build_latex()
    print("\n=== 2/2 上传包 ===")
    build_bundle()
    print(f"\n✓ 完成。上传材料在 {SUB}")
    print("  提醒：title_page 的 CRediT 分工与致谢若改了，重跑本脚本即可刷新 PDF。")


if __name__ == "__main__":
    main()
