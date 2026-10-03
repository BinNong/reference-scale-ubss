"""R8-2  把三张附录表提到正文，并重排表号与附录小节号（CSSP 意见：Major 7 / 小意见 10）。

老师已定：**提三张（旧 Table 23/24/39）到正文**。方案见 `paper/cssp_prereview_triage.md` §11a。

## 目标结构

    新增 §10「Robustness of the calibration」插在 §9 与 Discussion 之间：
      §10.1  噪声律（旧 Table 23 -> 新 Table 19）        原附录 A.5
      §10.2  混合几何（旧 Table 24 -> 新 Table 20）      原附录 A.6
      §10.3  阶数选择（旧 Table 39 -> 新 Table 21）      原附录 A.19
      §10.4  源数目失配（新 Table 22）                   本轮新实验
    原 §10 Discussion -> §11；原 §11 Conclusion -> §12
    附录删去 A.5 / A.6 / A.19，其余小节顺延：A.7->A.5 … A.18->A.16，A.20->A.17

## 为什么必须整体重排

表号按**首次出现顺序**给（第七轮刚为此重排过图号）。三张表挪到正文后顺序变了，
其后所有附录表必须整体移号；只改这三张会造出"正文里 Table 19 之前先出现 Table 23"
这类逆序——那正是第七轮修掉的缺陷。

## 映射（同时替换，不是逐条）

表号： 23->19  24->20  39->21  |  19->23  20->24  21->25  22->26
       25..38 -> 27..40       |  40->41  41->42      （22 留给新表）

## 幂等与回退

幂等：检测到终态（§10 与 §10.4 都在位、旧标记已消失）即退出。
**不提供 `--reverse`**：结构搬迁的反向实现比它的价值更贵，做错反而比没有更糟。
回退用改前备份 `/tmp/r8_pre/paper/manuscript.md` 直接覆盖即可。

用法：
    python3 src/apply_r8b_restructure.py --dry
    python3 src/apply_r8b_restructure.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

MOVED = {5: "10.1", 6: "10.2", 19: "10.3"}            # 附录小节号 -> 正文新号
APP_MAP = {7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11,
           14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 20: 17}
TAB_MAP = {23: 19, 24: 20, 39: 21,
           19: 23, 20: 24, 21: 25, 22: 26,
           **{n: n + 2 for n in range(25, 39)},
           40: 41, 41: 42}

SEC10_TITLE = "Robustness of the calibration"
SEC10_INTRO = (
    "The studies collected here report where the calibration is, and is not, insulated from the assumptions it "
    "rests on: the law of the noise (Section 10.1), the geometry of the mixing matrix (Section 10.2), the order "
    "input (Section 10.3) and the sensitivity of the clustering to a wrong order (Section 10.4). All four are "
    "synthetic; their real-speech counterparts are Sections 9.2 and 9.4."
)

N34_TITLE = "### 10.4 Sensitivity to the assumed source number"
N34_TEXT = (
    "\nEvery method in this paper is given the true $N$, as are the baselines, and Section 10.3 measures what an "
    "order estimator would return on each mask. The complementary question is what happens when the order is "
    "*wrong by construction*: here the column count is set to $\\hat N=N+d$ with $d\\in\\{-2,\\dots,2\\}$ on the same "
    "direction sets, so that any change in error is attributable to the mismatch alone.\n\n"
    "**Table 22.** Angle error (degrees) when the clustering is given $\\hat N=N+d$ columns, on the ladder and the "
    "$N\\in\\{3,\\dots,6\\}$ cells at $p=0.05$ and $p=0.20$ (ten seeds, $1{,}200$ clusterings). The masks are those of "
    "Section 10.3, so the rows differ only in the energy criterion. A missing column costs $90°$ in the matching "
    "metric while a spurious one costs only the angle of its nearest counterpart, so the table is asymmetric by "
    "construction.\n\n"
    "| Mask | $\\hat N=N-2$ | $N-1$ | $N$ | $N+1$ | $N+2$ |\n"
    "|---|---|---|---|---|---|\n"
    "| collinearity + balance only | $45.54$ | $25.66$ | $7.51$ | $1.73$ | $1.09$ |\n"
    "| conventional median $t_e{=}0.02$ | $45.64$ | $25.49$ | $7.19$ | $1.69$ | $1.04$ |\n"
    "| **NF-SSP** | $45.45$ | $24.06$ | **$0.55$** | $0.70$ | $0.95$ |\n\n"
    "Two things follow. First, **under-counting is catastrophic and almost independent of the mask**: at "
    "$\\hat N=N-1$ all three rules land at $24$–$26°$ and at $\\hat N=N-2$ at about $45.5°$, because a missing "
    "column is penalised by $90°$ in the matching metric and no gate repairs that. Over-counting is benign for "
    "all three ($\\le1.7°$), and the ungated masks are in fact *better* at $\\hat N=N+2$ than at $\\hat N=N$, since "
    "extra clusters only subdivide the admitted directions. Second, **only at the correct order does the "
    "calibrated gate separate**: $0.55°$ against $7.19°$ and $7.51°$, a factor of thirteen. Sensitivity to "
    "under-counting also grows as $N$ falls — $60.9°$ at $N=3$ against $32.9°$ at $N=6$ for $d=-2$, since one "
    "missing column is a larger fraction of the mixture.\n\n"
    "The practical reading agrees with Section 10.3 and sharpens it: the value of the calibrated gate is "
    "realised only when the order is right, and the order is best taken from the ungated or lightly gated set. "
    "We report this as a measurement of the dependence rather than as a solution to it, and it is one more "
    "reason why every comparison in this paper supplies $N$ rather than claiming an end-to-end pipeline.\n"
)


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def ws_pat(phrase: str) -> re.Pattern:
    """正文句里的空格可能是换行，按词构造容忍空白的正则。"""
    return re.compile(r"\s+".join(re.escape(w) for w in phrase.split()))


def ws_sub(s: str, old: str, new: str) -> str:
    p = ws_pat(old)
    if not p.search(s):
        die(f"找不到待改句：{old[:64]!r}")
    return p.sub(lambda _m: new, s, count=1)


def remap_tables(text: str) -> str:
    def run(mo: re.Match) -> str:
        head, body = mo.group(1), mo.group(2)
        return head + re.sub(r"\d+", lambda m: str(TAB_MAP.get(int(m.group(0)), int(m.group(0)))), body)
    return re.sub(r"\b(Tables?)((?:\s*(?:,|and|--|–|-)?\s*\d+)+)", run, text)


def remap_appendix_refs(text: str) -> str:
    def rep(mo: re.Match) -> str:
        pre, n = mo.group(1), int(mo.group(2))
        return f"Section {MOVED[n]}" if n in MOVED else f"{pre}A.{APP_MAP.get(n, n)}"
    return re.sub(r"((?:Appendix\s+)?)A\.(\d+)", rep, text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    s = MS.read_text()
    before = s

    if f"## 10. {SEC10_TITLE}" in s and "### 10.4 Sensitivity" in s:
        print("已是目标结构，跳过。")
        return

    # ---------- 1) 切出三块 ----------------
    i_app, i_fig = s.index("## Appendix A."), s.index("## Figures")
    head, app, tail = s[:i_app], s[i_app:i_fig], s[i_fig:]
    parts = re.split(r"(?m)^(?=### A\.\d+ )", app)
    blocks: dict[int, str] = {}
    for p in parts[1:]:
        m = re.match(r"### A\.(\d+) ", p)
        if not m:
            die(f"附录块标题解析失败：{p[:60]!r}")
        blocks[int(m.group(1))] = p
    print(f"附录小节 {sorted(blocks)}")
    for n in MOVED:
        if n not in blocks:
            die(f"附录里找不到 A.{n}")

    # ---------- 2) 建 §10 ----------------
    disc = "## 10. Discussion and Limitations"
    if disc not in head:
        die("找不到 '## 10. Discussion and Limitations'")
    moved = [re.sub(r"^### A\.\d+ ", f"### {MOVED[k]} ", blocks.pop(n))
             for n, k in ((5, 5), (6, 6), (19, 19))]
    sec10 = f"## 10. {SEC10_TITLE}\n\n{SEC10_INTRO}\n\n" + "\n".join(moved)
    head = head.replace(disc, sec10 + disc.replace("## 10.", "## 11."), 1)
    head = head.replace("## 11. Conclusion", "## 12. Conclusion", 1)
    if len(blocks) != 17:
        die(f"剩余附录小节应为 17 个，实为 {len(blocks)}")

    # ---------- 3) 附录：只删掉搬走的三块，**标题号不在这里改** ----------
    # 标题号与正文引用一律交给第 4 步的 remap_appendix_refs 单趟映射。
    # 教训：第一版在这里按 APP_MAP 改了标题号，第 4 步又对同一批 A.N 映射一次，
    # 于是 old A.7 与 old A.9 撞成同一个号、两节被静默吃掉（表号却是齐全的，很难发现）。
    app2 = parts[0] + "".join(blocks[n] for n in sorted(blocks))

    s = head + app2 + tail

    # ---------- 4) 全局改号 ----------------
    s = remap_tables(s)
    s = remap_appendix_refs(s)
    # A.19 里那句指的是 Discussion（换行可能夹在中间，故用正则）
    pat = re.compile(r"Section 10(\s+)lists that as a(\s+)limitation")
    if pat.search(s):
        s = pat.sub(lambda m: f"Section 11{m.group(1)}lists that as a{m.group(2)}limitation", s, count=1)
    elif not re.search(r"Section 11\s+lists that as a\s+limitation", s):
        die("§10.3 里指向 Discussion 的那句没找到")

    # ---------- 5) §10.4 与两张表的内容补充 ----------
    anchor = "\n## 11. Discussion and Limitations"
    if anchor not in s:
        die("找不到 §11 Discussion 的锚点（§10.4 必须插在 §10 内）")
    s = s.replace(anchor, "\n" + N34_TITLE + "\n" + N34_TEXT + anchor, 1)

    row = "| uniform (light tail) | $1.98$ | $3.00$ | $5.24$ | $8/8$ | $0.47°$ vs $10.40°$ |"
    if row not in s:
        die("找不到 Table 19 的 uniform 行")
    s = s.replace(row, row + "\n| Student-$t$ ($\\nu=3$) | $0.42$ | $0.59$ | $1.21$ | $8/8$ | "
                            "$1.44°$ vs $13.12°$ |", 1)

    old_cap = "**Table 19.** The calibrated gate under five noise laws"
    if old_cap not in s:
        die(f"找不到待改句：{old_cap[:64]!r}")
    s = s.replace(old_cap, "**Table 19.** The calibrated gate under six noise laws", 1)
    s = ws_sub(s, "makes the test systematic: four noise families in addition to the Gaussian one",
               "makes the test systematic: five noise families in addition to the Gaussian one")
    s = ws_sub(s, "Its *decision* is nevertheless robust in four of the five families",
               "Its *decision* is nevertheless robust in every family tested")

    tail_txt = "therefore biases $\\hat\\sigma^2$ upward."
    if tail_txt not in s:
        die("找不到 Table 19 题注结尾")
    s = s.replace(tail_txt, tail_txt + " \"Student-$t$\" is complex Student-$t$ with $\\nu=3$. A seventh "
                  "family — the real babble interference of Section 9.4 — is reported there rather than here, "
                  "because it is a *signal contaminant* rather than a noise law: its lower tail is contaminated "
                  "by a competing signal rather than merely misshaped, and the self-check disables the gate in "
                  "half of its seeds.", 1)

    # ---------- 6) 终检 ----------
    n_tab = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s)]
    n_fig = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", s)]
    if n_tab != list(range(1, 43)):
        die(f"表题注既不齐全也不按顺序：{[n for n in n_tab]}")
    if sorted(n_fig) != list(range(1, 12)):
        die(f"图号异常：{sorted(set(n_fig))}")
    for probe in ("## 10. Robustness of the calibration", "### 10.1 ", "### 10.2 ", "### 10.3 ",
                  "### 10.4 Sensitivity", "## 11. Discussion and Limitations", "## 12. Conclusion",
                  "**Table 22.**", "**Table 19.**", "**Table 20.**", "**Table 21.**"):
        if probe not in s:
            die(f"终检失败：{probe!r} 不在文中")
    # 判据不能用"题注号"：重排后 23/24/39 会指向**别的**表（旧 19/20/37）。只能按标题判搬运。
    if "### A.19 " in s:
        die("终检失败：附录小节号 A.19 仍有残留")
    got_sections = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", s)]
    if got_sections != list(range(1, 18)):
        clash = sorted({n for n in got_sections if got_sections.count(n) > 1})
        die(f"终检失败：附录小节应为 A.1..A.17 各一次，实为 {got_sections}（重复号 {clash}）")
    i_app2 = s.index("## Appendix A.")
    body_only, app_only = s[:i_app2], s[i_app2:]
    MOVED_TITLES = ("The calibrated gate under six noise laws",
                    "Angle error (degrees) against the minimum column separation",
                    "Automatic order selection from the mask",
                    "when the clustering is given $\\hat N=N+d$ columns")
    STAY_TITLES = ("Departures from the Gaussian noise law",
                   "Sensitivity to the conditioning of the mixing matrix",
                   "The gate's strictness and automatic order selection")
    for title in MOVED_TITLES:
        if title not in body_only:
            die(f"终检失败：正文里找不到搬来的材料 {title!r}")
        if title in app_only:
            die(f"终检失败：附录里仍留着 {title!r}")
    for title in STAY_TITLES:
        if title in app_only:
            die(f"终检失败：附录里仍留着应搬走的小节 {title!r}")
    # §10.4 必须落在 §10 之内（第一版曾把它插到 §12 之后——位置错、其余全绿，正是这类静默错）
    i10, i104 = s.index("## 10. Robustness of the calibration"), s.index("### 10.4 ")
    i11 = s.index("## 11. Discussion and Limitations")
    if not (i10 < i104 < i11):
        die(f"终检失败：§10.4 不在 §10 内（10 在 {i10}、10.4 在 {i104}、11 在 {i11}）")
    if re.search(r"(?m)^### A\.(1[89]|2\d) ", s):
        die("终检失败：附录仍出现 A.18 以上的小节号")
    if s.count("$") % 2:
        die(f"$ 不配对：{s.count('$')}")

    print(f"表 {len(n_tab)} 张，题注顺序 1..42 递增；图 {len(n_fig)} 张")
    print(f"词数 {len(before.split())} -> {len(s.split())}")

    if a.dry:
        print("\n[dry] 未写入。")
        return
    MS.write_text(s)
    print(f"\n已写入 {MS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
