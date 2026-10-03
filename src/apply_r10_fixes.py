"""第十轮（对外）：CSSP「第二轮」意见 —— 只做措辞与位置，不触实验、不重排表号。

背景：这份意见读的是全文（引用的 13 组数字全部准确），但它的六条 Major Concerns 里五条的
**实质内容已在稿中**（§4.2 的 Gram 谱展宽分析、§A.11/Table 32 的参考尺度分类、§11 的
"heuristic ... coarse detector" 定性、摘要里的 real+delay-free 限制、已改的标题）。
它似乎漏看了这些位置。因此本轮**不补实验**，只把已有内容提到更醒目处或补上显式措辞。

七条（两条在摘要、为守住 CSSP 的 250 词上限而配平）：

  1. 摘要：给自检加 "heuristic" 限定（该词此前只在 §11 出现 1 次）
  2. 摘要：把 "drifting by two orders" 的 "by" 去掉，抵掉第 1 条多出的 1 词 → 仍为 250 词
  3. §11 第四条：把 §A.8 的结构化源错误率（8/40 = 20%）提到正文
  4. §4.2：点名"活跃 Gram 的谱/条件数"，并把已测的 9.7%/8.3% 与 Gram 条件数 91→14 并列
  5. §9.1：补一句十五个配置是"覆盖设计"而非"调参集"
  6. §1 首段：把 instantaneous / delay-free / real-valued 的适用域前置
  7. §2：把 Table 32 的三分类要点内嵌到正文（该表原只在附录）

用法：
    python3 src/apply_r10_fixes.py --dry
    python3 src/apply_r10_fixes.py

_HAND_CORRECTED：条目 4 首次落稿后，把 "the condition number of that Gram" 收紧为
"the condition number of a $2\\times2$ Gram at the minimum separation"（避免把 91 读成任意列对
的 Gram 条件数；91/14 只对**夹角恰为最小间隔**的两列成立）。手稿已直接改为最终文本，
本脚本的 `new` 与之逐字一致，故重放仍幂等。
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")

E = [
    # ---------------------------------------------------------------- 1/2 摘要
    (
        "摘要：自检加 heuristic 限定",
        r"A retention self-check reverts to the classical criterion when the premise fails rather than failing silently, at a per-instance error rate near 1 in 16.",
        r"A heuristic retention self-check reverts to the classical criterion when the premise fails rather than failing silently, at a per-instance error rate near 1 in 16.",
    ),
    (
        "摘要：去掉一个 by，抵回被上一条多出的 1 词（CSSP 上限 250 词）",
        r"and shows the coefficient drifting by two orders of magnitude",
        r"and shows the coefficient drifting two orders of magnitude",
    ),
    # ---------------------------------------------------------------- 3 §11 第四条
    (
        "§11 第四条：把结构化源的 20% 错误率从 §A.8 提到正文",
        r"*Fourth*, on structured sources with overlap above $2.3$ the method is no better than a hand-tuned fixed threshold (Section 8.7).",
        r"*Fourth*, on structured sources with overlap above $2.3$ the method is no better than a hand-tuned fixed threshold (Section 8.7), and this is where the guard's decision errors concentrate: $8$ of its $40$ decisions on structured sources, a rate of $20\%$, against $2.7\%$ on the sparse ladder (Table 29).",
    ),
    # ---------------------------------------------------------------- 4 §4.2 谱/条件数
    # 2x2 单位对角 Gram（两列夹角 θ）的本征值是 1±cos θ，故条件数
    # (1+cos θ)/(1−cos θ) 在 12° 为 90.5、在 30° 为 13.9（降 6.5 倍）；
    # 而 §4.2 测得替换偏差只从 9.7% 变到 8.3%（降 1.17 倍）。
    # 两者并置正好支撑"不是邻近效应、而是谱展宽"这一判断。
    (
        "§4.2：点名活跃 Gram 的谱/条件数，并与已测偏差并列",
        r"First, it is not primarily a proximity effect: the Gram has unit diagonal, so its eigenvalues are spread even when the columns are far apart, and a full-rank $2\times2$ Gram already accounts for the $J=2$ figure — the deviation is $9.7\%$ at $12°$ and $8.3\%$ at $30°$.",
        r"First, it is not primarily a proximity effect, and what governs the deviation is the *spectrum of the active Gram* rather than the minimum column angle alone: the Gram has unit diagonal, so its eigenvalues are spread even when the columns are far apart, and a full-rank $2\times2$ Gram already accounts for the $J=2$ figure — the deviation is $9.7\%$ at $12°$ and $8.3\%$ at $30°$, so it barely moves (a factor of $1.2$) while the condition number of a $2\times2$ Gram at the minimum separation falls by a factor of six across the same range, from $91$ to $14$.",
    ),
    # ---------------------------------------------------------------- 5 §9.1 代表性
    (
        "§9.1：补一句十五配置是覆盖设计而非调参集",
        r"The design comprises fifteen configurations and eight seeds each. The clusterer, the debiased $\ell_1$ recovery and every metric are those of Section 8, so any difference is attributable to the gate.",
        r"The design comprises fifteen configurations and eight seeds each, chosen to cover the axes on which the reference scale is predicted to matter rather than to tune a coefficient: the window length fixes the number of TF points per source, the source count varies the number of directions to be resolved, the SNR sets the noise level, the two dense configurations realise the non-sparse case, and the noise type spans stationary Gaussian and non-stationary babble contamination, so a mechanism that failed to transfer across regimes would show up on at least one of these axes. The clusterer, the debiased $\ell_1$ recovery and every metric are those of Section 8, so any difference is attributable to the gate.",
    ),
    # ---------------------------------------------------------------- 6 §1 适用域前置
    # 注意：这是"句中插入"，old 必须延伸到插入点**之后**的文字（这里跨段落边界），
    # 否则 old 会是 new 的前缀 → 重跑时 old 仍在、new 也已存在，幂等失效。
    # 跨行只能靠拼接：raw 串里的 "\n" 是字面的反斜杠+n，不是换行。
    (
        "§1 首段：前置 instantaneous / delay-free / real-valued 适用域",
        r"so additional structure must be imposed." + "\n\n"
        + r"The dominant paradigm for underdetermined BSS (UBSS)",
        r"so additional structure must be imposed. We work throughout with *instantaneous, delay-free, real-valued* mixing — the setting in which the real–imaginary collinearity test is defined — and treat the complex and convolutive enlargements separately in Appendix A.13."
        + "\n\n" + r"The dominant paradigm for underdetermined BSS (UBSS)",
    ),
    # ---------------------------------------------------------------- 7 §2 内嵌分类
    (
        "§2：把 Table 32 的三分类要点内嵌进正文",
        r"Table 32 sets the classes of reference scale side by side with what each is a quantile of and whether that quantity moves with the activation probability. The noise-floor concept itself is standard",
        r"Table 32 sets the classes of reference scale side by side with what each is a quantile of and whether that quantity moves with the activation probability: a fixed fraction of the mixture maximum moves only weakly, because the maximum tracks the signal scale; a fixed percentile of the mixture moves exactly as $r(p)$ does; and a fixed multiple of the noise floor does not move by construction, which is what makes it calibratable to a target probability. The noise-floor concept itself is standard",
    ),
]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只报告将改动哪些条目")
    ap.add_argument("--manuscript", default=None, help="改这份稿件（默认 paper/manuscript.md）")
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
        # 护栏：new 不得包含完整的 old，否则重跑会无限追加（第九轮踩过）
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

    # ------------------------------------------------------------ 终检（断言最新状态）
    t = p.read_text()
    print("\n=== 终检 ===")
    for probe in (r"A heuristic retention self-check",
                  r"drifting two orders of magnitude",
                  r"a rate of $20\%$, against $2.7\%$",
                  r"spectrum of the active Gram",
                  r"from $91$ to $14$",
                  r"chosen to cover the axes on which the reference scale",
                  r"*instantaneous, delay-free, real-valued* mixing",
                  r"a fixed multiple of the noise floor does not move by construction"):
        ok = probe in t
        print(f"  {'OK  ' if ok else 'FAIL'} {probe[:64]}")
        if not ok:
            die("终检失败")
    # 条目 2 的效果：确认摘要里 'drifting by two' 已消失
    if "drifting by two orders" in t:
        die("终检失败：摘要里 'drifting by two orders' 仍在")

    # 摘要词数（CSSP 上限 250）
    m = re.search(r"## Abstract\s*\n+(.*?)\n+\s*\*\*Keywords", t, re.S)
    w = len(m.group(1).split())
    print(f"  摘要词数：{w}（CSSP 上限 250）{'  OK' if w <= 250 else '  FAIL 超限'}")
    if w > 250:
        die("摘要超过 CSSP 的 250 词上限")

    # 结构不变
    n_tab = len(re.findall(r"\*\*Table (\d+)\.\*\*", t))
    n_fig = len(re.findall(r"\*\*Fig\. (\d+)\.\*\*", t))
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    print(f"  表 {n_tab} 张 / 图 {n_fig} 张 / 附录小节 A.{secs[0]}..A.{secs[-1]}（{len(secs)} 节）")
    if n_tab != 43 or n_fig != 11 or secs != list(range(1, 19)):
        die("结构异常：表/图/附录小节数与预期不符")
    print("\n✓ 全部终检通过")


if __name__ == "__main__":
    main()
