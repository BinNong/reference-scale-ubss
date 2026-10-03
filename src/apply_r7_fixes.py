#!/usr/bin/env python3
"""第七轮修订：一份**基于片段**的审稿意见驱动。分三档，本脚本只落实前两档。

审稿意见自称"大修"，但明说只拿到图注 + 部分参考文献。逐条核查见
`.workbuddy/memory/2026-09-19.md`。九条里 **不成立 5 / 部分成立 3 / 成立 1**：

  1. [不成立] 题注残留 `(fig_real_summary.pdf)` —— 那是给 build_pdf.py 的**构建指令标记**，
     三个成品 PDF 里 `fig_` 字样均为 0。回信指出，不改稿。
  2. [不成立] [12][21] 期刊名截断 —— 稿件里卷期页齐全，截断发生在审稿人一侧。
  3. [不成立] 文献陈旧、仅 1 篇 2025 —— 28 篇里 13 篇是 2021–2026。
  4. [不成立] §9.5 未解释加权后的胜负反转 —— §9.5 + Table 16 整节解释，且图内已披露。
  5. [不成立] 基线缺出处、未对比 DBSCAN/k-means —— §4.1 已给出处，Table 28 与 §A.14 已正面对比。
  6. [成立]   "positive values mean the calibrated gate is better" 口语化 → 改正式表述（本条）。
  7. [部分]   图注术语未自足 → 在 Fig. 11 题注内补 NF-SSP 的指代（本条）。
  8. [部分]   图 11 黑白不可分 → 已重生成加填充纹样（`r5_fig_real_summary.py`，非本脚本）。
  9. [部分]   缺 SDR/SIR/SAR —— SDR 原已报（Table 6/15、Fig 3b/8b）；**SIR 已存档未报**，
     本轮补报（Tables 40–41 + §A.20）；SAR 未存档，重跑才能补，本轮不补，正文明说。

**另有一条与审稿人无关、本轮自查发现的旧缺陷**：

 10. [自查] §9.4 的 "The conventional gate loses $2.2$ dB of SDR on average and up to $4.5$ dB"
     里，**4.5 复算不出来**。试遍全部方法对 × {逐配置, 逐实例} × {mean, median, min, max}
     × 4-seed 子集，自然口径的上限只有 3.28（NF−median）与 3.34（oracle−median）；
     `src/` 里也没有任何脚本产出过这个数。同句的 "2.2" 是可复算的（NF−median 的逐配置
     配对差均值 2.177），据此把上限改为同口径的 **3.3**，并把"相对谁"写明。
     **根因是覆盖缺口**：verify_r5/r6 对这句话没有任何断言，所以它一直没被检查到——
     现由 `verify_r7.py` 的 G 组接管。

用法：  python3 apply_r7_fixes.py [--dry]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

E: list[tuple[str, str, str]] = []


def add(tag: str, old: str, new: str) -> None:
    E.append((tag, old, new))


# ============================================ 1. Fig. 11 题注：去口语化 + 补术语指代
add("Fig. 11 题注：positive values mean... → 正式表述，并补 NF-SSP 指代",
"against the max-referenced rule; positive values mean the calibrated gate is better. (b) The same runs summarised:",
"against the max-referenced rule; a positive value indicates that the proposed calibrated gate (NF-SSP, the noise-floor-referenced construction of Section 5) attains the lower angle error. (b) The same runs summarised:")

# ============================================ 2. §9.4：SDR 上限改正 + 指向 SIR 附录
# 注意这两条必须**原子且有序**：B 先把数字改对，C 再把指针句挂上去。
# 不能写成"old=原句 / new=原句+指针"——那样 old 是 new 的前缀，重跑时会重复插入指针句。
add("§9.4 的 SDR 上限 4.5 → 3.3：4.5 在任何口径下都复算不出来（见 verify_r7 的 G 组）",
"The conventional gate loses $2.2$ dB of SDR on average and up to $4.5$ dB.",
"Against the proposed gate, the conventional one loses $2.2$ dB of SDR on average and up to $3.3$ dB.")

add("§9.4 末段：补一句指向 §A.20 的 SDR/SIR 一致性",
"up to $3.3$ dB.",
"up to $3.3$ dB. The same reading holds in the standard separation metrics: Appendix A.20 reports the signal-to-interference ratio of these runs, where the conventional gate is the last of the five columns in all fifteen configurations and the proposed gate is ahead of it in all fifteen, subject to the same qualification against the max-referenced rule that we state below.")

# ============================================ 3. 新增 §A.20 + Tables 40--41
SEC_A20 = """### A.20 Separation-quality metrics on real speech

Angle error is the quantity this paper studies, but a separation result should also be reported in the standard evaluation measures for the task. Table 40 gives the signal-to-interference ratio of the same fifteen configurations and the same eight seeds as Table 15, and Table 41 collects the configuration means of SIR and SDR alongside the mean angle error. We report SIR rather than the full triplet because SIR is the term that the direction estimate acts on: SDR and SIR are both functions of the interference residual, and it is the interference residual that the mixing matrix controls. The artifacts term is omitted — it is set by the recovery stage rather than by the gate, and the archived runs record SDR and SIR only.

**Table 40.** Signal-to-interference ratio (dB, higher is better) on real speech, on the same instances as Table 15 and averaged over the same eight seeds. SIR is the interference term of the BSS-EVAL decomposition; the direction estimate is what acts on it.

| Configuration | NF-SSP | median $t_e{=}0.02$ | max $c{=}0.05$ | top-5% | Oracle-A |
|---|---|---|---|---|---|
| win 256 | 9.14 | 6.46 | 9.08 | 8.77 | 9.22 |
| win 512 | 12.50 | 8.25 | 12.62 | 12.63 | 12.70 |
| win 1024 | 15.02 | 10.45 | 15.37 | 15.27 | 15.38 |
| win 2048 | 14.88 | 9.90 | 15.28 | 15.10 | 15.32 |
| $N=3$ | 20.40 | 14.95 | 20.49 | 20.49 | 20.47 |
| $N=5$ | 11.48 | 7.08 | 11.73 | 11.75 | 11.96 |
| $N=6$ | 9.22 | 5.51 | 9.73 | 9.69 | 9.91 |
| SNR $0$ dB | 10.56 | 5.16 | 10.70 | 10.34 | 10.78 |
| SNR $10$ dB | 13.79 | 8.79 | 14.05 | 13.93 | 14.10 |
| SNR $30$ dB | 15.31 | 13.50 | 15.65 | 15.58 | 15.68 |
| SNR $40$ dB | 15.33 | 14.94 | 15.69 | 15.62 | 15.71 |
| 1 dense source | 11.88 | 7.84 | 12.02 | 12.08 | 12.27 |
| 2 dense sources | 9.02 | 5.36 | 9.20 | 9.06 | 9.32 |
| babble noise | 14.00 | 13.98 | 15.15 | 15.06 | 15.08 |
| noise-free | 15.25 | 15.20 | 15.67 | 15.62 | 15.71 |
| **mean** | **13.19** | **9.82** | **13.49** | **13.40** | **13.57** |

**Table 41.** Configuration means of the three metrics on the same runs. The angle-error row repeats the first four columns of the mean row of Table 15; the SIR row repeats the mean row of Table 40. The oracle-aided column has zero angle error by construction and is marked "—" rather than measured as a competitor.

| Metric | NF-SSP | median $t_e{=}0.02$ | max $c{=}0.05$ | top-5% | Oracle-A |
|---|---|---|---|---|---|
| angle error ($°$) | 1.12 | 8.26 | 0.88 | 0.94 | — |
| SDR (dB) | 4.85 | 2.67 | 4.93 | 4.90 | 4.95 |
| SIR (dB) | 13.19 | 9.82 | 13.49 | 13.40 | 13.57 |

The two metrics reproduce the ordering of the angle error. Counted per configuration, the conventional median-referenced gate is the last of the five columns by SIR in all fifteen cases, and the proposed gate exceeds it in all fifteen, by $3.36$ dB on the configuration means ($13.19$ against $9.82$ dB; paired over the fifteen configuration means, $t=6.78$, $p=9\\times10^{-6}$). The reverse disclosure of Section 9.4 carries over unchanged: the max-referenced gate with a single hand-set coefficient is ahead of the unweighted proposed gate in fourteen of the fifteen configurations, by $0.31$ dB on the means ($13.49$ against $13.19$). Among the three rules that estimate $\\mathbf{A}$ well the SIR means span only $0.31$ dB and the SDR means only $0.08$ dB, so neither metric resolves their ordering — which is consistent with their angle errors of $0.88$–$1.12°$ and with the reading we give in Section 9.4. What the standard metrics establish is the part of the claim that does not depend on the choice of the angle-error figure of merit: the conventional reference scale costs $3.36$ dB of SIR, and the derived scale removes that cost without selecting anything.
"""

add("新增 §A.20 + Tables 40--41（SIR/SDR 补报）",
    "an end-to-end pipeline.",
    "an end-to-end pipeline.\n\n" + SEC_A20)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--dry", action="store_true", help="只报告将改动哪些条目")
    a = a.parse_args()

    s = MS.read_text()
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

    L = s.split("\n")
    tabs = sorted(int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s))
    misfit = sum(1 for l in L if l.count("$") % 2 == 1 and not l.strip().startswith("$$"))
    print("表号:", "连续" if tabs == list(range(1, len(tabs) + 1)) else tabs,
          f"| 共 {len(tabs)} 张 | 行内 $ 失配 {misfit}")
    figs = sorted(int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", s))
    print("图号:", "连续" if figs == list(range(1, len(figs) + 1)) else figs, f"| 共 {len(figs)} 张")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
