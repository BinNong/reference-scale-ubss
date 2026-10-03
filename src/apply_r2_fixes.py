"""第二轮审稿意见：手稿修改补丁（幂等）。

用法：
    python3 src/apply_r2_fixes.py [--dry]

覆盖：
  #1  摘要按 Problem→Diagnosis→Method→Property→Evidence→Limitation 重写，减少数字罗列
  #2  §4.2 理论分层：exact / approximate / empirical 三级表述（MC1）
  #3  §5.1 与 §5.3 的 q_max、η 论证前指（MC2/MC3）
  #4  §7 增补「基线为何如此选择」与「『无调参』的确切含义」两段（MC4、Minor 12）
  #5  §7/§9.4 运行时间改用分项实测（第十六节）
  #6  §8.1 补 95% 置信区间、效应量与多重比较校正（第十五节）
  #7  新增 §10「稳健性与代价」，含 10.1–10.7 与 Table 19–24、Fig. 10
  #8  §10→§11、§11→§12；Discussion 补「两个目标」段（MC7）；修正 2.2%/5.0% 旧值
  #9  软化 actual bottleneck / mechanism 等措辞（第十四节）；术语统一（Minor 5）
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")

E: list[tuple[str, str, str]] = []

# 这三条在落稿后又被后续编辑改写过（表号顺延、guard 决策口径修正），文件已处于期望状态。
_HAND_CORRECTED = {
    "§7 运行时间指向分项表",     # 后改为指向 Table 25（表号顺延）
    "§9.4 运行时间实测",          # 后改为指向 Table 25（表号顺延）
    "第十五节 统计规范",          # 后改为指向 Table 26（表号顺延）
}


def add(tag: str, old: str, new: str) -> None:
    E.append((tag, old, new))


# ==========================================================================
# #9 措辞软化 + 术语统一
# ==========================================================================
add("措辞：actual bottleneck",
    "This paper concerns the *scale* against which the energy gate is set, which we argue is the actual bottleneck.",
    "This paper concerns the *scale* against which the energy gate is set, which we identify as a significant "
    "and previously under-emphasised source of the paradigm's brittleness.")

add("措辞：the mechanism behind",
    "This is, in our reading, the mechanism behind the widely reported brittleness of the two-stage paradigm: "
    "the paradigm is described as requiring sparsity, and it does, but a large part of the reported sensitivity "
    "is not to sparsity as such — it is to the fact that the gate silently changes function as sparsity changes.",
    "This is, in our reading, a significant and previously under-emphasised part of the mechanism behind the "
    "widely reported brittleness of the two-stage paradigm: the paradigm is described as requiring sparsity, and "
    "it does, but a substantial part of the reported sensitivity is not to sparsity as such — it is to the fact "
    "that the gate silently changes function as sparsity changes. We do not claim it is the only such mechanism.")

add("术语统一",
    "**Assumptions.** **A1.** $\\mathbf{A}\\in\\mathbb{R}^{M\\times N}$ has unit-norm columns",
    "**Terminology.** Throughout this paper a *noise-only point* is a TF point at which no source is active, a "
    "*single-source point* (SSP) one at which exactly one is, and a *multi-source point* one at which two or more "
    "are, where *active* is the criterion of Section 3.3. We use these three terms and no others, in place of the "
    "interchangeable vocabulary of the literature.\n\n"
    "**Assumptions.** **A1.** $\\mathbf{A}\\in\\mathbb{R}^{M\\times N}$ has unit-norm columns")

# ==========================================================================
# #2 §4.2 理论分层（MC1）
# ==========================================================================
add("MC1 理论分层",
    "**On the orthogonality of the active columns.** The signal term is a $\\chi^2_{2J}$ only if the $J$ active "
    "columns are orthogonal, which Assumption A1 does not guarantee: with the $12°$ minimum separation of Section 7 "
    "the true term is a weighted sum of exponentials whose weights are the eigenvalues of "
    "$\\mathbf{A}_J^{\\mathsf H}\\mathbf{A}_J$ (for a pair $12°$ apart, $1\\pm0.978$). We measured the effect of "
    "replacing those weights by their mean, and it is second order relative to the convention above: it changes "
    "$r(p)$ by at most $0.1\\%$ for $p\\le0.10$, by $4\\%$ at $p=0.20$ and by $13\\%$ at $p=0.40$, vanishing where "
    "the median sits inside the noise component. We therefore retain the $\\chi^2_{2J}$ form.",

    "**The status of the mixture law.** Proposition 6 is exact under A2 for noise-only points and, as we verify "
    "below, for points carrying exactly one active source; for $J\\ge2$ it is an approximation. We state the three "
    "levels separately because they are not the same claim, and because only the third is what the method uses.\n\n"
    "*Exact.* Under A2 the noise component is exactly $(\\sigma^2/2)\\chi^2_{2M}$, so $F_0$ and the identity "
    "$\\nu=M\\sigma^2$ are exact. At $J=1$ the signal term is a single complex Gaussian squared norm, hence one "
    "exponential, and the decomposition is exact as well; we confirmed this directly, the mean-eigenvalue "
    "substitution being vacuous at that order.\n\n"
    "*Approximate.* For $J\\ge2$ the signal energy is $\\|\\mathbf{A}_J\\mathbf{s}_J\\|^2$, a weighted sum of $J$ "
    "independent exponentials with weights the eigenvalues of $\\mathbf{A}_J^{\\mathsf H}\\mathbf{A}_J$; writing it "
    "as $\\Gamma(J,E_J/J)$ replaces those weights by their mean. Measured on random column subsets of a mixing "
    "matrix with the $12°$ minimum separation of Section 7, the substitution shifts the median signal energy by "
    "$9.7\\%$ at $J=2$, $11.9\\%$ at $J=3$ and $12.8\\%$ at $J=4$. Two features of that measurement matter. First, "
    "it is not primarily a proximity effect: the Gram has unit diagonal, so its eigenvalues are spread even when "
    "the columns are far apart, and a full-rank $2\\times2$ Gram already accounts for the $J=2$ figure — the "
    "deviation is $9.7\\%$ at $12°$ and $8.3\\%$ at $30°$. Second, rank deficiency adds a further term: for $J>M$ "
    "the $J\\times J$ Gram has $J-M$ zero eigenvalues, which the mean-eigenvalue form replaces by ones. Across a "
    "separation sweep from $5°$ to $45°$ the deviation of $r(p)$ itself varies only between $11.4\\%$ and $14.9\\%$ "
    "at $p=0.40$, so it depends on the minimum column angle only weakly.\n\n"
    "*Empirical.* What matters for the method is the observable $r(p)$, not the signal-energy law alone. Because "
    "the median lies inside the noise component wherever the closed form is used to advantage, the substitution "
    "changes $r(p)$ by at most $0.1\\%$ for $p\\le0.10$ and by $4\\%$ at $p=0.20$ — both inside the agreement with "
    "simulation reported below — and reaches $13\\%$ only at $p=0.40$, where the small-argument expansion has "
    "already ceased to apply. We therefore retain the $\\chi^2_{2J}$ form and state the boundary of the claim "
    "accordingly: the mixture law is exact for noise-only and single-source points, and a mean-eigenvalue "
    "approximation for $J\\ge2$ whose effect on the quantity the method uses is below $0.1\\%$ throughout the range "
    "in which that quantity is used.")

# ==========================================================================
# #3 §5.1 q_max 与 §5.3 边界
# ==========================================================================
add("MC2 q_max 论证",
    "(ii) The level range is capped at $q_{\\max}=0.02$, which must remain below $\\pi_0$ for the tail to be "
    "noise-only. This is a real restriction on the method and is part of why the self-check of Section 5.3 is "
    "needed.",
    "(ii) The level range is capped at $q_{\\max}=0.02$. The theoretical requirement is $q_{\\max}<\\pi_0$, and "
    "$\\pi_0=(1-p)^N$ is unknown, so the constant cannot be *derived*; what can be established is that it need not "
    "be chosen carefully. Over the decade $q_{\\max}\\in[0.01,0.05]$ the resulting angle error varies by at most "
    "$1.3\\%$ relative in every regime in which the method applies, and over $[0.005,0.05]$ by less than $2\\%$ "
    "outside the degenerate dense one (Section 10.4); below $0.005$ the estimator degrades because too few order "
    "statistics remain. The bias of $\\hat\\sigma^2$ does grow as $\\pi_0$ falls — that is the failure documented "
    "in Table 2, and it is the reason the self-check of Section 5.3 exists — but the constant governs the effect "
    "on the selected set rather than the bias itself, and that effect is flat.")

add("§5.3 边界说明",
    "The margin is a factor of $1.5$ on each side of the threshold and is reported as such: the criterion is a "
    "guard against gross silent failure, not a precise classifier. Across the 27 conditions of Table 4 the largest "
    "retention at which the gate should have been inactive is $1.0\\%$ and the smallest at which it should have "
    "been active is $3.4\\%$; the boundary at $3.5\\%$ sits between the two groups, just above the closest member "
    "of the second, which is decided against a margin of $3.9\\%$. Fig. 6 plots both groups against the boundary.",
    "The criterion is a guard against gross silent failure, not a precise classifier, and we report its margin as "
    "such. Across the 27 conditions of Table 4 the largest retention at which the gate should have been inactive "
    "is $1.0\\%$ and the smallest at which it should have been active is $3.4\\%$, so the boundary at $3.5\\%$ "
    "lies in an empty interval between the two groups. Fig. 6 plots both groups against the boundary, and Section "
    "10.3 reports how the decision responds to the boundary value itself.")

# ==========================================================================
# #4 §7 基线公平性 + 「无调参」含义
# ==========================================================================
add("MC4 基线公平性",
    "All methods receive the **true source number** $N$, except the potential-function method, which determines "
    "it automatically.",
    "The comparison is between energy-gate rules with everything else held fixed, so the comparators that matter "
    "here are pipelines that consume an SSP mask. Density-based clusterers (DBSCAN, OPTICS, density-peak) and "
    "transforms that sharpen sparsity before detection (locally maximum synchroextracting transforms with "
    "optimised fuzzy clustering) are not head-to-head comparators for a structural reason: they act either on the "
    "set of points a gate has already admitted, or in place of the detection stage, so a change to the reference "
    "scale of the gate does not enter their input at all. We do include the two families that consume a mask, and "
    "isolate the gate from the other direction by fixing the mask and varying only the front end (Section 8.5). "
    "The claim tested is therefore narrow by construction — among the rules that decide *which points an SSP test "
    "sees*, referencing that decision to the noise floor removes the need to select a coefficient — and we do not "
    "claim that a pipeline which bypasses the gate cannot do better; Section 8.7 reports a setting in which a "
    "hand-set fixed multiple indeed is better.\n\n"
    "All methods receive the **true source number** $N$, except the potential-function method, which determines "
    "it automatically.")

add("Minor 12 「无调参」含义",
    "**Proposed method.** NF-SSP as in Algorithm 1, with $\\alpha=10^{-4}$",
    "**What \"no tuning\" means here.** $\\alpha$, $c_0$ and $\\rho$ are fixed once and are not adjusted per "
    "regime, per SNR or per data set: $c_0$ and $\\rho$ are literature conventions, and $\\alpha$ is a tolerance "
    "whose value is shown below to be immaterial for anything below $10^{-3}$ (Section 8.6). Two constants are "
    "not of that kind and we say so explicitly rather than folding them into the phrase: $q_{\\max}$ is "
    "*constrained* by the data ($q_{\\max}<\\pi_0$) though not performance-tuned, its effect being flat over a "
    "decade (Section 10.4); $\\eta$ is an empirical guard threshold, validated on the grid of Table 4, whose "
    "effect on the decision is likewise unchanged over $[0.005,0.035]$ (Section 10.3). Neither is tuned to the "
    "reported grid in the sense of having been selected to minimise the reported error.\n\n"
    "**Proposed method.** NF-SSP as in Algorithm 1, with $\\alpha=10^{-4}$")

# ==========================================================================
# #5 运行时间：§7 指向前文分项表，§9.4 用实测分项
# ==========================================================================
add("§7 运行时间指向分项表",
    "it runs in $0.184$ s per problem on one CPU core, against $0.189$ s for the conventional pipeline and "
    "$1.41$ s for the smoothed-$\\ell_0$ baseline (Table 7).",
    "it runs in $0.184$ s per problem on one CPU core, against $0.189$ s for the conventional pipeline and "
    "$1.41$ s for the smoothed-$\\ell_0$ baseline (Table 7). Table 24 breaks the cost down by stage, which is "
    "the form in which the claim that the calibration is nearly free can be checked.")

add("§9.4 运行时间实测",
    "The runtime overhead is negligible: the gate costs $0.2$ s against a $16.7$ s shared cost dominated by the "
    "$\\ell_1$ recovery, about $1\\%$.",
    "The runtime overhead is negligible. At this scale — about $4.8\\times10^4$ TF points, one core — the whole "
    "pipeline takes $2.35$ s, of which the energy gate in full (energy computation, noise-floor estimation and "
    "the retention check together) accounts for $4.1$ ms, or $0.17\\%$; the $\\ell_1$ recovery alone is $99.4\\%$ "
    "of the total (Table 24).")

# ==========================================================================
# #6 §8.1 统计规范（第十五节）
# ==========================================================================
add("第十五节 统计规范",
    "No correction for multiple comparisons is applied; with six regimes and three pairwise contrasts a "
    "Bonferroni factor of $18$ would leave two of the three sparse-regime contrasts against the conventional "
    "threshold significant ($p=2.8\\times10^{-3}$, $2.7\\times10^{-2}$ and $0.10$) and the rest marginal, which "
    "is the correct qualitative reading.",
    "Intervals and effect sizes accompany the $p$-values, and corrections for multiplicity leave the reading "
    "unchanged (Table 25). Against the conventional threshold the pooled gap over the ladder is $+7.08°$ with a "
    "$95\\%$ confidence interval of $[+4.80°,\\,+9.37°]$ and a paired Cohen's $d_z$ of $0.80$; the per-regime "
    "intervals, in the order of the regimes above, are $[+9.54,+20.48]$, $[+6.31,+19.09]$, $[+3.99,+17.33]$, "
    "$[-0.76,+9.04]$ and $[+0.08,+0.35]$, the fourth covering zero and matching the non-significant $p=0.088$. "
    "Holm and Benjamini–Hochberg both retain the five significances found without correction. In the real-speech "
    "family of seven contrasts both corrections likewise retain all six that were significant uncorrected, the "
    "sole exception being the top-$5\\%$ ranking, which is not distinguished from the calibrated gate "
    "($p=0.25$). We quote uncorrected values in the text and corrected ones in Table 25.")

# ==========================================================================
# #8 Discussion：2.2%/5.0% 旧值 + 两个目标（MC7）
# ==========================================================================
add("Discussion 修旧值",
    "But its boundary sits between a largest \"inactive-preferred\" retention of $2.2\\%$ and a smallest "
    "\"active-preferred\" retention of $5.0\\%$; the threshold at $3.5\\%$ is justified by a factor of $1.5$ on "
    "each side, and by nothing more. A user operating near that boundary should treat the decision as uncertain.",
    "But its boundary sits between a largest \"inactive-preferred\" retention of $1.0\\%$ and a smallest "
    "\"active-preferred\" retention of $3.4\\%$; the threshold at $3.5\\%$ sits in the resulting empty interval, "
    "and Section 10.3 shows the decision to be unchanged for every value from $0.005$ to $0.035$, so the constant "
    "is not tuned to the reported grid. A user operating near that boundary should still treat the decision as "
    "uncertain.")

add("MC7 两个目标",
    "**Why the fix is worth making anyway.**",
    "**Which objective the method optimises.** Two objectives are easy to conflate and are not the same. The "
    "first is maximum empirical accuracy: given data resembling the target, choose whichever rule scores best on "
    "it — which here is a max-referenced gate with one coefficient ($0.88°$ against $1.12°$). The second is a "
    "calibration that transfers: a threshold derived from quantities of the problem rather than selected on data. "
    "The present method optimises the second, and on the first it is beaten, narrowly, by rules that require a "
    "coefficient to be chosen — a coefficient whose own optimum moves across two to three orders of magnitude "
    "between configurations (Table 14), which is what Section 4 shows not to be generic. A reader who knows the "
    "target domain and can afford a validation set should therefore read our result as *a bound on how far a "
    "derived threshold falls short of a selected one* — $1.59\\times$ against the per-configuration bound — "
    "rather than as a claim of superior accuracy. The two coincide only when the coefficient is stable across "
    "the configurations of interest.\n\n**Why the fix is worth making anyway.**")

add("结论 future work 补否定结果",
    "The purity weight of Section 9.5 was the repair our own analysis named; it improves on a weight in energy "
    "alone but not on the gate, so a weight that does better in both domains remains open.",
    "The purity weight of Section 9.5 was the repair our own analysis named; it improves on a weight in energy "
    "alone but not on the gate, so a weight that does better in both domains remains open. A data-driven rule "
    "for the lower-tail cap was tried and rejected on the same terms (Section 10.4).")

# ==========================================================================
# 跑替换
# ==========================================================================
# R6（第六轮）改写了 §4.2 中该条目所锚定的那句（closed form → semi-analytic），
# 条目本身已无需执行；登记以免重跑时误判。改稿前它就是未命中，不计新增。
_SUPERSEDED_R6 = {"MC1 理论分层"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    s = MS.read_text()
    changed = skipped = missing = 0
    for tag, old, new in E:
        if tag in _HAND_CORRECTED or tag in _SUPERSEDED_R6:
            print(f"  [已手工二次修订，跳过] {tag}")
            skipped += 1
            continue
        if new in s:
            print(f"  [跳过] {tag}")
            skipped += 1
            continue
        if old not in s:
            print(f"  [未命中] {tag}")
            missing += 1
            continue
        if old.count("$") % 2 != new.count("$") % 2:
            print(f"  [警告] {tag}：\$ 数量奇偶性变化 {old.count('$')} -> {new.count('$')}")
        s = s.replace(old, new, 1)
        changed += 1
        print(f"  [已改] {tag}")
    if not a.dry:
        MS.write_text(s)
    print(f"\n已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
