#!/usr/bin/env python3
"""第四轮审稿修订：手稿文本修改（内容部分）。

对应意见：M1–M7、Minor 1–5、8–10，以及 Questions 1–3。
结构改动（§10 移入附录、章节重编号）见 apply_r4_structure.py。

用法:
  python3 apply_r4_fixes.py          # 应用
  python3 apply_r4_fixes.py --dry    # 只报告命中情况
"""
from __future__ import annotations

import argparse
import pathlib
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

E: list[tuple[str, str, str]] = []


def add(tag, old, new):
    E.append((tag, old, new))


# ============================ A. 摘要（M2/M3/M6/M7/Minor 9）====================
NEW_ABSTRACT = (
    'Two-stage underdetermined blind source separation (UBSS) clusters the directions of time–frequency (TF) observations passing a *single-source point* test, whose energy gate discards noise-only points. For instantaneous, delay-free mixtures we show that the *reference scale* conventionally used to set that gate — a fixed multiple of the median or maximum energy — is not invariant to source sparsity, and that this drift, not the clustering step, accounts for a substantial part of the brittleness of the tested pipelines. A mixture-law analysis gives that ratio in closed form, locates its boundary exactly, and shows the coefficient changing meaning by nearly two orders of magnitude. We reference the gate to the noise floor instead, estimating it blindly from the lower tail via the exact chi-square quantile relation, and calibrate the multiple to a false-admission rate — a threshold *derived*, not selected. On a synthetic ladder it tracks a per-regime oracle without per-regime knowledge, on real speech it removes a silent order-of-magnitude penalty, and is insensitive to both design constants. A retention self-check reverts to the classical criterion when the premise fails rather than failing silently, at a per-instance error rate near one in sixteen. Two limits bound the claims: the collinearity test needs a real, delay-free mixing matrix, so convolutive mixtures and large arrays are out of reach; and on real speech the optimal gate is set by the energy-dependence of point reliability, and a max-referenced gate with one chosen coefficient is marginally more accurate — the derived threshold offers transferability, not accuracy.'
)

# ============================ B. 引言：范围前置（M2）、叙事重心（M3）=========
add(
    "M2 引言：适用范围前置",
    "Read as a whole the paper proceeds in four steps \u2014 a reference-scale *diagnosis*, a calibration "
    "that is *derived* rather than selected, a *synthetic validation* in which the closed form is checked "
    "against the law it assumes, and a *qualification* of the mechanism on real speech \u2014 and its "
    "claims are bounded by that sequence rather than by the diagnosis alone.",
    "Three restrictions define what the paper covers, and we state them before the contributions rather "
    "than in the limitations. The mixing is **instantaneous and delay-free with a real matrix**, which is "
    "not a technical convenience: the real\u2013imaginary collinearity criterion that the whole SSP stage "
    "rests on exists only for such mixtures, so for the convolutive case (the common one in acoustic "
    "practice) that stage, and therefore this construction, does not apply. The noise is **isotropic "
    "complex Gaussian within the band analysed**; the \u00a78.6 and Appendix measurements show what "
    "survives when it is not, and what does not. And the benefit is **concentrated at small sensor "
    "counts**, because the noise-only admission rate of the collinearity test falls from $0.1275$ at "
    "$M=2$ to $0.0034$ at $M=4$ (Corollary 1), leaving progressively less noise for the gate to remove "
    "and shrinking the gain to about a factor of two. Two-sensor, severely underdetermined separation is "
    "the setting the method is for.\n\n"
    "Read as a whole the paper proceeds in four steps \u2014 a reference-scale *diagnosis*, a calibration "
    "that is *derived* rather than selected, a *synthetic validation* in which the closed form is checked "
    "against the law it assumes, and a *qualification* of the mechanism on real speech \u2014 and its "
    "claims are bounded by that sequence rather than by the diagnosis alone.",
)

add(
    "M3 贡献 5 改为以逼近 oracle 为首要主张",
    "On LibriSpeech mixtures across fifteen configurations a conventionally set fixed threshold costs a "
    "mean factor of $12.8$ (up to $31$) against a per-configuration bound, while the calibrated gate with "
    "a single $\\alpha$ reduces the error to $1.12°$ on average, within $1.59\\times$ of that bound, the "
    "gate accounting for $0.17\\%$ of the runtime (Table 25).",
    "On LibriSpeech mixtures across fifteen configurations the calibrated gate with a single $\\alpha$ "
    "reaches $1.12°$ on average, within $1.59\\times$ of a per-configuration bound that is allowed to "
    "inspect the test data, while accounting for $0.17\\%$ of the runtime. The comparison we lead with is "
    "against that bound, not against the conventional default: a conventionally set fixed threshold "
    "costs a mean factor of $12.8$ (up to $31$) against the same bound, but it does so because the "
    "default level is one at which the gate is effectively inactive \u2014 $98\\%$ of noise-only points "
    "survive it \u2014 and we treat that number as supporting evidence rather than as the headline.",
)

# ============================ C. §2 参考尺度的分类与区别（M1）===============
add(
    "M1 §2：把 percentile 与噪声分量参考的区别集中陈述 + 指向分类表",
    "Thresholds are conventionally written relative to the maximum [5,7] or to a percentile [6,8,9] of "
    "the observed energy, and the relative level is fixed empirically. The noise-floor concept itself is "
    "standard in speech enhancement and adaptive spatial filtering, but to our knowledge the consequence "
    "of the *sparsity-dependence of the reference scale* has not been stated for the UBSS gate.",
    "Thresholds are conventionally written relative to the maximum [5,7] or to a percentile [6,8,9] of "
    "the observed energy, and the relative level is fixed empirically. The distinction that matters here "
    "is easy to lose sight of because both families are called *quantile* thresholds. A percentile of the "
    "mixture is a quantile of the quantity being thresholded, so the level it implies on the noise scale "
    "is a property of the mixture's shape and moves with it; a multiple of the noise floor is a quantile "
    "of the *noise component alone*, so the level it implies is fixed by the detector and the noise. The "
    "two agree only while the noise-only component dominates the mixture's lower half \u2014 which is the "
    "condition that fails as sparsity rises (Proposition 7). Table 27 sets the classes of reference scale "
    "side by side with what each is a quantile of and whether that quantity moves with the activation "
    "probability. The noise-floor concept itself is standard in speech enhancement and adaptive spatial "
    "filtering, and percentile-of-mixture thresholds are standard in this literature; what we have not "
    "found stated is the consequence of the *sparsity-dependence of the reference scale* for the UBSS "
    "gate, which Propositions 6 and 7 make quantitative.",
)

# ============================ D. §5.1 估计量的有限样本水平偏差 ================
add(
    "Q1/§5.1：补次序统计量水平约定的有限样本偏差（实测）",
    "(iii) The estimate is reported together with a dispersion diagnostic",
    "(iii) The level convention $q_i=i/(n+1)$ carries a small negative finite-sample bias, which we "
    "measured on pure Gaussian noise rather than assumed: at $n=FT=2112$ the estimator returns "
    "$\\hat\\sigma^2/\\sigma^2=0.977$, at $8448$ it returns $0.984$ and at $33792$ it returns $0.994$, "
    "i.e. a bias falling roughly as $n^{-1/2}$ and consistent with the difference between the median "
    "position of an order statistic and the nominal level $i/(n+1)$. A $2\\%$ shift in $\\hat\\nu$ is far "
    "below the insensitivity documented in Appendix A.4, so we keep the convention and report the "
    "measurement; it is part of the error budget of Table 2 rather than a free parameter.\n\n"
    "(iv) The estimate is reported together with a dispersion diagnostic",
)

# ============================ E. §5.3 守卫错误分布 + spread 条件的作用（M7/Q2）==
add(
    "M7+Q2 §5.3：错误分布与 spread 子条件的作用",
    "The criterion is a guard against gross silent failure, not a precise classifier, and we report its "
    "margin as such. Across the 27 conditions of Table 4 the largest retention at which the gate should "
    "have been inactive is $1.0\\%$ and the smallest at which it should have been active is $3.4\\%$, so "
    "the boundary at $3.5\\%$ lies in an empty interval between the two groups. Fig. 2 plots both groups "
    "against the boundary, and Section 10.3 reports how the decision responds to the boundary value "
    "itself.",
    "The criterion is a guard against gross silent failure, not a precise classifier, and we report its "
    "margin as such. Across the 27 conditions of Table 4 the largest retention at which the gate should "
    "have been inactive is $1.0\\%$ and the smallest at which it should have been active is $3.4\\%$, so "
    "the boundary at $3.5\\%$ lies in an empty interval between the two groups. Fig. 2 plots both groups "
    "against the boundary, and Appendix A.3 reports how the decision responds to the boundary value "
    "itself.\n\n"
    "**Where the residual errors lie, and which sub-condition catches what.** Counting decisions per seed "
    "over the same 27 conditions ($270$ decisions) gives $17$ material errors, $14$ of them applying the "
    "gate where leaving it off would have been better and $3$ the reverse. They are *not* concentrated at "
    "the corner of the sparsity\u2013SNR plane that motivates the guard: only $4$ of the $17$ sit at "
    "$\\mathrm{SNR}\\le5$ dB, and the sparse ladder contributes $4$ of its $150$ decisions ($3\\%$). They "
    "concentrate instead where the point model itself is weakest \u2014 $8$ of $40$ decisions ($20\\%$) "
    "on sources with genuine TF structure (chirps, AM\u2013FM tones, transients) and $3$ of $20$ ($15\\%$ "
    "on Laplace and block-sparse sources) \u2014 which is the same place the closed form is weakest and "
    "is worth stating plainly (Table 28).\n\n"
    "The check has three clauses and one of them is doing all the work in a way we did not expect. The "
    "retention clause is the binding one in all $68$ synthetic decisions that turn the gate off; the "
    "$n_{\\text{base}}\\ge n_{\\min}$ clause never binds alone, and the dispersion clause "
    "$\\mathrm{spread}\\le0.5$ changes *no* decision in those $270$. On real speech the picture reverses: "
    "$13$ of the $15$ configurations never exceed a dispersion of $0.401$ across eight seeds, so the "
    "clause is silent there, and the five decisions in which it is the *only* failing clause are all in "
    "the two configurations whose noise is not a stationary white floor \u2014 babble ($4$ of $8$ seeds, "
    "dispersion up to $0.733$) and the noise-free recordings ($1$ of $8$). We therefore keep the clause, "
    "not as a redundant guard but as the one that fires precisely where the premise fails; Appendix A.3 "
    "gives the ablation.",
)

# ============================ F. §6 Prop 4 / Prop 5（Minor 8、M4）============
add(
    "Minor 8：Prop 4 在命题陈述处标注下界",
    "**Proposition 4 (contamination ratio).** With independent per-source activation at probability $p$, "
    "the expected number of admitted noise-only points per admitted single-source point is",
    "**Proposition 4 (contamination ratio).** With independent per-source activation at probability $p$, "
    "the expected number of admitted noise-only points per admitted single-source point is bounded below "
    "by",
)

add(
    "M4：Prop 5 降级为标度论证并改名",
    "**Proposition 5 (perturbation bound).** Let $\\mathbf{G}=\\sum_l w_l\\mathbf{u}_l\\mathbf{u}_l^\\top"
    "=\\mathbf{G}_s+\\mathbf{G}_0$",
    "*Scaling argument.* We keep the perturbation calculation of the classical analysis, but as a "
    "scaling argument rather than a bound: with an undetermined constant it cannot be evaluated for any "
    "particular configuration, and the experiments below show where it stops describing the data. Let "
    "$\\mathbf{G}=\\sum_l w_l\\mathbf{u}_l\\mathbf{u}_l^\\top=\\mathbf{G}_s+\\mathbf{G}_0$",
)

# ============================ G. §7 局部 SNR 与记录清单（Minor 3、Major 8）====
add(
    "Minor 3：Table 6 表注加局部 SNR",
    "**Table 6.** Mixing-matrix angle error (degrees, lower is better) and SDR (dB); 10 seeds, "
    "SNR $=20$ dB, $N=4$, $M=2$. *SCA-tuned* is an oracle that selects the best threshold per regime "
    "using the test data.",
    "**Table 6.** Mixing-matrix angle error (degrees, lower is better) and SDR (dB); 10 seeds, "
    "SNR $=20$ dB in the plane-average convention of Section 3.3, $N=4$, $M=2$. *SCA-tuned* is an oracle "
    "that selects the best threshold per regime using the test data. Because the convention averages "
    "signal power over the whole TF plane, the local SNR at an active point is higher by "
    "$-10\\log_{10}P(\\text{act})$: the nominal $20$ dB corresponds to $31.1$, $27.3$, $24.6$, $22.3$ and "
    "$20.6$ dB at $p=0.02$, $0.05$, $0.10$, $0.20$ and $0.40$. Comparisons with work that defines SNR "
    "locally should be made against those values, not against $20$ dB.",
)

add(
    "Minor 3：Table 8 表注加局部 SNR",
    "**Table 8.** Angle error (degrees) versus SNR; ten seeds per cell.",
    "**Table 8.** Angle error (degrees) versus SNR; ten seeds per cell. The nominal SNR follows the "
    "plane-average convention of Section 3.3, so the local SNR at an active point is higher by "
    "$7.3$ dB at $p=0.05$ and $2.3$ dB at $p=0.20$ \u2014 e.g. a nominal $0$ dB here is $7.3$ dB locally "
    "in the $p=0.05$ column.",
)

add(
    "Major 8：§7 记录清单补入第四轮归档",
    "with summaries in `r2_runtime.json` and `r2_stats.json` and the "
    "signal-energy check in `r2_prop6_rank.json`.",
    "with summaries in `r2_runtime.json` and `r2_stats.json` and the "
    "signal-energy check in `r2_prop6_rank.json`. The revision adds `r4_density_frontend.json` ($60$ "
    "records) for a third mask-consuming front end, `r4_guard_errors.json` ($270$ decisions) for the "
    "per-seed accounting and the dispersion ablation, `r4_robust_floor.json` ($92$) for the robust-scale "
    "comparison, `r4_table2.json` for Table 2, and `r4_floor_bias.json` for the pure-noise bias of the "
    "level convention.",
)

# ============================ H. §8.8 两项模型正式化（M4）===================
add(
    "M4 §8.8：把两项模型提为正式（事后）描述",
    "A two-term model accounts for the pattern. Write $\\mathrm{err}^2 = a/FT + b$: the first term is the "
    "noise-limited contribution the perturbation argument describes, the second an $FT$-independent "
    "floor. With three points and two parameters the fit is not a test of the functional form, so its "
    "value lies in the coefficients rather than in any goodness-of-fit statistic.",
    "Since the perturbation argument has no determined constant, the useful description of this "
    "experiment is the empirical one, and we prefer it to the proposition: **$\\mathrm{err}^2 = a/FT + b$**, "
    "fitted per configuration, where the first term is the noise-limited contribution the scaling "
    "argument describes and the second an $FT$-independent floor. With three points and two parameters "
    "the fit is not a test of the functional form, so we use it as a two-parameter summary of each cell "
    "rather than as a model that has been validated; its value is that $b$ is directly interpretable, "
    "being the squared error that enlarging the observation cannot remove.",
)

# ============================ I. §9 两个机制与单系数稳定性（M6、Q3）==========
add(
    "M6 §9 开头：两个机制不同 + Q3 单系数稳定性",
    "The synthetic ladder of Section 8 activates each source independently at each time\u2013frequency "
    "point. That is precisely the assumption on which the two-stage paradigm rests, and therefore the "
    "assumption it is least able to test. We now repeat the comparison on real recordings.",
    "The synthetic ladder of Section 8 activates each source independently at each time\u2013frequency "
    "point. That is precisely the assumption on which the two-stage paradigm rests, and therefore the "
    "assumption it is least able to test. We now repeat the comparison on real recordings.\n\n"
    "One thing should be said before the numbers, because it changes how the two halves of the paper "
    "relate. **The mechanism that dominates here is not the one the diagnosis of Section 4 is about.** "
    "On these recordings the noise-only fraction sits at $\\pi_0\\approx0.43$\u2013$0.51$, on the "
    "favourable side of the knee, and the measured reference-scale ratio is only $1.9$\u2013$2.2$; the "
    "conventional gate therefore does not fail here because its reference scale has drifted. It fails "
    "because the median of the observed energy is too small a number to sit where the gate should sit: "
    "the optimal multiple runs from $5$ to $1000$ (Table 14) while the conventional default is $0.02$. "
    "Both failures are silent and both are removed by the same derived rule, but they are different "
    "failures, and the synthetic ladder is what isolates the first one.",
)

add(
    "Q3：单系数稳定性用逐档最优系数的实测范围回答",
    "and on the first it is beaten, narrowly, by rules that require a coefficient to be chosen \u2014 a "
    "coefficient whose own optimum moves across two to three orders of magnitude between configurations "
    "(Table 14), which is what Section 4 shows not to be generic.",
    "and on the first it is beaten, narrowly, by rules that require a coefficient to be chosen. We "
    "checked whether that coefficient is in fact stable on this corpus, since if it were, a "
    "single-coefficient rule would be the right answer and our argument would be weaker. It is not "
    "stable: the per-configuration optimum of the max-referenced rule runs over $0.001$\u2013$0.1$, a "
    "factor of $100$, and that of the median-referenced rule over $5$\u2013$1000$, a factor of $200$ "
    "(Table 14 and Appendix A.9). What the max-referenced rule has instead is *flatness*: holding "
    "$c=0.05$ costs $0.92°$ on average against $0.49°$ for the per-configuration optimum, a penalty of "
    "$0.43°$, whereas holding the median-referenced default at its conventional $0.02$ costs an order of "
    "magnitude. The two rules therefore differ in kind and not only in degree: the maximum of the energy "
    "is a proxy for the *signal* scale, which is stable across these configurations by construction, "
    "while the median of the energy is a reference on the *mixture* scale, which is what Section 4 shows "
    "to drift. This is the sense in which we read the negative result as consistent with the diagnosis "
    "rather than opposed to it, and it is why we present the derived threshold as a transferable "
    "calibration rather than as the most accurate gate.",
)

# ============================ J. Minor 1 / Minor 2 ========================
add(
    "Minor 1：p 值记号改为 p_{\\rm v}",
    "Against the conventional fixed threshold the reduction is significant at $p=1.6\\times10^{-4}$, "
    "$1.5\\times10^{-3}$ and $5.6\\times10^{-3}$ for $p=0.02$, $0.05$ and $0.10$, and at "
    "$6.7\\times10^{-3}$ at $p=0.40$; it is **not** significant at $p=0.20$ ($p=0.088$), where the "
    "conventional threshold has already become usable, nor at the dense extreme ($p=0.345$), where the "
    "gate is disabled.",
    "Throughout this paragraph a subscripted symbol denotes the significance level and a bare symbol the "
    "activation probability. Against the conventional fixed threshold the reduction is significant at "
    "$p_{\\rm v}=1.6\\times10^{-4}$, $1.5\\times10^{-3}$ and $5.6\\times10^{-3}$ for $p=0.02$, $0.05$ "
    "and $0.10$, and at $6.7\\times10^{-3}$ at $p=0.40$; it is **not** significant at $p=0.20$ "
    "($p_{\\rm v}=0.088$), where the conventional threshold has already become usable, nor at the dense "
    "extreme ($p_{\\rm v}=0.345$), where the gate is disabled.",
)

# ============================ K. Minor 4：Table 2 表注 ====================
add(
    "Minor 4：Table 2 表注说明与 Table 22 的口径差异",
    "**Table 2.** Properties of the blind noise-floor estimator ($N=4$, $M=2$, SNR $=20$ dB, five seeds).",
    "**Table 2.** Properties of the blind noise-floor estimator ($N=4$, $M=2$, SNR $=20$ dB in the "
    "plane-average convention, five seeds, $q_{\\max}=0.02$). Row `dense` is the noise-free extreme of "
    "the ladder. The corresponding row of Table 22 is computed over eight seeds and is level-resolved, "
    "so the two differ by sampling and not by method: this row is $77.1$ over five seeds and $84.2$ over "
    "eight, and its per-seed values span $55.6$\u2013$99.6$.",
)

# ============================ L. §12 结论（M6）===================
# 该项在第三轮修订中已完成（"accounts for a substantial part of the observed brittleness of
# the two-stage pipelines tested here" 已在稿中），脚本不再重复，改为显式跳过。
_ALREADY_DONE = {
    "M6 结论：把\"不是聚类的问题\"收敛为\"在所测流水线上\"",   # 第三轮已改
    # 下面两条的 new 文本被**后续的结构改动**再次改写（Section 10.x -> Appendix A.x、
    # 表号重排），因此重跑时既找不到 new 也找不到 old。文件已处于期望状态，显式跳过。
    "M1 §2：把 percentile 与噪声分量参考的区别集中陈述 + 指向分类表",
    "M7+Q2 §5.3：错误分布与 spread 子条件的作用",
}

# ============================ M. Minor 2：局限编号 ========================
SEVENTH = (
    "*Seventh*, the real-speech evaluation rests on one corpus. All $960$ real-speech runs draw on "
    "LibriSpeech dev-clean, and although the fifteen configurations vary the window length, the source "
    "count, the SNR, the denseness of two sources and the noise type, they cannot speak to corpora with "
    "different recording conditions, languages, or array geometries. The claim the real data supports is "
    "narrow and we state it as such: on this corpus the conventional reference scale fails by an order of "
    "magnitude and the derived one does not. A second corpus is the natural next step and we did not have "
    "access to one for this study.\n\n"
)


# 第六轮（R6）重写了下列条目的目标文本（Prop 8 命名、Table 1 列名、§1 与 §9 措辞、
# Table 37 整体重测、§7 归档清单），这些条目本身已无需再执行；保留登记以免重跑该轮时
# 误判为需要改动。已核对：在 R6 之前的手稿上它们仍为 0 未命中（见 MEMORY）。
_SUPERSEDED_R6 = {
    'M2 引言：适用范围前置',
    'M6 §9 开头：两个机制不同 + Q3 单系数稳定性',
}

# R8（CSSP 投稿前意见）：全稿 "per-configuration bound" 统一改名 best-observed reference，
# 贡献 5 与 §10 的两条目标串因此被重写（后者的自证式框句同时被精简）。登记而非删除。
_SUPERSEDED_R8 = {
    'M3 贡献 5 改为以逼近 oracle 为首要主张',
    'Q3：单系数稳定性用逐档最优系数的实测范围回答',
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    s = MS.read_text()
    changed = skipped = missed = 0

    for tag, old, new in E:
        if tag in _ALREADY_DONE or tag in _SUPERSEDED_R6 or tag in _SUPERSEDED_R8:
            print(f"  [已修订，跳过] {tag}")
            skipped += 1
            continue
        if new in s:
            print(f"  [跳过] {tag}")
            skipped += 1
            continue
        if old not in s:
            print(f"  [未命中] {tag}")
            missed += 1
            continue
        s = s.replace(old, new, 1)
        print(f"  [已改] {tag}")
        changed += 1

    # 摘要单独处理
    if NEW_ABSTRACT in s:
        print("  [跳过] 摘要重写")
    else:
        lines = s.split("\n")
        i = lines.index("## Abstract")
        assert lines[i + 2].startswith("Two-stage"), "摘要段位置不符"
        lines[i + 2] = NEW_ABSTRACT
        s = "\n".join(lines)
        print(f"  [已改] 摘要重写（{len(NEW_ABSTRACT.split())} 词）")
        changed += 1

    # Minor 2：把 *Seventh* 段移到局限列表末尾（当前错序为 First..Fourth, Seventh, Fifth, Sixth）
    print(f"\n=== Minor 2：局限编号（当前顺序见下）===")
    lines = s.split("\n")
    import re as _re
    _order = _re.findall(r"^\*(First|Second|Third|Fourth|Fifth|Sixth|Seventh)\*", s, _re.M)
    _want = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
    idx = [i for i, l in enumerate(lines) if l.startswith("*Seventh*, the real-speech")]
    if _order == _want:
        idx = []                       # 顺序已正确 -> 不再重排（保证幂等）
        print("  [跳过] 编号已是正确顺序")
    elif idx:
        i = idx[0]
        # 找到该段结束（空行）
        j = i
        while j < len(lines) and lines[j].strip():
            j += 1
        block = lines[i:j + 1]
        rest = lines[:i] + lines[j + 1:]
        # 插到 *Sixth* 段之后
        k = next((q for q, l in enumerate(rest) if l.startswith("*Sixth*, on real speech")), None)
        if k is not None:
            m = k
            while m < len(rest) and rest[m].strip():
                m += 1
            newlines = rest[:m + 1] + block + rest[m + 1:]
            s = "\n".join(newlines)
            print("  [已改] *Seventh* 段移到 *Sixth* 之后")
            changed += 1
        else:
            print("  [未命中] 找不到 *Sixth* 段")
            missed += 1
    else:
        print("  [跳过] 编号已是正确顺序")
        skipped += 1

    # 校验顺序
    import re
    order = re.findall(r"^\*(First|Second|Third|Fourth|Fifth|Sixth|Seventh)\*", s, re.M)
    want = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
    print(f"  编号顺序: {order}  {'正确' if order == want else '**仍不正确**'}")
    if order != want:
        sys.exit("编号顺序未修正，中止")

    if not a.dry:
        MS.write_text(s)
    print(f"\n已改 {changed} | 跳过 {skipped} | 未命中 {missed}   （{'演练' if a.dry else '已写入'}）")
    if missed:
        sys.exit("有未命中项，请先修正脚本再写入")


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
