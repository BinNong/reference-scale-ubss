#!/usr/bin/env python3
"""第六轮修订：由「顶级期刊审稿人」口径意见的逐条核查驱动。

核查方式与结论见 `.workbuddy/memory/2026-09-19.md`。分四档，本脚本只落实前两档
（第三档手稿已答过，回信指回原文；第四档建议不接）。另有两条是**改稿自查时发现、
审稿人并未指出**的问题（第 6、7 条）。

  1. Prop 8 的 "exact quantile relation" 名不副实。`F^{-1}(q) > F_0^{-1}(q)` 对任意
     q>0 严格成立（信号能量同样支撑在 (0,∞)），原限定句"the q-quantile of the mixture
     still lies inside the noise component"因此永远不成立。改名 + 写出显式亏项。
  2. §2 的 reference-scale 分类只有两类（max / 百分位），漏掉了自适应阈值那一类。
     补引 Hassan & Ramli 2023（ATFT）与 Chen et al. 2024（双传感器 max 经验阈值），
     并把"为什么三类里只有第三类不随稀疏度漂移"讲清。
  3. 次序统计量的口径三处不一致（§5.1 / Algorithm 1 / A.17 与 Table 37）。实测
     nfr.estimate_noise_power 用 30 个**名义**几何水平、floor(0.02n) 上界、整数截断后
     去重 → n=2112 实际只有 **20** 个（不是审稿人算的 42），n=4.8e4 有 27 个。
  4. Prop 7 正文一句越界（"necessarily on the signal scale"），证明只给出"受信号分量影响"。
  5. 保留判据的两类错判率补精确二项区间（数字原已在 Table 21）。
  6. **[自查]** §A.17 描述的"每个分位数用 np.partition（O(n)）而非全排序"与实现不符：
     nfr.py 第 164 行是 np.sort。因此 Table 37 的"selection"与"full sort"两列实际都在
     排序（归档 ratio 随 n 收敛到 1.00），且原计时无预热。重测三条真实路线并重写该节。
  7. **[自查]** "closed form" 降级为 "semi-analytic"（§4.2/Prop 6/贡献1/Tables 1,37 等）。
     另修 §2 一处重复句、§9 的 "real recordings" 措辞（实为真实语音源 + 合成瞬时混合）。

用法：  python3 apply_r6_fixes.py [--dry]
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


# ================================================== 1. Prop 8：exact → 下尾近似
add("Prop 8 命题名与限定句 + 亏项段 + 次序统计量条数（§5.1）",
"""where $Q_q$ is the quantile function. The relation is exact for every $q$ such that the $q$-quantile of the mixture still lies inside the noise component.

We use a geometric sequence of small levels $q_i$, each attached to the expected level of the corresponding order statistic, $q_i=i/(n+1)$ rather than $i/n$ — using the nominal level with the raw index treats the sample minimum as an arbitrarily small quantile and biases the estimate severely in the lower tail. The per-level estimates are combined by their median, so that isolated deviations do not propagate.""",
"""where $Q_q$ is the quantile function. The identity is exact for the *noise-only* law; applied to a quantile of the mixture it is an approximation, and the next paragraph states how far the two differ.

**What the relation is, and what it is not.** The estimator reads a quantile of the *mixture*, and the active components put mass in the same lower tail, so the inversion is an approximation rather than an identity. Write $q=\\pi_0F_0(x_q)+(1-\\pi_0)F_s(x_q)$, with $F_0$ the noise-only CDF and $F_s$ the signal-component mixture CDF of Proposition 6, both evaluated at the level read. Then

$$
F_0(x_q)\\;=\\;\\frac{q}{\\pi_0}-\\frac{1-\\pi_0}{\\pi_0}F_s(x_q)\\;<\\;q,
$$

so the level read lies *below* the true noise-only $q$-quantile and $\\hat\\sigma^2$ is low by $F_0(x_q)/q$ in the mass variable. The elementary bound obtained by discarding $F_s$ from the numerator, $F_0(x_q)\\ge\\big(q-(1-\\pi_0)\\big)/\\pi_0$, is vacuous throughout the ladder, where $1-\\pi_0>q$; what controls the approximation there is the small-argument expansion of Section 4.2, under which $F_s(x_q)$ is exactly the quantity evaluated. What the method rests on is therefore not exactness but the stability of the *consequence*: the deficit is Table 2, and Appendix A.4 shows the resulting angle error to move by at most $1.3\\%$ relative across a decade of $q_{\\max}$, so the gate is placed stably even where the estimate is biased.

We use $n_q=30$ nominally geometric levels $q_i$ over the index range $1\\le i\\le\\lfloor q_{\\max}n\\rfloor$, each attached to the expected level of the corresponding order statistic, $q_i=i/(n+1)$ rather than $i/n$ — using the nominal level with the raw index treats the sample minimum as an arbitrarily small quantile and biases the estimate severely in the lower tail. Integer truncation collapses the lowest of the nominal levels, so the number of order statistics actually read is $20$ at the $n=2112$ of the synthetic ladder and $27$ at the $n\\approx4.8\\times10^4$ of Section 9, rising to $28$ at $n=10^6$ (Table 37). The per-level estimates are combined by their median, so that isolated deviations do not propagate.""")

add("Prop 8 命题标题",
"**Proposition 8 (exact quantile relation).** Under A2",
"**Proposition 8 (noise-dominated lower-tail relation).** Under A2")

add("摘要：exact quantile 无需改（原文已只说 chi-square quantile relation）；此处只改 Algorithm 1 注释",
"5  s2_i         <- 2*e_(i) / Q_{q_i}(chi2_{2M})          exact quantile inversion",
"5  s2_i         <- 2*e_(i) / Q_{q_i}(chi2_{2M})          lower-tail inversion")

add("Fig. 2 题注",
"Per-point energies feed a blind noise-floor estimator built on the exact chi-square quantile relation",
"Per-point energies feed a blind noise-floor estimator built on the chi-square lower-tail relation")

add("§11 结论：exact chi-square quantile relation",
"estimating that floor blindly from the lower tail with the exact chi-square quantile relation",
"estimating that floor blindly from the noise-dominated lower tail with the chi-square quantile relation")

add("贡献2：exact quantile relation 的措辞",
"estimate the floor blindly from the lower tail of the energy distribution using the exact quantile relation $s^2 = 2e_{(q)}/Q_q(\\chi^2_{2M})$",
"estimate the floor blindly from the noise-dominated lower tail of the energy distribution using the chi-square quantile relation $s^2 = 2e_{(q)}/Q_q(\\chi^2_{2M})$, which is exact at a noise-only point and is applied as a lower-tail approximation")

# ================================================== 2. §2：补第三类 + 补引
add("§1 引言：惯例遗漏自适应变体",
"In practice the threshold is written as a fixed multiple of the median or of the maximum of the observed energies, and the multiple is chosen once, empirically, on a validation set.",
"In practice the threshold is written relative to the mixture's own statistics — a fixed multiple of the median or of the maximum of the observed energies, with the multiple chosen once, empirically, on a validation set, or, in the adaptive variants, from a statistic of the mixture computed at run time [27].")

add("§2 两类 → 三类",
"Thresholds are conventionally written relative to the maximum [5,7] or to a percentile [6,8,9] of the observed energy, and the relative level is fixed empirically. The distinction that matters here is easy to lose sight of because both families are called *quantile* thresholds.",
"Thresholds are conventionally written relative to a functional of the *mixture*, in two families. The first fixes a relative level on a statistic of the mixture — its maximum [5,7,28] or one of its percentiles [6,8,9] — and sets that level empirically. The second makes the reference *data-adaptive*: the adaptive time–frequency thresholding of [27] normalises each TF vector to unit norm, forms the mean of the resulting norms, and sets the threshold from that mean, so that the reference follows the mixture rather than being fixed in advance. Neither family estimates a noise floor, and none of these works calibrates the level to a target probability. The distinction from what we propose is easy to lose sight of because all of them are called *quantile* thresholds.")

add("§2 重复句",
"which Propositions 6 and 7 make quantitative. Propositions 6 and 7 make it quantitative: the effective noise-floor multiple",
"which Propositions 6 and 7 make quantitative: the effective noise-floor multiple")

add("参考文献补 [27] [28]",
"[26] V. Panayotov, G. Chen, D. Povey, S. Khudanpur, \"LibriSpeech: an ASR corpus based on public domain audio books,\" *Proc. IEEE ICASSP*, 2015, pp. 5206–5210.",
"""[26] V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on public domain audio books," *Proc. IEEE ICASSP*, 2015, pp. 5206–5210.

[27] N. Hassan, D. A. Ramli, "Sparse component analysis (SCA) based on adaptive time–frequency thresholding for underdetermined blind source separation (UBSS)," *Sensors*, vol. 23, no. 4, 2060, 2023.

[28] J. Chen, H. Zhang, S. Sun, "Exploiting time–frequency sparsity for dual-sensor blind source separation," *Electronics*, vol. 13, no. 7, 1227, 2024.""")

# ================================================== 3. Algorithm 1 的索引口径
add("Algorithm 1 第 3 行：ceil → floor + 名义 30 + 去重",
"3  i            <- geometric indices 1 ... ceil(q_max*n),  q_max = 0.02",
"3  i            <- 30 geometric indices in 1 ... floor(q_max*n), deduplicated   q_max = 0.02")

# ================================================== 4. Prop 7 越界句
add("Prop 7 正文越界句",
"Below it the median is on the noise scale ($r=\\mathcal{O}(1)$); above it the median is necessarily on the signal scale, where $r$ grows with the local SNR.",
"Below it the median is on the noise scale ($r=\\mathcal{O}(1)$); above it is no longer a quantile of the noise component alone and is pulled towards the signal scale, where $r$ grows with the local SNR.")

# ================================================== 5. 保留判据：二项区间
add("§5.3 保留判据的两类错判率 + 精确二项区间",
"which is the same place the closed form is weakest and is worth stating plainly (Table 27).",
"""which is the same place the semi-analytic form is weakest and is worth stating plainly (Table 27). Reported as rates rather than counts, the same $270$ per-seed decisions give $14/270=5.2\\%$ false activations (exact $95\\%$ confidence interval $2.86$–$8.55\\%$) and $3/270=1.1\\%$ false fallbacks ($0.23$–$3.21\\%$). The two directions are not symmetric, and the interval on the fallback rate is wide enough that the balance should be read as a statement of which error the guard prefers rather than as a calibrated false-positive level.""")

add("§A.3 保留判据：补一句性质与区间",
"It should not be read as erring toward\ncaution: at $0.035$ the errors are $14$ to $3$ in favour of acting.",
"It should not be read as erring toward caution: at $0.035$ the errors are $14$ to $3$ in favour of acting, or $5.2\\%$ against $1.1\\%$ of the $270$ decisions with exact $95\\%$ intervals of $2.86$–$8.55\\%$ and $0.23$–$3.21\\%$. The boundary is a globally fixed constant validated on this grid, not a separated held-out set, and a user who requires a calibrated error rate rather than a coarse guard should read the interval rather than the point estimate.")

# ================================================== 6. closed form → semi-analytic
add("摘要：closed form → semi-analytic",
"A mixture-law analysis gives that ratio in closed form, locates its boundary exactly",
"A mixture-law analysis gives that ratio in semi-analytic form, locates its boundary exactly")

add("贡献1 标题与正文",
"**1. A diagnosis with a closed form.** We show that a threshold referenced to the median of the observed energy is equivalent to a threshold referenced to the noise floor with a multiple equal to $r(p)=\\mathrm{median}(e)/\\nu$, and we give a closed form for $r(p)$ (Section 4.2).",
"**1. A diagnosis with a semi-analytic form.** We show that a threshold referenced to the median of the observed energy is equivalent to a threshold referenced to the noise floor with a multiple equal to $r(p)=\\mathrm{median}(e)/\\nu$, and we give a semi-analytic form for $r(p)$ (Section 4.2).")

add("§1 四步概述",
"a *synthetic validation* in which the closed form is checked against the law it assumes",
"a *synthetic validation* in which the semi-analytic form is checked against the law it assumes")

add("§4.2 小节标题",
"### 4.2 Closed form for the reference-scale ratio",
"### 4.2 A semi-analytic form for the reference-scale ratio")

add("Prop 6 标题",
"**Proposition 6 (closed form for $r(p)$).** At a TF point",
"**Proposition 6 (semi-analytic form for $r(p)$).** At a TF point")

add("§4.2 说明：中位数靠二分，故非严格闭式",
"whose partial moments $m_k$ follow from the chi-square CDF. The mixture median is then found by bisection.",
"whose partial moments $m_k$ follow from the chi-square CDF. The mixture median is then found by bisection, so the result is a *semi-analytic* characterization — an expansion that is closed-form at each order, evaluated numerically for the median — rather than a closed-form expression in the elementary sense. We use the term in that sense throughout and say so wherever the status of the claim matters.")

add("§4.2 数值验证段",
"**Numerical verification.** Table 1 compares the closed form with simulation (Fig. 1). The agreement is within $1.5\\%$ throughout, including $+0.2\\%$ at $p=0.20$, which lies *above* the knee. The closed form is undefined at $p\\ge0.40$",
"**Numerical verification.** Table 1 compares the semi-analytic form with simulation (Fig. 1). The agreement is within $1.5\\%$ throughout, including $+0.2\\%$ at $p=0.20$, which lies *above* the knee. The form is undefined at $p\\ge0.40$")

add("§4.2 数值验证段第二句",
"The boundary of the closed form is therefore set by the position of the median",
"The boundary of the form is therefore set by the position of the median")

add("Table 1 题注",
"five seeds; measured against closed form). The closed form uses the single-source energy convention",
"five seeds; measured against the semi-analytic form). The latter uses the single-source energy convention")

add("Table 1 表头",
"| $p$ | $\\pi_0$ | $r$ measured | $r$ closed form | rel. error | noise mass $\\ge1/2$ |",
"| $p$ | $\\pi_0$ | $r$ measured | $r$ semi-analytic | rel. error | noise mass $\\ge1/2$ |")

add("§10 讨论：computable in closed form",
"its misalignment is computable in closed form",
"its misalignment is computable in semi-analytic form")

add("§A.6 首句",
"and Section 4.2 shows the\nclosed form to be insensitive to that angle",
"and Section 4.2 shows the\nsemi-analytic form to be insensitive to that angle")

add("§11 结论：closed-form reference-scale ratio",
"The closed-form reference-scale ratio locates the drift exactly",
"The semi-analytic reference-scale ratio locates the drift exactly")

add("Table 33 题注",
"against the closed form of Section 4.2 and the blind floor estimate of Section 5.1. The closed form is not defined beyond the knee",
"against the semi-analytic form of Section 4.2 and the blind floor estimate of Section 5.1. The latter is not defined beyond the knee")

add("Table 33 表头",
"| Regime | real $\\mathbf{A}$ | complex, fixed | complex, per frequency | closed form | $\\hat\\sigma^2/\\sigma^2$ |",
"| Regime | real $\\mathbf{A}$ | complex, fixed | complex, per frequency | semi-analytic | $\\hat\\sigma^2/\\sigma^2$ |")

add("Table 33 后正文",
"The three columns agree with each other to within $2\\%$ at every regime, and with the closed form to\nwithin the Monte-Carlo error wherever the closed form applies",
"The three columns agree with each other to within $2\\%$ at every regime, and with the semi-analytic form to\nwithin the Monte-Carlo error wherever the latter applies")

# ================================================== 7. §9 措辞
add("§9 标题",
"## 9. Evaluation on Real Speech",
"## 9. Evaluation on Real-Speech Sources")

add("§9 开篇",
"We now repeat the comparison on real recordings.",
"We now repeat the comparison on real speech sources, mixed synthetically under the same instantaneous model; the section is an evaluation of source TF structure under the paper's mixing assumptions, not of a real multichannel recording.")

# ================================================== 8. §A.17 与 Table 37 重写（自查）
add("§A.17 首段：交货实现是排序而非选择",
"""Section 5.1 takes thirty order statistics of the energy vector. A single selection is $O(n)$ in
expectation against $O(n\\log n)$ for a full sort, and with $O(n)$ points per TF frame this is the
only super-constant step in the gate; it is worth saying how it behaves at the scales the paper does
not reach.""",
"""Section 5.1 takes $n_q=30$ nominally geometric levels, which integer truncation reduces to twenty
distinct order statistics at the $n=2112$ of the synthetic ladder and twenty-seven at the
$n\\approx4.8\\times10^4$ of Section 9. The shipped estimator sorts the energy vector and reads those
order statistics off the sorted array, so its realized cost is $O(n\\log n)$ and not the $O(n)$
selection we described in the previous version of this section; we correct the description rather
than the code, because the results were produced by the sort. It is worth saying how the step behaves
at the scales the paper does not reach, and how the two routes compare.""")

add("Table 37 题注与表体重写",
"""**Table 37.** Cost of the blind floor estimate against the length $n=FT$ of the energy vector (single core, median of twenty repetitions; the same estimator as Table 25, which measured $2.99$ ms at $n=4.8\\times10^4$). "histogram" is the $4096$-bin approximation of the same thirty quantiles, and the last two columns are the accuracy of the two routes on pure Gaussian noise, where both should return $1$.

| $n$ | selection (this paper) | full sort | histogram | ns per point | $\\hat\\sigma^2/\\sigma^2$ (exact) | $\\hat\\sigma^2/\\sigma^2$ (histogram) |
|---|---|---|---|---|---|---|
| $2{,}112$ | 0.30 ms | 0.07 ms | 0.13 ms | 142 | 1.022 | 1.030 |
| $8{,}192$ | 0.61 ms | 0.37 ms | 0.25 ms | 75 | 1.059 | 0.998 |
| $32{,}768$ | 1.98 ms | 1.71 ms | 0.71 ms | 60 | 0.957 | 0.871 |
| $131{,}072$ | 8.19 ms | 7.86 ms | 2.44 ms | 63 | 1.010 | 1.113 |
| $524{,}288$ | 35.9 ms | 35.5 ms | 9.74 ms | 68 | 0.998 | 0.911 |
| $1{,}048{,}576$ | 77.9 ms | 77.9 ms | 19.4 ms | 74 | 1.003 | 1.086 |""",
"""**Table 37.** Cost of the blind floor estimate against the length $n=FT$ of the energy vector. Protocol: median of forty repetitions after five warm-up calls, single thread, one realization per size for the timings; the two error columns are the mean *absolute* relative error of $\\hat\\sigma^2/\\sigma^2$ over twenty seeds on pure Gaussian noise, where both routes should return $1$. "levels" is the number of order statistics actually read, thirty nominal geometric levels being reduced by integer truncation (Section 5.1). "sort + gather" is the shipped route, `np.sort` followed by the index read; "partition + gather" is a single `np.partition` on the same level indices, which is the route that is $O(n)$ in expectation; "histogram" is the $4096$-bin approximation, whose upper range is set at the median of the energies. The archive is `r6_quantile_cost.json`.

| $n$ | levels | sort + gather | partition + gather | histogram | exact error | histogram error |
|---|---|---|---|---|---|---|
| $2{,}112$ | $20$ | $0.018$ ms | $0.073$ ms | $0.253$ ms | $9.4\\%$ | $9.4\\%$ |
| $8{,}192$ | $24$ | $0.047$ ms | $0.261$ ms | $0.521$ ms | $6.3\\%$ | $6.4\\%$ |
| $32{,}768$ | $26$ | $0.215$ ms | $0.824$ ms | $1.34$ ms | $3.4\\%$ | $3.4\\%$ |
| $131{,}072$ | $27$ | $1.23$ ms | $3.28$ ms | $4.87$ ms | $3.7\\%$ | $3.6\\%$ |
| $524{,}288$ | $28$ | $6.22$ ms | $15.2$ ms | $16.9$ ms | $1.6\\%$ | $2.1\\%$ |
| $1{,}048{,}576$ | $28$ | $13.7$ ms | $23.0$ ms | $34.4$ ms | $0.8\\%$ | $1.7\\%$ |""")

add("§A.17 三点结论段重写",
"""Three practical points. The estimate scales linearly — the fitted log–log slope is $0.92$ over three
decades, against $1.11$ for a full sort, which is the expected $n\\log n$ — and costs about $75$ ns per
point at the top of the range, so a million-point spectrogram pays $78$ ms for a step that sits inside
a pipeline whose $\\ell_1$ recovery at that scale costs tens of seconds: the share falls as the
problem grows, from $0.17\\%$ in Table 25 to $0.02\\%$ here. Second, at the scales this paper uses, the
thirty selections cost about four times a single full sort ($0.30$ ms against $0.08$ ms at
$n=2{,}112$), because the implementation trades one pass for thirty; a single `partition` with all
thirty levels at once, or a sort below some cutoff, would remove that constant, and we note it as an
implementation detail rather than a property of the estimator. Third, the histogram approximation is
the genuine speed-up — a factor of four at $n=10^6$ — but it is not free in accuracy: on pure noise
its error is $8.6$–$13\\%$ where the exact estimate is within $1\\%$, and the error is one-sided, which
matters because a floor that is systematically high is a gate that is systematically aggressive.
Since the exact route is already a negligible share of the pipeline and the approximation is not, we
keep the exact one, and report the trade-off for implementations that cannot.""",
"""Three practical points, all of which differ from what the previous version of this section reported.

*First, the shipped route scales as $n\\log n$.* Its fitted log–log slope is $1.10$ across the six
sizes, against $0.94$ for the partition route, which is the linear law a genuine selection algorithm
should show. The distinction is not academic — it is the difference between the description we gave
and the code we ran — but it does not change a single number in this paper, because sorting and
partitioning return the same order statistics to machine precision: we verified that the two routes
produce bitwise-identical floor estimates at every size in the table. We therefore correct the text
and leave the implementation alone.

*Second, the $O(n)$ route is asymptotically better and slower in wall clock at every size measured.*
A single `np.partition` on the same twenty to twenty-eight indices costs $1.7$–$5.4$ times a full sort
over the range in the table ($0.073$ ms against $0.018$ ms at $n=2{,}112$; $23.0$ ms against
$13.7$ ms at $n=10^6$), because NumPy's sort is better optimised than introselect at this number of
order statistics, and the crossover lies beyond any size this paper reaches. The constant term is the
other half of the story: at $n=2{,}112$ the estimator spends about $0.10$ ms in the $\\chi^2$ quantile
call itself, five times what the data path costs, which is why the step looks expensive at small $n$
and cheap at large $n$ — from $8.6$ ns per point at $n=2{,}112$ to $13$ ns at $n=10^6$, and under
$0.05\\%$ of the pipeline once the $\\ell_1$ recovery of Table 25 is scaled to the same size.

*Third, the histogram approximation is dominated rather than merely inaccurate.* Under the same
protocol it is slower than a full sort at every size measured ($0.253$ ms against $0.018$ ms at
$n=2{,}112$; $34.4$ ms against $13.7$ ms at $n=10^6$), because setting its upper range at the median
costs one $O(n)$ selection before the $O(n)$ bin pass; the factor-of-four speed-up reported earlier
compared it against a full *sort plus interpolation* in a run without warm-up. And over twenty seeds
its accuracy is indistinguishable from the exact route at the sizes this paper uses — a mean absolute
error of $9.4\\%$ against $9.4\\%$ at $n=2{,}112$, $3.4\\%$ against $3.4\\%$ at $n=32{,}768$ — becoming
worse only at the top of the range ($1.7\\%$ against $0.8\\%$ at $n=10^6$). We keep the exact route on
both grounds, and note the one number the earlier accuracy claim got wrong: quoted from a single
realization it put the exact estimate "within $1\\%$" on pure noise, whereas the twenty-seed mean
absolute error is $9.4\\%$ at $n=2{,}112$, the worst seed reaching $27\\%$. That dispersion is the
finite-sample cost of reading twenty order statistics from a $2{,}112$-point array, and it is the same
quantity of which the $-2.3\\%$ finite-sample bias of Section 5.1(iii) is one component; it enters no
result, because Appendix A.4 shows the downstream gate to be insensitive to it, but it belongs with
the cost rather than with the accuracy, and is stated here.""")


# ================================================== 9. §7 归档清单与措辞
add("§7 旧归档标注被取代",
"`r5_complexity.json` for the quantile step of Table 37, `r5_band_floor.json`",
"`r5_complexity.json` for the quantile step of Table 37 as originally measured, superseded in the sixth round by `r6_quantile_cost.json` (Appendix A.17), `r5_band_floor.json`")


add("§7 第六轮归档补入清单",
"the gate-plus-weight design of Section 9.5 and Fig. 11. Code, the fixed configurations, the dependency list",
"the gate-plus-weight design of Section 9.5. The sixth round adds the re-measurement of the quantile step, `r6_quantile_cost.json` (six sizes, three routes, twenty seeds for the accuracy columns), which is the record behind the corrected Table 37 and Appendix A.17. Code, the fixed configurations, the dependency list")

add("§7 closed-form quantities 措辞",
"the closed-form quantities of Sections 4–5 are verified independently by two further scripts that evaluate the regularised incomplete Beta and chi-square functions directly and reproduce $p_c$ by Monte Carlo, agreeing with the closed form to within $0.2\\%$",
"the derived quantities of Sections 4–5 are verified independently by two further scripts that evaluate the regularised incomplete Beta and chi-square functions directly and reproduce $p_c$ by Monte Carlo, agreeing with the derivation to within $0.2\\%$")


# ================================================== 12. 渲染核查抓到的 closed form 残留
add("§4.2 正文：closed form → semi-analytic",
"exactly where the closed form is most delicate",
"exactly where the semi-analytic form is most delicate")

add("§4.2 Empirical 段：closed form → semi-analytic",
"wherever the closed form is used to advantage",
"wherever the semi-analytic form is used to advantage")

add("Fig. 1 题注一：closed form of Proposition 6",
"against the closed form of Proposition 6 (line)",
"against the semi-analytic form of Proposition 6 (line)")

add("Fig. 1 题注二：the closed form is not defined beyond it",
"and the closed form is not defined beyond it",
"and the form is not defined beyond it")


# R8（CSSP 投稿前意见）把 §11 结论里 "The semi-analytic reference-scale ratio locates the
# drift exactly" 一句改为「边界精确、比值是半解析近似」，本条的目标串因此被重写。
# **登记而非删除**：条目原样保留，重跑时显式跳过。
_SUPERSEDED_R8 = {
    '§11 结论：closed-form reference-scale ratio',
}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--dry", action="store_true", help="只报告将改动哪些条目")
    a = a.parse_args()

    s = MS.read_text()
    changed = skipped = missing = 0
    for tag, old, new in E:
        if tag in _SUPERSEDED_R8:
            print(f"  [R8 已承接] {tag}")
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
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
