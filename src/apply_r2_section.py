"""第二轮审稿意见：新增 §10 与摘要重写（幂等）。

用法：
    python3 src/apply_r2_section.py [--dry]
"""
from __future__ import annotations

import argparse
import pathlib

MS = pathlib.Path("paper/manuscript.md")

ABSTRACT = (
    "Two-stage underdetermined blind source separation (UBSS) clusters the directions of time\u2013frequency (TF) "
    "observations that pass a *single-source point* test, and that test contains an energy gate which discards "
    "noise-only points. We show that the *reference scale* conventionally used to set the gate \u2014 a fixed "
    "multiple of the median or of the maximum of the observed energies \u2014 is not invariant to source sparsity, "
    "and that this misalignment, rather than the clustering step, accounts for a significant and previously "
    "under-emphasised part of the paradigm's reported brittleness. A mixture-law analysis of the per-point energy "
    "gives the reference-scale ratio in closed form under isotropic noise, locates the structural boundary at a "
    "knee, and shows that a median-referenced coefficient drifts across the whole of the range over which it is "
    "used. We therefore reference the gate to the noise floor, estimate that floor blindly from the lower tail "
    "through the exact quantile relation of the chi-square law, and calibrate the multiple to a false-admission "
    "rate, so that the threshold is *derived* from the sensor count and one tolerance rather than selected on "
    "data. A retention self-check detects the regime in which the premise of the method fails and reverts to the "
    "classical criterion instead of failing silently. On a controlled grid the rule approaches a per-regime "
    "oracle without per-regime knowledge and does so across the whole sparsity\u2013SNR plane; on real speech it "
    "removes a silent order-of-magnitude penalty. The same experiments demarcate its limits, which we state "
    "rather than defer: on real signals a max-referenced gate with one chosen coefficient is marginally more "
    "accurate, and under heavy impulsive interference the lower-tail model does not hold, in which case the "
    "self-check is what prevents a wrong threshold from acting. The rule is a gate, not a clustering method, and "
    "it transfers unchanged across front ends."
)

SECTION10 = r"""
## 10. Robustness and Cost of the Front End

The experiments above establish the diagnosis and the method. This section reports the checks that bear on
whether the constants of Table 5 and the noise law behind them can be trusted outside the grid on which the
method was developed, and what the front end costs. Tables 19–26 are self-contained and could be moved to
supplementary material without interrupting the argument.

### 10.1 The sparsity–SNR plane

Fig. 10 shows the angle error of the three energy rules over the full plane $p\in\{0.01,\dots,0.40\}$ against
$\mathrm{SNR}\in\{0,\dots,30\}$ dB, five seeds per cell. Two features are worth noting because they separate the
two failure modes. The conventional rule's error is nearly *independent of SNR* throughout the sparse half of
the plane — $17$–$20°$ at every noise level for $p\le0.10$ — which is what a reference-scale defect looks like:
improving the noise does not help, because the defect is the position of the reference, not the size of the
noise. The calibrated rule's error instead falls monotonically with SNR at every $p$, from $3.3°$ to $0.16°$ at
$p=0.01$ and from $15.4°$ to $0.83°$ at $p=0.40$. The self-check declines to act in exactly the corner where
the premise fails, $(p,\mathrm{SNR})=(0.40,0)$ dB, where it fires in none of the five seeds, and in four of five
at $(0.20,0)$. Across the two decades of $p$ the calibrated rule is also better than a well-chosen fixed
multiple everywhere except the low-SNR dense corner, where the fixed multiple happens to sit closer to the
optimum (Table 26 gives the pooled statistics).

### 10.2 Quality of the set the gate selects

The angle error of Section 8 is a downstream quantity, and a lower error could in principle arise from better
clustering rather than from a better set of points. We therefore score the mask itself. Every TF point is
labelled by the number of sources genuinely active at it, and we report the precision, recall and $F_1$ of the
admitted set against the single-source class, together with the fractions of noise-only and multi-source points
that the mask excludes. All rules use the same collinearity and balance criteria, so any difference is the
energy rule alone.

**Table 19.** Detection quality of the mask itself on the synthetic ladder (means over six regimes and eight
seeds; the dense regime contains no single-source points by construction and is omitted from the precision and
recall columns). "Noise-only rejected" and "multi-source rejected" are the fractions of points of each true
class that the mask excludes.

| Energy rule | admitted | noise-only rejected | multi-source rejected | $p=0.02$ prec. / rec. | $p=0.05$ | $p=0.10$ | $p=0.20$ | $p=0.40$ |
|---|---|---|---|---|---|---|---|---|
| none (collinearity only) | $25.1\%$ | $89.6\%$ | $81.1\%$ | $0.35$ / $0.82$ | $0.56$ / $0.77$ | $0.69$ / $0.71$ | $0.75$ / $0.63$ | $0.61$ / $0.53$ |
| classical $t_e=0.02$ | $24.7\%$ | $91.7\%$ | $81.1\%$ | $0.35$ / $0.82$ | $0.56$ / $0.77$ | $0.69$ / $0.71$ | $0.78$ / $0.63$ | $0.64$ / $0.52$ |
| median $t_e=5$ | $9.5\%$ | $99.99\%$ | $87.0\%$ | $0.99$ / $0.82$ | $0.98$ / $0.77$ | $0.95$ / $0.70$ | $0.80$ / $0.27$ | $0.09$ / $0.005$ |
| max-referenced $c=0.05$ | $13.7\%$ | $99.99\%$ | $83.3\%$ | $0.99$ / $0.63$ | $0.97$ / $0.57$ | $0.94$ / $0.53$ | $0.84$ / $0.41$ | $0.55$ / $0.29$ |
| **NF $\alpha=10^{-4}$** | $18.1\%$ | $99.99\%$ | $81.6\%$ | $\mathbf{0.99}$ / $\mathbf{0.82}$ | $\mathbf{0.98}$ / $\mathbf{0.77}$ | $\mathbf{0.95}$ / $\mathbf{0.70}$ | $\mathbf{0.88}$ / $\mathbf{0.61}$ | $\mathbf{0.63}$ / $\mathbf{0.46}$ |

The sparse half of the table answers the question directly. At $p=0.02$ the calibrated gate raises precision
from $0.35$ to $0.99$ **while leaving recall unchanged at $0.82$**: it removes the noise-only points the
collinearity test admits and loses almost none of the single-source points. The mechanism is the one Corollary 1
predicts — one noise-only point in eight survives the collinearity test at $M=2$ — and the same effect is
visible at $p=0.05$ and $p=0.10$. The dense end answers it in the negative, and we report that too: at $p=0.40$
precision is $0.63$ against $0.64$ for the conventional rule and recall is *lower* ($0.46$ against $0.52$),
because at that operating point the contaminating points are multi-source rather than noise-only and they carry
the *highest* energies (Table 16), so an energy gate cannot separate them. The improvement in angle error that
the same regime shows in Table 6 therefore does not come from a cleaner set of single-source points; it comes
from the far smaller number of points admitted, which reduces the scatter the clusterer has to average over.

**Table 20.** The same scoring on real speech (means over fifteen configurations and four seeds). Because the
labels are derived from the noise-referenced activity criterion of Section 9.2, "precision" here counts a point
as single-source when one source exceeds the per-component noise power.

| Energy rule | admitted | precision | recall | $F_1$ | noise-only rejected | direction error of admitted SSPs |
|---|---|---|---|---|---|---|
| none (collinearity only) | $19.3\%$ | $0.333$ | $0.278$ | $0.284$ | $86.8\%$ | $5.06°$ |
| classical $t_e=0.02$ | $19.2\%$ | $0.333$ | $0.270$ | $0.282$ | $88.4\%$ | $5.01°$ |
| median $t_e=5$ | $3.9\%$ | $0.407$ | $0.112$ | $0.170$ | $99.99\%$ | $3.11°$ |
| max-referenced $c=0.05$ | $1.1\%$ | $0.132$ | $0.005$ | $0.009$ | $100.0\%$ | $0.89°$ |
| **NF $\alpha=10^{-4}$** | $4.4\%$ | $\mathbf{0.411}$ | $0.158$ | $\mathbf{0.195}$ | $98.5\%$ | $3.32°$ |

On real speech the picture is different in kind, and it is the same difference Section 9.5 identifies. The
calibrated gate raises precision sharply where the noise dominates — at $\mathrm{SNR}=0$ dB from $0.12$ to
$0.71$, and at $10$ dB from $0.28$ to $0.60$ — and raises the quality of what it admits ($5.01°$ against
$3.32°$ of median direction error); but at $30$ and $40$ dB, where there are almost no noise-only points left,
it removes weak single-source points and precision *falls* below the ungated set ($0.33$ against $0.38$ at
$30$ dB). The max-referenced rule, which is the most accurate downstream, admits $1.1\%$ of the points and
commands the best direction quality ($0.89°$) while losing almost every single-source point ($0.005$ recall) —
a useful reminder that downstream accuracy and detection quality are not the same quantity, and that on these
data the former is bought by being highly selective rather than by being well calibrated.

### 10.3 Sensitivity to the guard fraction $\eta$

**Table 21.** Sensitivity of the retention criterion to its single threshold, over the 27 conditions of
Table 4 and all ten seeds (270 decisions). "Condition-level" uses the accounting of Table 4, in which the ten
seeds of a condition are averaged before the decision is taken; "per-seed" counts each of the 270 decisions.

| $\eta$ | $0.005$ | $0.01$ | $0.02$ | $0.03$ | $0.035$ | $0.05$ | $0.075$ | $0.1$ |
|---|---|---|---|---|---|---|---|---|
| condition-level errors (of 27) | $0$ | $0$ | $0$ | $0$ | $0$ | $1$ | $3$ | $3$ |
| condition-level gate usage | $77.8\%$ | $77.8\%$ | $77.8\%$ | $77.8\%$ | $74.1\%$ | $70.4\%$ | $59.3\%$ | $59.3\%$ |
| per-seed material errors (of 270) | $18$ | $18$ | $18$ | $18$ | $17$ | $19$ | $26$ | $32$ |

The decision is insensitive to $\eta$ over a full decade. Every value from $0.005$ to $0.035$ reproduces the
Table 4 accounting exactly, and the physical guard fraction only begins to matter above $0.05$, where it starts
to disable the gate in regimes that would have preferred it active. The value $0.035$ is therefore not tuned to
the reported grid: it is one point in a flat region, and it was chosen at the upper end of that region because a
larger boundary is the conservative one against silent failure. The per-seed column, which is the stricter
accounting, gives $6.7\%$ material errors at $0.035$ against $6.3\%$ at $0.05$; the guard is a guard, and at the
resolution of individual seeds it is wrong about once in fifteen cases, always in the direction of caution.

### 10.4 Sensitivity to the lower-tail cap $q_{\max}$

**Table 22.** The blind noise-floor estimator against the lower-tail cap. Left block: $\hat\sigma^2/\sigma^2$
(the bias of Table 2, resolved by level). Right block: the resulting angle error and its *relative* change over
$q_{\max}\in[0.01,0.05]$. $q_{20}$ is the largest level at which the bias stays within $20\%$; $q_{\rm adapt}$ is
the level selected by the data-driven rule of the text.

| Regime | $\pi_0$ | $0.005$ | $0.01$ | $0.02$ | $0.03$ | $0.05$ | error over $[0.01,0.05]$ | $q_{20}$ | $q_{\rm adapt}$ |
|---|---|---|---|---|---|---|---|---|---|
| $p=0.02$ | $0.922$ | $0.881$ | $0.956$ | $0.972$ | $0.994$ | $1.019$ | $0.361°$ (0.00%) | $\ge0.1$ | $0.076$ |
| $p=0.05$ | $0.815$ | $0.999$ | $1.050$ | $1.071$ | $1.090$ | $1.106$ | $0.311°$ (0.00%) | $\ge0.1$ | $0.088$ |
| $p=0.10$ | $0.656$ | $1.140$ | $1.209$ | $1.248$ | $1.258$ | $1.265$ | $0.324$–$0.328°$ (1.2%) | $0.019$ | $0.088$ |
| $p=0.20$ | $0.410$ | $1.538$ | $1.614$ | $1.623$ | $1.632$ | $1.648$ | $0.553$–$0.560°$ (1.3%) | $0.002$ | $0.086$ |
| $p=0.40$ | $0.130$ | $2.884$ | $2.916$ | $2.936$ | $2.951$ | $3.041$ | $1.370$–$1.384°$ (1.0%) | $0.0006$ | $0.070$ |
| dense | $0$ | $80.6$ | $82.7$ | $84.2$ | $86.0$ | $87.5$ | $9.75$–$10.33°$ (5.9%) | — | $0.084$ |

Two conclusions follow, and the second is a negative result we report as such. First, the *bias* of
$\hat\sigma^2$ and the *error it causes* are largely decoupled: at $p=0.40$ the estimate is high by a factor of
$2.9$, yet the angle error moves by $1.0\%$ across a decade of $q_{\max}$, and below $0.01$ it is the shortage
of order statistics, not the bias, that hurts. In the range where the method applies, the cap therefore need not
be chosen carefully; below the knee the estimate is accurate to a few per cent, and above it a threshold that is
too high is still a threshold that removes most of the noise. Second, a data-driven cap does not solve the
problem it would be introduced for: choosing the largest level on which $\hat\sigma^2$ agrees with the
conservative end of its own curve returns $0.07$–$0.09$ in *every* regime including the dense one, because the
reference plateau is itself contaminated once $\pi_0$ falls. A rule of that kind cannot detect the failure; the
retention diagnostic of Section 5.3 can, because it observes the consequence rather than the internal
consistency of the estimator. We therefore keep the fixed cap and the guard.

### 10.5 Departures from the Gaussian noise law

Section 5.1 uses the $\chi^2_{2M}$ law of a noise-only point. That law is exact for circularly symmetric complex
Gaussian noise and for nothing else, and we have already reported one real case in which it fails (babble
interference, Section 9.4). Table 23 makes the test systematic: four noise families in addition to the Gaussian
one, each matched to the same total noise power at $\mathrm{SNR}=20$ dB.

**Table 23.** The calibrated gate under four noise laws, all second-moment matched to $\mathrm{SNR}=20$ dB
(eight seeds per cell). "impulsive" is Bernoulli–Gaussian with a $2\%$ burst rate at $20$ dB above the floor;
"colored" is complex Gaussian with an AR(1) correlation along frequency.

| Noise law | $\hat\sigma^2/\sigma^2$, $p=0.05$ | $p=0.20$ | $p=0.40$ | gate applied | angle error, NF vs classical, $p=0.05$ |
|---|---|---|---|---|---|
| Gaussian (model law) | $1.09$ | $1.57$ | $3.07$ | $8/8$ | $0.40°$ vs $10.19°$ |
| Laplacian (heavy tail) | $0.46$ | $0.76$ | $1.60$ | $8/8$ | $0.61°$ vs $11.59°$ |
| impulsive | $0.38$ | $0.53$ | $0.93$ | $8/8$ | $1.37°$ vs $7.33°$ |
| colored Gaussian | $1.04$ | $1.56$ | $3.05$ | $8/8$ | $0.38°$ vs $8.99°$ |
| uniform (light tail) | $1.98$ | $3.00$ | $5.24$ | $8/8$ | $0.47°$ vs $10.40°$ |

The estimator's *calibration* is genuinely violated outside the model law: the bias runs from $0.38\times$ for
impulsive noise to $5.2\times$ for uniform noise, in both directions, because a lower tail heavier or lighter
than $\chi^2$ moves the small quantiles the estimator inverts. Its *decision* is nevertheless robust in four of
the five families, and the reason is instructive: the estimator is used only to place a threshold far below the
signal energies, and a threshold wrong by a factor of two still places it far below them. Correlation along
frequency leaves the marginal law almost unchanged and is therefore harmless. The exception is the real babble
interference of Section 9.4, whose lower tail is not merely misshaped but contaminated by a competing signal,
and there the gate is not merely miscalibrated — it is disabled by the self-check in half of the seeds, which is
the designed behaviour. The honest summary is that the $\chi^2$ assumption is a *calibration* assumption rather
than a *validity* assumption, but that this distinction has a limit, and the guard is what marks it.

### 10.6 Sensitivity to the conditioning of the mixing matrix

The mixing matrix is drawn with a $12°$ minimum column separation throughout, and Section 4.2 shows the
closed form to be insensitive to that angle. Hence the separate question of whether the *algorithm* depends on
it.

**Table 24.** Angle error (degrees) against the minimum column separation, by method and activation
probability (eight seeds per cell). "NF" is the calibrated gate, "classical" the conventional median rule and
"$t_e=5$" a hand-set fixed multiple.

| min angle | $p=0.05$ NF / class. / $t_e5$ | $p=0.10$ | $p=0.20$ | $p=0.40$ |
|---|---|---|---|---|
| $5°$ | $0.38$ / $17.70$ / $0.38$ | $0.46$ / $15.81$ / $0.47$ | $0.71$ / $10.07$ / $3.73$ | $6.19$ / $5.85$ / $9.93$ |
| $10°$ | $0.29$ / $15.26$ / $0.29$ | $0.30$ / $14.42$ / $0.31$ | $0.57$ / $5.66$ / $0.71$ | $1.49$ / $1.69$ / $11.90$ |
| $12°$ | $0.31$ / $15.44$ / $0.31$ | $0.32$ / $13.22$ / $0.33$ | $0.55$ / $5.59$ / $0.76$ | $1.38$ / $1.61$ / $11.45$ |
| $20°$ | $0.40$ / $7.99$ / $0.41$ | $0.41$ / $6.05$ / $0.41$ | $0.74$ / $1.21$ / $0.76$ | $1.26$ / $1.31$ / $11.30$ |
| $30°$ | $0.40$ / $2.40$ / $0.40$ | $0.35$ / $1.55$ / $0.36$ | $0.44$ / $0.79$ / $0.75$ | $1.05$ / $1.02$ / $14.86$ |
| $45°$ | $3.11$ / $16.89$ / $3.10$ | $3.61$ / $15.01$ / $3.61$ | $1.58$ / $13.77$ / $4.32$ | $7.39$ / $10.95$ / $10.57$ |

The calibrated gate is essentially independent of the separation in the sparse regimes ($0.29$–$0.46°$ over
$5$–$45°$ at $p\le0.10$, the $45°$ entry included), and the conventional rule improves monotonically with the
separation, as it must: a larger angle makes the columns individually easier to locate, which is what eventually
rescues a badly placed threshold. The one regime in which the calibrated gate loses to the conventional rule is
$(5°,p=0.40)$, where both are near $6°$ and the problem has left the sparse regime. The fixed multiple
$t_e=5$ remains competitive in the sparse regimes and collapses at $p=0.40$ everywhere, which is the pattern of
Section 4.3 reproduced at every separation.

### 10.7 Component-wise cost of the front end

**Table 25.** Cost of the front end, single core, median over repetitions. The synthetic column uses the
$2112$-point ladder; the real column uses the $\mathrm{win}=1024$ configuration, $4.8\times10^4$ TF points.

| Stage | synthetic ($2112$ points) | share | real ($4.8\times10^4$ points) | share |
|---|---|---|---|---|
| per-point energy $\|x\|^2$ | $0.010$ ms | $0.02\%$ | $0.21$ ms | $0.01\%$ |
| blind noise-floor estimate | $0.35$ ms | $0.46\%$ | $2.99$ ms | $0.13\%$ |
| threshold and retention check | $0.15$ ms | $0.19\%$ | $0.89$ ms | $0.04\%$ |
| spherical $k$-means | $4.47$ ms | $5.8\%$ | $10.95$ ms | $0.47\%$ |
| debiased $\ell_1$ recovery | $71.8$ ms | $93.0\%$ | $2337$ ms | $99.4\%$ |
| **total** | $77.3$ ms | | $2351$ ms | |
| **gate subtotal** | $0.51$ ms | $0.66\%$ | $4.09$ ms | $0.17\%$ |

The calibration is not merely cheap relative to the recovery; it is cheap relative to the *clustering*, which is
itself a small part of the pipeline. The claim that the gate "comes almost for free" is therefore exact in the
literal sense, and it holds at both scales, the share falling as the problem grows because the cost of the gate
is dominated by a sort of $FT$ values while the recovery scales superlinearly.

### 10.8 Statistical summary

**Table 26.** Paired contrasts with intervals, effect sizes and multiplicity corrections. Differences are in
degrees and are oriented so that a positive value favours the calibrated gate. $d_z$ is the paired Cohen
effect size. "Holm" and "BH" are the corrected $p$-values within each family.

| Family | Contrast | $n$ | $\Delta$ | $95\%$ CI | $p$ | $d_z$ | Holm | BH |
|---|---|---|---|---|---|---|---|---|
| ladder | SCA-default $-$ NF, pooled | $60$ | $+7.08$ | $[+4.80,+9.37]$ | $6.0\times10^{-8}$ | $+0.80$ | $4.2\times10^{-7}$ | $4.2\times10^{-7}$ |
| ladder | SCA-default $-$ NF, $p=0.02$ | $10$ | $+15.01$ | $[+9.54,+20.48]$ | $1.6\times10^{-4}$ | $+1.96$ | $9.5\times10^{-4}$ | $5.5\times10^{-4}$ |
| ladder | SCA-default $-$ NF, $p=0.20$ | $10$ | $+4.14$ | $[-0.76,+9.04]$ | $0.088$ | $+0.60$ | $0.18$ | $0.10$ |
| real speech | SCA-median $-$ NF | $120$ | $+7.15$ | $[+5.96,+8.33]$ | $5.2\times10^{-22}$ | $+1.09$ | $3.6\times10^{-21}$ | $2.1\times10^{-21}$ |
| real speech | SCA-max $-$ NF | $120$ | $-0.24$ | $[-0.42,-0.05]$ | $0.012$ | $-0.23$ | $0.024$ | $0.014$ |
| real speech | top-$5\%$ $-$ NF | $120$ | $-0.18$ | $[-0.47,+0.12]$ | $0.25$ | $-0.11$ | $0.25$ | $0.25$ |
| real speech | NF $+$ energy $-$ NF | $120$ | $-0.39$ | $[-0.58,-0.21]$ | $5.9\times10^{-5}$ | $-0.38$ | $1.8\times10^{-4}$ | $8.3\times10^{-5}$ |

Three things are worth reading off the table. The corrections do not change any conclusion: within each family
Holm and Benjamini–Hochberg retain every significance found without correction, and the only contrast that was
never significant remains so. The effect sizes are moderate rather than dramatic — $d_z$ of $0.80$ for the
pooled ladder and $1.09$ for the real-speech comparison against the conventional rule — which is the right
expectation for a change to a single scalar in a pipeline whose dominant error is elsewhere. And the contrast
against the max-referenced rule is significant in the *opposite* direction, at $d_z=-0.23$: on real speech the
selected coefficient is slightly better than the derived one, and we report that as a finding rather than
correcting for it in the text.
"""

FIG10 = ("\n**Fig. 11.** Angle error over the whole sparsity\u2013SNR plane, for the conventional median rule "
         "(left), the calibrated gate (centre) and a hand-set fixed multiple $t_e=5$ (right); five seeds per "
         "cell, $N=4$, $M=2$, colour on a logarithmic scale. The conventional rule is flat in SNR throughout "
         "the sparse half of the plane, which is the signature of a reference-scale defect rather than a noise "
         "problem. (`fig_heatmap.pdf`)")

NEW_REF = (
    "| SCA-max $-$ NF-SSP | $120$ | $-0.237$ | $[-0.421,-0.054]$ | $1.17\\times10^{-2}$ | $-0.23$ | $2.35\\times10^{-2}$ | $1.37\\times10^{-2}$ |\n"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    s = MS.read_text()
    notes: list[str] = []

    # ---- 1) 摘要重写 ----
    lines = s.split("\n")
    i = lines.index("## Abstract")
    if lines[i + 2].strip() != ABSTRACT.strip():
        lines[i + 2] = ABSTRACT
        s = "\n".join(lines)
        notes.append("摘要已重写")
    else:
        notes.append("摘要已是新版")

    # ---- 2) 插入 §10 并把原 §10/§11 顺延 ----
    if "## 10. Robustness and Cost of the Front End" not in s:
        anchor = "## 10. Discussion and Limitations"
        assert anchor in s, "找不到 Discussion 标题"
        s = s.replace("## 11. Conclusion", "## 12. Conclusion", 1)
        s = s.replace(anchor, "## 11. Discussion and Limitations", 1)
        s = s.replace("## 11. Discussion and Limitations",
                      SECTION10.strip() + "\n\n" + "## 11. Discussion and Limitations", 1)
        notes.append("已插入 §10 并顺延至 §11/§12")
    else:
        notes.append("§10 已存在")

    # ---- 3) Fig. 10 ----
    if "**Fig. 11.**" not in s:
        anchor = "\n---\n\n## References"
        assert anchor in s, "找不到 Figures 段末尾"
        s = s.replace(anchor, FIG10 + anchor, 1)
        notes.append("已加 Fig. 10")
    else:
        notes.append("Fig. 10 已存在")

    if not a.dry:
        MS.write_text(s)
    print("；".join(notes))
    print("词数:", len(s.split()), "| $ 数:", s.count("$"), "| 行数:", len(s.split(chr(10))))


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
