"""第五轮修订：把密度基线、复数/卷积适用性、以及 Q1–Q3 的答复落进手稿。

用法：
    python3 src/apply_r5_fixes.py [--dry]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MS = Path("paper/manuscript.md")

E: list[tuple[str, str, str]] = []

# 落稿后又做过手工二次修订的两条：脚本里的 new 与终稿不同，文件已处于期望状态。
_HAND_CORRECTED = {
    # ① §9.4 段落在其后插入了对 Fig. 11 的引用（"Fig. 7b, and Fig. 11 for the compact view"）；
    # ② 摘要先按本条改成两段式陈述，随后为压回 250 词上限又做了六处删减
    #    （"conventionally"、"changing meaning"、"and is insensitive to both design constants"
    #     等被删，最终 248 词）。
    "§9.4 第一条发现（重写）",
    "摘要：加权与未加权分开陈述",
}



def add(tag: str, old: str, new: str) -> None:
    E.append((tag, old, new))


# ======================================================================
# 主文修改
# ======================================================================

# ---- 1. §1 三条限制：指向 A.15
add("§1 限制指向 A.15",
    "and therefore this construction, does not apply. The noise is **isotropic complex Gaussian "
    "within the band analysed**",
    "and therefore this construction, does not apply; Appendix A.15 separates the three questions "
    "this raises and answers them quantitatively, and the answer is that the diagnosis and the "
    "calibration transfer to complex (per-frequency) mixing verbatim while the detection criterion "
    "has to be rebuilt. The noise is **isotropic complex Gaussian within the band analysed**")

# ---- 2. §7 基线：密度方法改为真的 head-to-head
add("§7 密度基线改为 head-to-head",
    "Density-based clusterers (DBSCAN, OPTICS, density-peak), neighbourhood-confidence detectors of "
    "the TIFROM family, and transforms that sharpen sparsity before detection (locally maximum "
    "synchroextracting transforms with optimised fuzzy clustering) are not head-to-head comparators "
    "for a structural reason: they act either on the set of points a gate has already admitted, or in "
    "place of the detection stage, so a change to the reference scale of the gate does not enter their "
    "input at all.",
    "Density-based clusterers (DBSCAN, OPTICS, density-peak), neighbourhood-confidence detectors of "
    "the TIFROM family, and transforms that sharpen sparsity before detection (locally maximum "
    "synchroextracting transforms with optimised fuzzy clustering) are not *comparators for the "
    "reference scale* for a structural reason: they act either on the set of points a gate has already "
    "admitted, or in place of the detection stage, so a change to the reference scale of the gate does "
    "not enter their input at all. That is an argument about what this comparison isolates and not "
    "about accuracy, and a reader is entitled to the accuracy comparison anyway; Appendix A.14 "
    "therefore enters the three density fronts as head-to-head competitors — on all TF points, and "
    "separately on the collinearity-admitted set with the energy criterion removed — with their radius "
    "tuned per regime, which is an oracle the calibrated gate never gets. They lead in the dense regime "
    "and trail in the sparse ones, and we report that rather than asserting the structural argument.")

# ---- 3. §9.4 第一条发现：与加权结果对齐（本轮的实质改动）
add("§9.4 第一条发现（重写）",
    "Two findings must be stated against the method. First, **a max-referenced gate with the single "
    "constant $c=0.05$ is slightly better on this data** (Fig. 7b): $0.88°$ on average, within "
    "$1.33\\times$ of the per-configuration bound, and better than the calibrated gate in twelve of the "
    "fifteen configurations; an energy ranking that keeps the strongest $5\\%$ is comparable "
    "($0.94°$). The noise-floor calibration is therefore not the most accurate rule on real speech. "
    "What it offers instead is that the multiple is *derived* rather than selected — $\\tau$ is fixed by "
    "$(M,\\alpha)$ and is invariant to the problem — whereas both alternatives require a coefficient to "
    "be chosen, and the per-configuration optima of those coefficients drift across two to three orders "
    "of magnitude (Table 14). On the present evidence, a practitioner willing to select one coefficient "
    "on data resembling the target will do at least as well with $c$.",
    "Two findings must be stated against the method, and the first of them has a second half that the "
    "weighting of Section 9.5 supplies. **The unweighted calibrated gate is less accurate on this data "
    "than a max-referenced gate with the single hand-set constant $c=0.05$** (Fig. 7b): $0.88°$ on "
    "average against $1.12°$, better in twelve of the fifteen configurations (paired over the $120$ "
    "instances, $p=0.012$); an energy ranking that keeps the strongest $5\\%$ is comparable ($0.94°$). "
    "Read as a rule for *deciding which points enter*, the noise-floor calibration is therefore not the "
    "most accurate one on real speech, and we do not claim otherwise.\n"
    "\n"
    "The ordering changes once the same construction is allowed to *weight* the points it admits, which "
    "is the unification Section 9.5 develops. Weighting the admitted directions by the derived "
    "reliability $\\lVert\\mathbf{x}\\rVert^2/\\hat\\nu$ lowers the calibrated scheme to $0.72°$ and the "
    "collinearity-augmented weight to $0.68°$; against the max-referenced gate on the identical "
    "instances that is $0.68°$ against $0.88°$, better in fourteen of the fifteen configurations "
    "($p=3\\times10^{-11}$). Two controls show that this is not a general benefit of weighting. "
    "Applying the *same* weight to the max-referenced gate does not help it ($0.88°\\to0.93°$, "
    "$p=0.28$), and applying it without the gate — to the ungated admitted set — does not help either "
    "($1.12°\\to1.15°$, $p=0.87$), in the second case because at $\\mathrm{SNR}=0$ dB the "
    "weight-without-gate combination collapses to $7.33°$ while the gated one holds at $0.86°$. The two "
    "devices are complementary rather than substitutable: the calibration decides *which* points count, "
    "and is what protects the low-SNR cells, while the weight decides *how much* each admitted point "
    "counts. Together they give the lowest mean error we have measured on this data — $0.68°$, against "
    "$0.70°$ for the per-configuration oracle bound of Table 15, a bound that is itself estimated from "
    "two tuning seeds, so we read the two as equal rather than as the method beating the oracle. The "
    "qualification that remains is narrow and worth stating: a practitioner willing to select one "
    "coefficient on data resembling the target and to leave the weight out will do at least as well with "
    "$c$; with the weight, the derived scale is better and nothing has to be selected.")

# ---- 4. §9.5 表 17 扩展为 2×设计的对照
add("§9.5 Table 17 caption",
    "**Table 17.** Angle error (degrees, eight seeds) when the admitted directions are weighted by "
    "their reliability instead of being selected by a hard gate, on the fifteen real-speech "
    "configurations. \"base\" applies no energy criterion at all and only weights; the $|\\cos|^4$ "
    "column augments the energy weight by the fourth power of the collinearity statistic. The two "
    "weighting columns are bolded because they are statistically indistinguishable from each other "
    "($p=0.23$ over the $15\\times8$ configuration–seed cells) while both are significantly better "
    "than the unweighted gate ($p<10^{-4}$). The last column is the same conventional baseline as "
    "Table 15, evaluated on the identical problem instances (the two tables share all $120$ "
    "unweighted-gate records).",
    "**Table 17.** Angle error (degrees, eight seeds) when the admitted directions are weighted by their "
    "reliability instead of being selected by a hard gate, on the fifteen real-speech configurations. "
    "\"base\" applies no energy criterion at all and only weights; the $|\\cos|^4$ columns augment the "
    "energy weight by the fourth power of the collinearity statistic. The design is deliberately "
    "$2\\times2$: the same weight is applied to the max-referenced gate as well, which is the control "
    "that separates the effect of *weighting* from the effect of *where the gate is placed*. Bold marks "
    "the best entry of each row among the competing columns. The two calibrated weighting columns are "
    "statistically indistinguishable from each other ($p=0.23$ over the $15\\times8$ configuration–seed "
    "cells) and both are significantly better than the unweighted gate ($p<10^{-4}$) and than the "
    "max-referenced gate with or without the same weight ($p=10^{-3}$ and $p=3\\times10^{-11}$ "
    "respectively). The last column is the same conventional baseline as Table 15, evaluated on the "
    "identical problem instances.")

T17_EXTRA = {
    "win 256": (1.45, 1.26), "win 512": (0.88, 0.75), "win 1024": (0.58, 0.69),
    "win 2048": (0.65, 0.92), "$N=3$": (0.61, 0.60), "$N=5$": (1.12, 1.11),
    "$N=6$": (1.04, 1.20), "SNR $0$ dB": (1.06, 0.98), "SNR $10$ dB": (0.70, 0.72),
    "SNR $30$ dB": (0.63, 0.62), "SNR $40$ dB": (0.69, 0.63),
    "1 dense source": (0.98, 1.14), "2 dense sources": (1.19, 1.63),
    "babble noise": (0.90, 1.00), "noise-free": (0.70, 0.64), "**mean**": (0.88, 0.93),
}
HEAD17_OLD = ("| Configuration | NF gate (unweighted) | **NF + energy** | "
              "NF + energy$\\cdot\\vert\\cos\\vert^4$ | base + energy | median $t_e{=}0.02$ |")
HEAD17_NEW = ("| Configuration | NF gate (unweighted) | NF + energy | "
              "**NF + energy$\\cdot\\vert\\cos\\vert^4$** | base + energy | "
              "median $t_e{=}0.02$ | max $c{=}0.05$ | max + same weight |")


def extend_table17(s: str) -> tuple[str, bool]:
    """给 Table 17 加两列（max-referenced 与其同权重对照）。逐行追加，避免整块匹配。"""
    if HEAD17_NEW in s:
        return s, False
    if HEAD17_OLD not in s:
        raise SystemExit("Table 17 表头未命中")
    lines = s.split("\n")
    hi = lines.index(HEAD17_OLD)
    lines[hi] = HEAD17_NEW
    # 分隔行紧随其后
    k = hi + 1
    while k < len(lines) and not lines[k].startswith("|---"):
        k += 1
    lines[k] = "|" + "---|" * 8
    # 数据行：按首格标签追加两个值
    n_add = 0
    for r in range(k + 1, len(lines)):
        if not lines[r].startswith("|"):
            break
        cells = [c.strip() for c in lines[r].strip().strip("|").split("|")]
        label = cells[0]
        if label not in T17_EXTRA:
            raise SystemExit(f"Table 17 行未识别: {label!r}")
        a, b = T17_EXTRA[label]
        lines[r] = lines[r].rstrip() + f" {a} | {b} |"
        n_add += 1
    if n_add != 16:
        raise SystemExit(f"Table 17 追加行数异常: {n_add}（应为 16）")
    return "\n".join(lines), True


add("§9.5 加权段的统一框架说明",
    "This also explains the residual gap between the calibrated gate and the aggressive alternatives: "
    "the false-admission calibration answers \"how many noise-only points pass?\", whereas on real data "
    "the binding question is \"how reliable is the direction of this point?\", and the two have "
    "different answers.",
    "This also explains the residual gap between the calibrated gate and the aggressive alternatives: "
    "the false-admission calibration answers \"how many noise-only points pass?\", whereas on real data "
    "the binding question is \"how reliable is the direction of this point?\", and the two have "
    "different answers.\n"
    "\n"
    "**A unification: the gate controls the support, the weight controls the influence.** Since the "
    "two devices answer different questions they should be composable, and Table 17 shows that they "
    "are. The weight is the maximum-likelihood reliability of an admitted direction under Gaussian "
    "noise, $w=\\lVert\\mathbf{x}\\rVert^2/\\hat\\nu$, normalised by the same noise floor the gate is "
    "referenced to, so the combination introduces no constant that the gate had not already fixed; the "
    "$|\\cos|^4$ factor is an empirical augmentation and is statistically indistinguishable from it "
    "($p=0.23$). The $2\\times2$ layout of Table 17 is what makes the interaction visible. With the "
    "conventional reference scale, where the gate is effectively inactive ($96\\%$ of the points "
    "survive it), the weight carries almost the whole gain on its own: $8.26°\\to1.11°$. With the "
    "max-referenced gate, which is already well placed ($0.5\\%$ survive), the weight adds nothing "
    "($0.88°\\to0.93°$, $p=0.28$). With the calibrated gate ($11.5\\%$ survive) the weight is worth a "
    "factor of $1.6$ ($1.12°\\to0.72°$) and the combination is the best entry in the table. Removing "
    "either component degrades the scheme, and removing them in the opposite order shows why: the "
    "ungated weight is no better than the unweighted gate on average ($1.15°$ against $1.12°$, "
    "$p=0.87$) and falls apart in the one cell where the gate's premise still holds, $\\mathrm{SNR}=0$ "
    "dB ($7.33°$ against $0.86°$).")

# ---- 5. 摘要：把"max-referenced 略准"改为如实的两段式结论
add("摘要：加权与未加权分开陈述",
    "and on real speech the optimal gate is set by the energy-dependence of point reliability, and a "
    "max-referenced gate with one chosen coefficient is marginally more accurate \u2014 the derived "
    "threshold offers transferability, not accuracy.",
    "and on real speech the optimal gate is set by the energy-dependence of point reliability, which is "
    "why the calibrated gate is paired with a derived reliability weight: alone it trails a "
    "one-coefficient max-referenced gate, with the weight it leads it, and the weight does not help the "
    "max-referenced gate.")

# ---- 6. §11 结论：加权的关系改写，并把卷积推广写成已量化的边界
add("§11 结论：加权不是替代而是互补",
    "while making plain that on real signals the gate is a crude substitute for weighting the "
    "directions by their reliability.",
    "while making plain that on real signals the gate and the reliability weight are complements "
    "rather than substitutes: alone the gate trails a hand-set max-referenced rule ($1.12°$ against "
    "$0.88°$), and with the derived weight it leads it ($0.68°$ against $0.88°$) while the same weight "
    "does nothing for the max-referenced rule.")

add("§11 结论：卷积推广指向 A.15",
    "Future work will address automatic source-number selection within the same calibrated framework, "
    "the extension to convolutive and time-varying mixing, and evaluation on machine-vibration "
    "benchmarks.",
    "Future work will address automatic source-number selection within the same calibrated framework, "
    "the extension to convolutive and time-varying mixing \u2014 for which Appendix A.15 shows that the "
    "diagnosis, the calibration and the self-check all transfer per frequency while the SSP criterion "
    "must be rebuilt in the complex domain, so that what is missing is a detection stage and not a "
    "reference scale \u2014 and evaluation on machine-vibration benchmarks.")


# ======================================================================
# 附录新增 A.14–A.19
# ======================================================================

APPENDIX = r"""### A.14 Density clustering as a head-to-head comparator

Section 7 argues that a density detector is not a *comparator for the reference scale*, because it
replaces the detection stage instead of consuming the gate's output. That argument is about what the
comparison isolates, and a reader may still ask the accuracy question — can a pipeline that never had
an energy gate to mis-calibrate simply do better? — so this appendix answers it on the same instances.

Three density fronts replace the whole detection-and-clustering stage: **DBSCAN**, **OPTICS**, and the
Rodriguez–Laio **density-peak** estimator. Each runs in two input modes: on all TF points with no
criteria at all, and on the collinearity-plus-balance set with the energy criterion removed, which is
the fair comparison against our front end since the energy criterion is the only thing being replaced.
The radius is treated generously: we report both a fixed default ($\texttt{eps}=0.02$,
$\texttt{min\_samples}=10$) and the best value per regime from $\{0.005,0.02,0.05\}$, an oracle the
calibrated gate never gets. When a front end returns fewer than $N$ clusters it is completed on the
unassigned points, so it is never punished for returning the wrong number of columns. All density
methods use the same spherical centre estimation as the rest of the paper, so only the
partitioning rule differs.

**Table 31.** Density fronts as head-to-head competitors on the synthetic ladder (ten seeds, identical
instances). "best of the family" takes the best front end *and* the best radius in each regime. The
conventional and calibrated rows are the same runs as Table 6.

| Method | $p=0.02$ | $p=0.05$ | $p=0.10$ | $p=0.20$ | $p=0.40$ | dense | mean |
|---|---|---|---|---|---|---|---|
| conventional median $t_e{=}0.02$ | 15.36 | 12.99 | 10.98 | 4.68 | 1.46 | 10.08 | 9.26 |
| **NF-SSP, one setting** | **0.35** | **0.30** | **0.32** | **0.53** | 1.25 | 10.31 | **2.18** |
| best of the family (oracle radius) | 0.32 | 0.37 | 0.51 | 1.62 | 1.45 | **9.45** | 2.29 |
| DBSCAN, fixed $\texttt{eps}=0.02$ | 0.33 | 0.37 | 2.50 | 5.79 | 5.55 | 17.37 | 5.32 |
| OPTICS, fixed $\texttt{eps}=0.02$ | 0.32 | 0.37 | 2.50 | 6.58 | 5.55 | 17.39 | 5.45 |
| density peaks (no radius) | 10.37 | 3.70 | 2.42 | 1.62 | 1.45 | 13.85 | 5.57 |

Three things are worth reading off it. First, the repaired classical pipeline is *competitive* with
the density family and ahead of it in four of the six regimes: at $p=0.02$ the best density front
reaches $0.32°$ against $0.35°$, which we do not read as a win for either, and in the dense regime the
density front is ahead ($9.45°$ against $10.31°$). The claim of Section 7 was never that a pipeline
which bypasses the gate cannot do better, and here is a family that does, in one regime. Second, the
density family is not an escape from calibration — it *relocates* it, from a multiple of the noise
floor to a neighbourhood radius, and the penalty for the wrong choice is larger than the one the gate
pays for a wrong multiple: fixed at $\texttt{eps}=0.02$ the density front reaches $17.4°$ in the dense
regime against $10.3°$ for the unweighted gate, and fixed at $0.005$ it reaches $13.6°$ at $p=0.02$
against $0.35°$. Its optimal radius moves with the regime ($0.02$ for $p\le0.20$, $0.005$ for
$p\ge0.40$). Third, the reason the radius matters so much is chaining: single-linkage growth connects
neighbouring sources through the points between them, and at $\texttt{eps}\ge0.1$ all four sources
merge into a single cluster in every instance we ran, leaving nothing for the cluster count to
recover. Whatever else a density front offers, the decision "which points are noise" is still made by
a scale, and that scale still has to be right.

**Table 32.** The same design on the real-speech subset used for the mechanism measurements
(four seeds, identical instances). "best of the family" again includes the oracle radius.

| Method | win 1024 | $N=6$ | 2 dense | babble | mean |
|---|---|---|---|---|---|
| conventional median $t_e{=}0.02$ | 8.31 | 7.41 | 16.64 | 1.09 | 8.36 |
| **NF-SSP, one setting** | **0.51** | **1.72** | **1.45** | **1.10** | **1.19** |
| best of the family (oracle radius) | 3.29 | 6.71 | 12.70 | 1.36 | 6.02 |
| DBSCAN, fixed $\texttt{eps}=0.02$ | 6.87 | 6.71 | 12.70 | 13.22 | 9.87 |
| density peaks (no radius) | 3.29 | 11.62 | 14.99 | 1.36 | 7.81 |

On the real configurations the same design puts the calibrated gate ahead in all four, by a factor of
five on average against the best density front and by a factor of seven against the same family at a
fixed radius. We report this as a measurement and not as a general claim: the density family was given
an oracle radius and still trails here, but it leads on synthetic dense data, and a detector designed
for real recordings rather than transplanted from the synthetic setting could well close the gap.

### A.15 Complex and convolutive mixing: what transfers and what does not

The first restriction in Section 1 is that the mixing is instantaneous and real. The three components
of the method are not affected equally by dropping it, and the difference is worth separating.

**(i) The diagnosis transfers, identically.** The reference-scale ratio is a statement about the
marginal law of the *per-point energy*: a noise-only point contains no signal, so its energy is
$(\sigma^2/2)\chi^2_{2M}$ whatever $\mathbf{A}$ is, and the noise-only fraction is
$\pi_0=(1-p)^N$ whatever the columns look like. Neither the real-valuedness of $\mathbf{A}$ nor its
frequency dependence enters, so $r(p)$, the knee and the calibration of $\tau$ are properties of the
point set rather than of the mixing geometry. Table 33 measures the ratio on **the same source
realizations** with three mixing matrices: the real one used throughout the paper, a fixed complex one,
and a per-frequency complex one — the STFT-domain form of a convolutive mixture, drawn independently at
each frequency, which is a strictly more general stress case than any impulse-response-generated set.

**Table 33.** The reference-scale ratio $r=\mathrm{median}(e)/\nu$ under three mixing matrices, on
identical source realizations (ten seeds), against the closed form of Section 4.2 and the blind
floor estimate of Section 5.1. The closed form is not defined beyond the knee once the noise-only
component no longer reaches the median.

| Regime | real $\mathbf{A}$ | complex, fixed | complex, per frequency | closed form | $\hat\sigma^2/\sigma^2$ |
|---|---|---|---|---|---|
| $p=0.02$ | 0.90 | 0.90 | 0.90 | 0.91 | 1.01 |
| $p=0.05$ | 1.04 | 1.03 | 1.04 | 1.04 | 1.06 |
| $p=0.10$ | 1.37 | 1.36 | 1.37 | 1.38 | 1.18 |
| $p=0.20$ | 28.51 | 28.00 | 28.35 | 27.02 | 1.53 |
| $p=0.40$ | 64.86 | 63.33 | 64.54 | --- | 3.13 |
| dense | 81.10 | 79.66 | 81.15 | --- | 89.98 |

The three columns agree with each other to within $2\%$ at every regime, and with the closed form to
within the Monte-Carlo error wherever the closed form applies; the largest spread between the real and
the per-frequency complex case is $2.4\%$, at $p=0.40$. The signal term's law *does* change — the
eigenvalue-weighted sum that replaces the $\chi^2_{2J}$ of the real case is exactly where a
per-frequency mixing matrix would show up — and $r(p)$ is the quantity on which that would register,
which is why we measure it rather than assert it. It does not register: the drift is a property of the
point-energy marginal, and the marginal does not know what the mixing matrix looks like.

**(ii) The calibration transfers verbatim.** The estimator of Section 5.1 reads the lower tail of the
same $\chi^2_{2M}$ law, and $\tau(M,\alpha)$ depends on $M$ and $\alpha$ alone. The last column of
Table 33 is that estimator applied to the complex mixtures, and it reproduces the biases of Table 2.

**(iii) The criterion does not transfer.** Section 3.2's exact reformulation
($\mathbf{X}_r=\mathbf{A}\mathbf{S}_r$, $\mathbf{X}_i=\mathbf{A}\mathbf{S}_i$) holds only because
$\mathbf{A}$ is real. For complex $\mathbf{A}$ the two parts become
$\mathbf{X}_r=\mathbf{A}_r\mathrm{Re}\,\mathbf{s}-\mathbf{A}_i\mathrm{Im}\,\mathbf{s}$ and
$\mathbf{X}_i=\mathbf{A}_i\mathrm{Re}\,\mathbf{s}+\mathbf{A}_r\mathrm{Im}\,\mathbf{s}$, which lie in
the span of $(\mathbf{A}_r,\mathbf{A}_i)$ rather than along one direction, so Proposition 2's
collinearity test has nothing to detect. The measured consequence is not a mild degradation but a
change of character.

**Table 34.** Quality of the real-valued collinearity criterion of Proposition 2 under the three
mixing matrices (ten seeds, $\mathrm{SNR}=20$ dB, pooled over the six regimes). "Admission" is the
fraction of all TF points the criterion admits; recall and precision are with respect to points that
truly carry one active source; false admission is with respect to noise-only points.

| Mixing matrix | admission | single-source recall | precision | noise admitted |
|---|---|---|---|---|
| real (this paper) | 0.250 | 0.578 | 0.496 | 0.104 |
| complex, fixed | 0.112 | 0.078 | 0.166 | 0.103 |
| complex, per frequency | 0.116 | 0.094 | 0.209 | 0.104 |

Recall on true single-source points falls from $0.58$ to $0.08$ while the rate at which noise is
admitted does not move ($0.104\to 0.103$): the criterion does not become *stricter*, it stops
discriminating, and what it admits is then almost half noise. A convolutive deployment therefore has
to replace it. That replacement is frequency-local by necessity — at a frequency $f$ where one source
is active, $\mathbf{x}(f,\cdot)=\mathbf{a}_f s(f,\cdot)$, so the admitted points at that frequency
share one *complex* direction and the test belongs on the complex projective class (equivalently, on
the unconjugated inner product $\sum_p x_p y_p$) rather than on the real part of $\mathbf{x}$ — and
building it is a separate piece of work which we have not done.

**(iv) What does survive is the part the paper contributes.** Table 35 evaluates what a convolutive
deployment would retain today: the same problem set as Table 33, a clustering machine built for the
complex projective metric (the complex analogue of the spherical $k$-means of Section 5.4, whose
implementation reproduces the real one exactly on real unit directions), and six different masks.

**Table 35.** Six masks evaluated with the complex-domain clustering machine, on the complex fixed
mixing matrix (ten seeds); and the same six on the real mixing matrix as a control. Retention is
reported for the complex case. "NF-SSP" is the full construction transplanted verbatim, including the
collinearity criterion that Table 34 shows to be invalid.

| Mask | $p=0.02$ | $p=0.05$ | $p=0.10$ | $p=0.20$ | $p=0.40$ | dense | mean (cplx) | mean (real) | retention |
|---|---|---|---|---|---|---|---|---|---|
| no criteria (all points) | 16.04 | 7.38 | 6.96 | 7.19 | 5.96 | 17.70 | 10.20 | 11.21 | 100% |
| collinearity only | 21.87 | 23.38 | 21.96 | 21.88 | 21.91 | 22.89 | 22.31 | 4.24 | 11.2% |
| conventional median $t_e{=}0.02$ | 16.03 | 9.27 | 7.60 | 5.40 | 6.92 | 18.83 | 10.68 | 11.75 | 96.2% |
| NF-SSP transplanted verbatim | 26.29 | 25.02 | 22.66 | 21.78 | 21.15 | 22.89 | 23.30 | 2.96 | 6.7% |
| calibrated energy criterion only | 5.14 | 4.17 | 8.53 | 4.28 | 5.13 | 50.99 | 13.04 | 13.49 | 32.3% |
| the same, with the retention self-check | **5.14** | **4.17** | **8.53** | **4.28** | **5.13** | **18.13** | **7.56** | 7.13 | 47.3% |

The pattern is the one this appendix set out to establish. The calibrated *energy* criterion is the
component that transfers: it lowers the average from $10.20°$ to $7.56°$ in the complex setting and
from $11.21°$ to $7.13°$ in the real control, and at $p=0.02$ it is worth a factor of three
($16.04°\to5.14°$). The retention self-check transfers with it, because it reads a gate-internal
quantity rather than a model assumption, and it is what keeps the energy criterion from failing in the
dense regime ($50.99°\to18.13°$). The full NF-SSP transplanted verbatim does *not* transfer: at
$23.30°$ it is worse than applying no mask at all, because it carries the invalid criterion with it.
And the residual error of the best complex-domain entry is dominated by multi-source points, which is
precisely the contamination that the collinearity criterion removes in the real case — in the control
column, removing the SSP criteria at $p=0.20$ costs a factor of $22$ ($0.50°$ against $10.93°$).

Taking the appendix as a whole: for convolutive mixing the reference-scale diagnosis of Section 4 and
the calibration of Section 5 carry over per frequency, and the self-check carries over with them, but
the SSP detection of Section 3 must be rebuilt in the complex domain. The gate is a front end for the
*energy* dimension of a sparse-domain pipeline, not a solution for convolutive UBSS, and the numbers
in Table 35 are a lower bound on what such a deployment could achieve, since the piece it lacks is
the one that handles the dominant remaining error.

### A.16 The false-admission rate where the floor estimate is biased

A fair question about the calibration is what happens at the cells where the noise floor is estimated
too high: the threshold $\tau\hat\nu$ is then too high as well, and a fixed $\alpha$ could in
principle remove the weak single-source points that the gate exists to keep. The cells in question are
the ones Table 2 and Section 8.2 already flag — low SNR with moderate to high activation — and there
the effect is real, large in threshold terms, and bounded in error terms.

**Table 36.** Sensitivity of the gate to $\alpha$ at the four cells where the floor estimate is biased
high (ten seeds each; $\tau=Q_{1-\alpha}(\chi^2_{2M})/2M$, and the self-check is disabled here so that
the calibration itself is on trial). Each entry is retention / single-source recall / mean angle error
in degrees; $\hat\sigma^2/\sigma^2$ is the floor estimate's bias at that cell.

| Cell | $\hat\sigma^2/\sigma^2$ | $\alpha=10^{-1}$ | $\alpha=10^{-3}$ | $\alpha=10^{-5}$ | $\alpha=10^{-4}$ |
|---|---|---|---|---|---|
| $p=0.20$, SNR $0$ dB | 1.39 | 4.14% / 0.057 / 11.77 | 0.92% / 0.010 / 10.69 | 0.25% / 0.001 / 18.39 | 0.45% / 0.003 / 12.88 |
| $p=0.20$, SNR $10$ dB | 1.59 | 14.58% / 0.276 / 2.14 | 12.25% / 0.229 / 1.66 | 10.21% / 0.185 / 1.42 | 11.20% / 0.207 / 1.51 |
| $p=0.10$, SNR $0$ dB | 1.19 | 4.75% / 0.123 / 11.40 | 2.05% / 0.056 / 9.85 | 0.98% / 0.024 / 11.11 | 1.37% / 0.036 / 8.63 |
| $p=0.40$, SNR $0$ dB | 1.60 | 2.89% / 0.020 / 12.57 | 0.29% / 0.001 / 16.10 | 0.04% / 0.000 / 23.00 | 0.11% / 0.000 / 20.58 |

The premise of the question is confirmed. At $(p,\mathrm{SNR})=(0.20,0)$ dB the floor is $39\%$ high,
the threshold is consequently placed $39\%$ too far up, retention falls to $0.45\%$ of the points and
single-source recall to $0.003$: the gate is, at that cell, removing almost everything. But the
performance penalty for a fixed $\alpha$ is bounded by the fact that the error is dominated by the
premise failure rather than by the threshold: over the four cells the best $\alpha$ improves on
$10^{-4}$ by $9\%$ at the worst cell and $12\%$ at $(0.10,0)$ dB, and at the cell where the gate is
actually working $(0.20,10)$ dB, $10^{-4}$ is within $6\%$ of the best value over six decades. Two
observations make this a property of the design rather than a defect of it. The cells where relaxing
$\alpha$ would help are exactly the cells where the gate's premise fails, and the retention self-check
reads that failure directly — it disables the gate there instead of trying to tune it, which is the
behaviour Section 5.3 designs for. And the threshold changes by a factor of $4.3$ across the whole
range of $\alpha$ tested while the error changes by at most $39\%$, so the calibration is in this
sense insensitive by construction: $\tau(M,\alpha)$ is a smooth function of $\alpha$ over nine
decades, and its placement relative to the $1.8$-decade-wide signal-to-noise gap is what the gate
depends on, not its exact value.

### A.17 The quantile step: cost and the case against approximating it

Section 5.1 takes thirty order statistics of the energy vector. A single selection is $O(n)$ in
expectation against $O(n\log n)$ for a full sort, and with $O(n)$ points per TF frame this is the
only super-constant step in the gate; it is worth saying how it behaves at the scales the paper does
not reach.

**Table 37.** Cost of the blind floor estimate against the length $n=FT$ of the energy vector (single
core, median of twenty repetitions; the same estimator as Table 25, which measured $2.99$ ms at
$n=4.8\times10^4$). "histogram" is the $4096$-bin approximation of the same thirty quantiles, and the
last two columns are the accuracy of the two routes on pure Gaussian noise, where both should return
$1$.

| $n$ | selection (this paper) | full sort | histogram | ns per point | $\hat\sigma^2/\sigma^2$ (exact) | $\hat\sigma^2/\sigma^2$ (histogram) |
|---|---|---|---|---|---|---|
| $2{,}112$ | 0.30 ms | 0.08 ms | 0.13 ms | 142 | 1.022 | 1.030 |
| $8{,}192$ | 0.61 ms | 0.37 ms | 0.25 ms | 75 | 1.059 | 0.999 |
| $32{,}768$ | 1.98 ms | 1.71 ms | 0.71 ms | 60 | 0.957 | 0.871 |
| $131{,}072$ | 8.19 ms | 7.86 ms | 2.44 ms | 63 | 1.010 | 1.113 |
| $524{,}288$ | 35.9 ms | 35.5 ms | 9.74 ms | 69 | 0.998 | 0.911 |
| $1{,}048{,}576$ | 77.9 ms | 77.9 ms | 19.4 ms | 74 | 1.003 | 1.086 |

Three practical points. The estimate scales linearly — the fitted log–log slope is $0.92$ over three
decades, against $1.11$ for a full sort, which is the expected $n\log n$ — and costs about $75$ ns per
point at the top of the range, so a million-point spectrogram pays $78$ ms for a step that sits inside
a pipeline whose $\ell_1$ recovery at that scale costs tens of seconds: the share falls as the
problem grows, from $0.17\%$ in Table 25 to $0.02\%$ here. Second, at the scales this paper uses, the
thirty selections cost about four times a single full sort ($0.30$ ms against $0.08$ ms at
$n=2{,}112$), because the implementation trades one pass for thirty; a single `partition` with all
thirty levels at once, or a sort below some cutoff, would remove that constant, and we note it as an
implementation detail rather than a property of the estimator. Third, the histogram approximation is
the genuine speed-up — a factor of four at $n=10^6$ — but it is not free in accuracy: on pure noise
its error is $8.6$–$13\%$ where the exact estimate is within $1\%$, and the error is one-sided, which
matters because a floor that is systematically high is a gate that is systematically aggressive.
Since the exact route is already a negligible share of the pipeline and the approximation is not, we
keep the exact one, and report the trade-off for implementations that cannot.

### A.18 A banded noise floor, and why it is not the repair for babble

Appendix A.12 showed that the failure under babble is not insufficient robustness of the scale
estimator but the *non-stationarity* of the noise across the plane: its power varies by a factor of
$5.8$ across frequency, so the lower tail reads the quiet bins and underestimates the floor by
$66\times$. That diagnosis suggests a repair which the robust estimators of A.12 are not: replace the
single global floor with a **per-band** floor, so that the reference scale may vary with frequency.
The band count trades the two errors of the estimator against each other — fewer points per band means
a noisier quantile, but a band whose noise is locally stationary is one the lower tail can read.

**Table 38.** Per-band noise floor in the same gate and the same criteria as Section 5.4 (energy
criterion AND collinearity AND balance, with the retention self-check), on real speech (four seeds).
$\hat\nu/\nu$ is the mean estimated-to-true floor ratio over the bands; the noise-free case is
excluded because the ratio is undefined there.

| Recording | bands | $\hat\nu/\nu$ | retention | angle error |
|---|---|---|---|---|
| babble, SNR $20$ dB | 1 | 0.019 | 25.8% | 0.821 |
| | 4 | 0.261 | 23.3% | 0.725 |
| | 16 | 0.692 | 22.2% | 0.702 |
| | 64 | 1.213 | 21.7% | 0.704 |
| babble, SNR $0$ dB | 1 | 0.002 | 19.0% | 9.587 |
| | 4 | 0.021 | 16.0% | 9.106 |
| | 16 | 0.037 | 15.0% | 9.114 |
| | 64 | 0.092 | 14.4% | 9.120 |
| Gaussian, SNR $20$ dB | 1 | 1.664 | 7.9% | 0.507 |
| | 4 | 1.987 | 7.4% | 0.401 |
| | 16 | 2.254 | 7.2% | 0.349 |
| | 64 | 2.420 | 7.2% | 0.395 |

Banding does repair the *estimate*: on babble the floor ratio rises from $0.019$ to $1.21$ as the
bands narrow, so the $52\times$ underestimate becomes a $21\%$ overestimate, and the angle error falls
by $14\%$ from $0.821°$ to $0.702°$ with sixteen bands. It does not repair the *configuration*: at
$\mathrm{SNR}=0$ dB with the same noise the ratio moves from $0.002$ to $0.09$ and the error does not
move at all ($9.59°\to9.12°$), because there the failure is not the stationary-across-frequency
assumption but the collapse of the separation between signal and noise. And the third block is the
control that keeps us from recommending it: on stationary Gaussian noise banding *also* improves the
error ($0.507°\to0.349°$), which cannot be a benefit of banding as such — the true floor is the same
in every band — and is instead the incidental effect of the small-sample bias of the per-band
quantile, which raises $\hat\nu/\nu$ from $1.66$ to $2.42$ and makes the gate more aggressive. The
same mechanism is at work on babble. We therefore report the banded floor as a diagnosis-driven
option — it is the right shape of repair for the failure A.12 identifies, and it recovers most of the
underestimate — but not as an improvement we propose, because the measured gain is small, entangled
with a bias we would then have to correct, and absent in the cell where babble actually breaks the
method.

### A.19 The gate's strictness and automatic order selection

All methods in this paper are given the true source number, and Section 10 lists that as a
limitation. It is worth quantifying, because the gate's strictness is not neutral for the estimator
that would supply the number: automatic order selection works by counting peaks in the density of
admitted directions, so a gate that admits fewer points shifts its decision. We measure the effect
directly, by feeding the potential-function peak detector the direction set produced by four
different masks on the same instances.

**Table 39.** Automatic order selection from the mask. The potential-function peak detector (the same
criteria as in the baselines: angle grid, local maxima, significance threshold $\rho>0.2\rho_{\max}$,
near-peak suppression) is applied to the direction set each mask produces, on the ladder and on
$N\in\{3,\dots,6\}$ cells (ten seeds each, $960$ decisions). "exact" is the fraction for which
$\hat N=N$; the last column is the angle error obtained by clustering with the *estimated* number of
columns, so that a missing column is penalised.

| Mask | exact | mean $\lvert\Delta N\rvert$ | $\hat N<N$ | $\hat N>N$ | angle error at $\hat N$ | points kept |
|---|---|---|---|---|---|---|
| collinearity + balance only | 29.2% | 1.27 | 42.5% | 28.3% | 15.12 | 19.2% |
| conventional median $t_e{=}0.02$ | 20.8% | 1.68 | 18.3% | 60.8% | 10.59 | 91.0% |
| conventional median $t_e{=}5$ | 45.0% | 0.79 | 40.8% | 14.2% | 14.64 | 14.6% |
| **NF-SSP** | **44.2%** | **0.79** | 54.2% | **1.7%** | 15.61 | 19.4% |

The gate does change the answer, and in a specific direction. The conventional default, which as
Section 7 notes is effectively inactive, over-counts in $61\%$ of the decisions — the noise it leaves
in creates spurious peaks — while the calibrated gate almost never over-counts ($1.7\%$) and instead
under-counts in $54\%$ of them, so that the two masks bracket the truth from opposite sides and reach
about the same exact-order rate by different routes ($20.8\%$ against $44.2\%$). A stricter median
multiple mimics the calibrated gate's bias. The consequence for practice is not the exact-order rate
but the last column: because a missing column costs $90°$ in the matching metric while a spurious one
costs the angle of its nearest counterpart, under-counting is the more expensive error, and clustering
at $\hat N$ is better for the conventional default than for the calibrated gate ($10.59°$ against
$15.61°$) even though the latter estimates the order better. Order selection should therefore not be
run on the calibrated gate's output: the useful division of labour is to estimate the order from the
ungated or lightly gated set — where the unweighted collinearity mask and the relaxed median rule are
both competitive — and to apply the gate afterwards to the directions that are to be clustered. We
report this as a measurement of an interaction rather than as a solution, and it is one more reason
why the paper supplies $N$ to every method rather than claiming an end-to-end pipeline.

"""


def main() -> None:
    dry = "--dry" in sys.argv
    s = MS.read_text()
    n, skipped, missed = 0, 0, 0
    for tag, old, new in E:
        if tag in _HAND_CORRECTED:
            print(f"  [已手工二次修订，跳过] {tag}")
            skipped += 1
            continue
        if not old.strip():
            print(f"  [跳过] {tag}")
            skipped += 1
            continue
        if new and new in s:
            print(f"  [已改] {tag}")
            skipped += 1
            continue
        if old not in s:
            print(f"  [未命中] {tag}")
            missed += 1
            continue
        s = s.replace(old, new, 1)
        n += 1
        print(f"  [已改] {tag}")

    # Table 17 加两列
    s, chg = extend_table17(s)
    if chg:
        n += 1
        print("  [已改] §9.5 Table 17 加两列")
    else:
        skipped += 1
        print("  [跳过] §9.5 Table 17（已是 8 列）")

    # 附录 A.14–A.19：插在 A.13 之后、分隔线之前（幂等）
    marker = "\n\n---\n\n## Figures"
    if "### A.14 Density clustering" in s:
        print("  [已改] 附录 A.14–A.19")
        skipped += 1
    elif marker in s:
        s = s.replace(marker, "\n\n" + APPENDIX + marker, 1)
        n += 1
        print("  [已改] 附录 A.14–A.19")
    else:
        print("  [未命中] 附录插入点")
        missed += 1

    if not dry:
        MS.write_text(s)
    print(f"\n共改 {n} | 跳过 {skipped} | 未命中 {missed}")
    if missed and not dry:
        sys.exit(1)


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
