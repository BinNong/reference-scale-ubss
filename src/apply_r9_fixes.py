#!/usr/bin/env python3
"""第九轮改稿：CSSP 审稿意见（Minor Revision）的**措辞类**修订。

本脚本**不含任何数字改动**——表格数字与含数字的句子由 `apply_r9b_realnum.py` 处理，
因为那些要等真实语音 oracle 重跑（held-out 调参种子）完成。因此本脚本可以在重跑之前先落地。

覆盖意见的以下条目（编号为 `paper/cssp_review_triage.md` 中的编号）：

  · 3.2  "maximum-likelihood reliability" -> noise-normalised energy reliability（2 处）
  · 3.3  false-admission-calibrated -> 标明是 noise-only null 下的 *nominal* 率（3 处）
  · 3.4  标题 self-calibrating -> noise-floor-referenced
  · 3.6  "the optimal gate" -> "the best-observed gate"（3 处）
  · 4.1  Proposition 2 把非退化条件写进命题陈述
  · 4.2  Assumption A2 用协方差记号写全
  · 4.4  runtime 的 $2.6\\%$ 精度 -> essentially the same cost（2 处）
  · 4.5  §2 的绝对化表述 -> 限定到"本文考察的实现"
  · 4.6  §10.5 显式声明该研究是诊断、不是端到端验证
  · 3.5  §9.3 澄清三个 opt 列是 *energy-only* oracle（与合成梯的 joint oracle 不同）

用法：
    python3 src/apply_r9_fixes.py --dry
    python3 src/apply_r9_fixes.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

# (标签, 原文, 新文)。原文必须在稿中**恰好出现一次**。
E: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- 3.4 标题
    (
        "标题：self-calibrating -> noise-floor-referenced",
        r"# The Reference Scale of Energy Thresholding in Sparse-Domain Blind Source Separation: A Diagnosis and a Blind Self-Calibrating Gate",
        r"# The Reference Scale of Energy Thresholding in Sparse-Domain Blind Source Separation: A Diagnosis and a Noise-Floor-Referenced Gate",
    ),
    # -------------------------------------------------------- 4.5 §2 绝对化表述
    (
        "§2：Every implementation -> 本文考察的实现",
        r"Every implementation of the above includes an energy or magnitude threshold, and its necessity is universally acknowledged.",
        r"The implementations considered here all include an energy or magnitude threshold, and its necessity is widely acknowledged.",
    ),
    (
        "§2：none of these works -> to the best of our knowledge",
        r"Neither family estimates a noise floor, and none of these works calibrates the level to a target probability.",
        r"Neither family estimates a noise floor, and, to the best of our knowledge, none of the works cited above calibrates the level to a target probability.",
    ),
    # ------------------------------------------------------------- 4.2 A2 记法
    (
        "A2：用协方差记号写全",
        r"**A2 (isotropic TF-domain noise).** The noise has independent, identically distributed real and imaginary parts, both zero-mean and isotropic. This holds exactly for circularly symmetric complex Gaussian noise, the standard TF-domain model.",
        r"**A2 (isotropic TF-domain noise).** Writing $\mathbf{n}=\mathbf{n}_r+j\mathbf{n}_i$, its real and imaginary parts are independent, zero-mean and isotropic, with $\operatorname{Cov}(\mathbf{n}_r)=\operatorname{Cov}(\mathbf{n}_i)=\frac{\sigma^2}{2}\mathbf{I}_M$ and $\mathbf{n}_r\perp\mathbf{n}_i$. Circularly symmetric complex Gaussian noise realises the assumption exactly and is the standard TF-domain model; it is this covariance structure that gives a noise-only point the $\chi^2_{2M}$ law of Section 5.1.",
    ),
    # --------------------------------------------------- 4.1 Proposition 2 条件
    (
        "Prop 2：非退化条件进命题陈述",
        r"**Proposition 2 (exactness without noise).** At a TF point where exactly one source is active, $|\cos(\mathbf{X}_r,\mathbf{X}_i)|=1$. *Proof.* With $\mathbf{s}=s_n\mathbf{e}_n$, Section 3.2 gives $\mathbf{X}_r=\mathrm{Re}(s_n)\mathbf{a}_n$ and $\mathbf{X}_i=\mathrm{Im}(s_n)\mathbf{a}_n$, both multiples of $\mathbf{a}_n$, so the cosine is $\pm1$ whenever $\mathrm{Re}(s_n)\mathrm{Im}(s_n)\neq0$. $\square$",
        r"**Proposition 2 (exactness without noise).** At a TF point where exactly one source is active and $\mathrm{Re}(s_n)\mathrm{Im}(s_n)\neq0$, $|\cos(\mathbf{X}_r,\mathbf{X}_i)|=1$. *Proof.* With $\mathbf{s}=s_n\mathbf{e}_n$, Section 3.2 gives $\mathbf{X}_r=\mathrm{Re}(s_n)\mathbf{a}_n$ and $\mathbf{X}_i=\mathrm{Im}(s_n)\mathbf{a}_n$, both multiples of $\mathbf{a}_n$, so the cosine is $\pm1$; the non-degeneracy condition is what keeps it defined, since $\mathbf{X}_i$ vanishes when $s_n$ is real. $\square$",
    ),
    # ------------------------------------------------- 3.3 贡献 2 的 nominal 限定
    (
        "贡献 2：false-admission-calibrated -> null-calibrated（+ nominal）",
        r"**2. A blind, false-admission-calibrated gate.** We reference the gate to the noise floor, estimate the floor blindly from the noise-dominated lower tail of the energy distribution using the chi-square quantile relation $s^2 = 2e_{(q)}/Q_q(\chi^2_{2M})$, which is exact at a noise-only point and is applied as a lower-tail approximation, and calibrate the threshold to a false-admission rate $\alpha$, so that a single constant",
        r"**2. A blind, null-calibrated gate.** We reference the gate to the noise floor, estimate the floor blindly from the noise-dominated lower tail of the energy distribution using the chi-square quantile relation $s^2 = 2e_{(q)}/Q_q(\chi^2_{2M})$, which is exact at a noise-only point and is applied as a lower-tail approximation, and calibrate the threshold to a *nominal* false-admission rate $\alpha$ under the noise-only null — nominal because the rate is exact only when the true floor is known, whereas the algorithm inserts the lower-tail estimate $\hat\nu$ — so that a single constant",
    ),
    # ------------------------------------------------------- 3.3 Algorithm 1 标题
    (
        # 注意：Algorithm 的题注由 `build_pdf.algorithm_block()` **整行直出**成 LaTeX
        # （`\textbf{Algorithm N}\enspace {title}`），不经过 pandoc，因此**不能用 markdown 强调**——
        # 星号会原样出现在成品里（`build_pdf` 的成品校验会报"字面量 *强调*"，实测抓到过一次）。
        # 正文段落里的强调不受此限（贡献 2 的 *nominal* 就是正常的）。
        "Algorithm 1：标题加 nominal 限定（题注不用 markdown 强调）",
        r"**Algorithm 1** NF-SSP gate: noise-floor-referenced, false-admission-calibrated single-source-point selection.",
        r"**Algorithm 1** NF-SSP gate: noise-floor-referenced, null-calibrated single-source-point selection. The threshold is calibrated to a nominal false-admission rate under the noise-only null; the realised rate differs from it by the error of the lower-tail floor estimate.",
    ),
    # --------------------------------------------------------- 3.3 §12 nominal
    (
        "§12：calibrating the multiple to a false-admission rate -> nominal",
        r"calibrating the multiple to a false-admission rate yields a rule with one meaningful constant",
        r"calibrating the multiple to a nominal false-admission rate under the noise-only null yields a rule with one meaningful constant",
    ),
    # --------------------------------------------------- 3.2 §9.5 ML 措辞（一）
    (
        "§9.5：maximum-likelihood reliability -> noise-normalised energy reliability",
        r"The weight is the maximum-likelihood reliability of an admitted direction under Gaussian noise, $w=\lVert\mathbf{x}\rVert^2/\hat\nu$, normalised by the same noise floor the gate is referenced to,",
        r"The weight is a noise-normalised energy reliability, $w=\lVert\mathbf{x}\rVert^2/\hat\nu$: the energy of an admitted point expressed in units of the noise floor, which is the natural reliability proxy when the only modelled impairment is additive noise. It is normalised by the same noise floor the gate is referenced to,",
    ),
    (
        "§9.5：maximum-likelihood operation -> natural operation",
        r"suggests a better device than a hard gate: if a point's reliability is a smooth function of its energy, the maximum-likelihood operation is to weight each direction by that reliability rather than to discard points.",
        r"suggests a better device than a hard gate: if a point's reliability is a smooth function of its energy, the natural response is to weight each direction by that reliability rather than to discard points.",
    ),
    # ------------------------------------------------------------ 3.6 optimal gate
    (
        "§9.5 小节标题：optimal -> best-observed",
        r"### 9.5 Why the optimal gate is more aggressive on real signals",
        r"### 9.5 Why the best-observed gate is more aggressive on real signals",
    ),
    (
        "贡献 5：the optimal gate -> the best gate on the tested grid",
        r"so the optimal gate is more aggressive than any false-admission calibration can be",
        r"so the best gate on the tested grid is more aggressive than any false-admission calibration can be",
    ),
    (
        "摘要：the optimal gate is set by -> the best gate on the tested grid is set by",
        r"and on real speech the optimal gate is set by the energy-dependence of point reliability",
        r"and on real speech the best gate on the tested grid is set by the energy-dependence of point reliability",
    ),
    (
        # 摘要受 CSSP 的 250 词上限约束（本行所在的整段最终为 250 词），故这里只加 "nominal"，
        # 不写 "under the noise-only null"（那 5 个词留在贡献 2 与 Algorithm 题注里）。
        "摘要：calibrate ... to a false-admission rate -> nominal",
        r"and calibrate the multiple to a false-admission rate — a threshold *derived*, not selected.",
        r"and calibrate the multiple to a nominal false-admission rate — a threshold *derived*, not selected.",
    ),
    (
        "摘要：Two limits bound this -> Two limits（压词）",
        r"Two limits bound this: the collinearity test needs a real, delay-free mixing matrix",
        r"Two limits: the collinearity test needs a real, delay-free mixing matrix",
    ),
    (
        "摘要：one in sixteen -> 1 in 16（压词）",
        r"at a per-instance error rate near one in sixteen",
        r"at a per-instance error rate near 1 in 16",
    ),
    (
        "摘要：删 therefore（压到 CSSP 的 250 词以内）",
        r"and the calibrated gate is therefore paired with a derived reliability weight",
        r"and the calibrated gate is paired with a derived reliability weight",
    ),
    # ------------------------------------------------------------- 4.4 runtime 精度
    (
        "§8.1：2.6% faster -> essentially the same cost",
        r"The gate adds negligible computational overhead: the estimator is $2.6\%$ *faster* than the conventional pipeline it replaces and $7.7\times$ faster than the smoothed-$\ell_0$ baseline, and the gate's own stages account for $0.66\%$ of the runtime (Table 27).",
        r"The gate adds negligible computational overhead: the estimator runs at essentially the same cost as the conventional pipeline it replaces ($0.184$ s against $0.189$ s, a difference well inside the run-to-run timing variance) and $7.7\times$ faster than the smoothed-$\ell_0$ baseline, and the gate's own stages account for $0.66\%$ of the runtime (Table 27).",
    ),
    (
        "§7：runtime 精度改为 essentially the same",
        r"it runs in $0.184$ s per problem on one CPU core, against $0.189$ s for the conventional pipeline and $1.41$ s for the smoothed-$\ell_0$ baseline (Table 7)",
        r"it runs in $0.184$ s per problem on one CPU core — essentially the same as the conventional pipeline ($0.189$ s) and $7.7\times$ faster than the smoothed-$\ell_0$ baseline ($1.41$ s) (Table 7)",
    ),
    # -------------------------------------------------------------- 4.6 §10.5 定位
    (
        "§10.5：显式声明是诊断而非端到端验证",
        "noise — a detection problem, not a calibration one.\n\n## 11. Discussion and Limitations",
        "noise — a detection problem, not a calibration one. We state the scope of this study plainly:\n"
        "it validates the failure analysis of Section 10.1 on recorded noise and diagnoses the modelling\n"
        "component a reverberant setting would require; it is not an end-to-end validation of the proposed\n"
        "SSP detector on reverberant mixtures, which would need a controlled reverberation time.\n\n"
        "## 11. Discussion and Limitations",
    ),
    # --------------------------------------------- 3.5 §9.3 energy-only oracle 澄清
    (
        "§9.3：澄清三个 opt 列是 energy-only oracle",
        r"With the usual coefficient $t_e=0.02$ relative to the median (Fig. 8a), the two-stage pipeline attains a mean angle error of $8.26°$, against $0.70°$ for the best best-observed reference.",
        r"The three tuned columns vary the energy coefficient only, with the collinearity threshold held at its default $c_0=0.98$; they are therefore an *energy-reference* oracle, narrower than the joint $(c_0,t_e)$ oracle of the synthetic ladder (Section 8.1), which also retunes the detector. With the usual coefficient $t_e=0.02$ relative to the median (Fig. 8a), the two-stage pipeline attains a mean angle error of $8.26°$, against $0.70°$ for the best-observed reference.",
    ),
]


# 同轮的 `apply_r9b_realnum.py`（数字更新）会改动 §9.3 同一段落里的两个数字
# （best-observed 0.70° -> 0.69°、penalty 12.8x -> 12.9x、31x -> 32x），
# 于是本条目的 `new` 不再与稿件逐字匹配。本条的效果（插入 energy-only oracle 的澄清句、
# 顺手修掉 "best best-observed" 的重复词）**已经达成**，故登记跳过而非删除条目。
_SUPERSEDED_R9B = {
    "§9.3：澄清三个 opt 列是 energy-only oracle",
}


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--dry", action="store_true", help="只报告将改动哪些条目")
    a = a.parse_args()

    s = MS.read_text()
    changed = skipped = missing = 0
    for tag, old, new in E:
        if tag in _SUPERSEDED_R9B:
            print(f"  [R9b 已承接] {tag}")
            skipped += 1
            continue
        # 护栏：若 new 仍包含完整的 old，则第一次应用后 old 依然存在 -> 每次运行都会再改一遍，
        # 幂等失效（实测 §10.5 那条踩过：new = old + 追加句）。判据是 new 里不再出现完整 old。
        if old in new:
            print(f"  ✗ 条目设计有误（new 包含 old，幂等会失效）：{tag}")
            sys.exit(1)
        n = s.count(old)
        if n == 1:
            s = s.replace(old, new, 1)
            print(f"  ✓ {tag}")
            changed += 1
            continue
        if n > 1:
            print(f"  ✗ 原文不唯一（{n} 次）：{tag}")
            missing += 1
            continue
        # 精确未命中 -> 回退到「容忍换行/多余空白」的匹配。手稿里有些段落是硬换行的
        # （如 §10.5 那段），跨行的句子用逐字匹配必然落空。
        pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
        hits = list(pat.finditer(s))
        if len(hits) == 1:
            m = hits[0]
            s = s[:m.start()] + new + s[m.end():]
            print(f"  ✓ {tag}  （容忍换行的匹配）")
            changed += 1
            continue
        if new in s:
            print(f"  [已改，跳过] {tag}")
            skipped += 1
            continue
        msg = "原文不存在" if not hits else f"原文不唯一（{len(hits)} 次）"
        print(f"  ✗ 未命中（{msg}）：{tag}")
        missing += 1

    print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing} | 词数 {len(s.split())}")
    if a.dry:
        return
    if missing:
        print("有未命中项，未写入。")
        sys.exit(1)
    if changed == 0:
        print("稿件无变化（已是目标状态）。")
        return
    MS.write_text(s)
    print(f"已写入 {MS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
