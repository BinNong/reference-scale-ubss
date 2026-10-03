"""R8（CSSP 投稿前意见）—— 理论限定、术语收紧、防御性表述精简。

来源：`paper/cssp_prereview_triage.md` 的逐条核查。本轮只处理**不触发表号**的文字改动：

  1. 假定层：新增 **A3（源独立激活）**。原稿 Assumptions 只有 A1（混合矩阵）与 A2（噪声），
     而 Prop 6 的 π_J = C(N,J)p^J(1−p)^{N−J} 与 p* = 1−2^{−1/N} 都依赖它——从未写在假设里。
     审稿意见（§12）要求把理论限定为 independent activation model 下的 model-level diagnosis。
  2. §11 结论里 `The semi-analytic reference-scale ratio locates the drift exactly` 一句越界：
     精确的是 **probability-mass boundary**，不是 r(p) 本身（r(p) 对 J≥2 只是 mean-eigenvalue 近似）。
     稿中 §1 与 Prop 6 的说明本来已分层，只有结论这句没收住。
  3. 术语：
     · `per-configuration bound` → `best-observed reference`（9 处 + Table 15 的定义句）。
       原词是"bound"但实测是"指定网格上的最佳观测值"，不是数学下界。
       未采用意见建议的 `oracle-tuned reference`——稿中已有 `oracle-tuned threshold`
       指合成梯上逐档调参的 oracle，二者会撞名。
     · `self-calibrating` → `false-admission-calibrated`（2 处：贡献列表 + Algorithm 1 题注）。
     · Prop 7 与 §4.3 的 `knee` 正式化为 **probability-mass boundary**，knee 降为俗称。
     · `blind` 在 §5.1 首次出现处限定为 "blind w.r.t. the noise floor and the multiple"。
  4. 防御性表述精简（约 130 词）。**不删七条 limitation 中的任何一条**——
     那是前几轮审稿人明确追问过的内容；只压缩自证式框句。

幂等：再跑一次应报"全部跳过"。
用法：
    python3 src/apply_r8_fixes.py --dry
    python3 src/apply_r8_fixes.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

# 全局改名：先做，且方向可判定，因此幂等（不是 (old,new) 条目，避免重跑报未命中）
RENAME = [("per-configuration bound", "best-observed reference")]

# (标签, 旧文, 新文) —— 下面的 old/new 一律按**改名之后**的文本写
E: list[tuple[str, str, str]] = [

("假定层：新增 A3（源独立激活）",
 "This holds exactly for circularly symmetric complex Gaussian noise, the standard TF-domain model.",

 "This holds exactly for circularly symmetric complex Gaussian noise, the standard TF-domain model. "
 "**A3 (independent source activation).** At each TF point every source is active or inactive independently of the "
 "others with a common probability $p$, so that the number of active sources follows the binomial law "
 "$\\pi_J=\\binom{N}{J}p^{J}(1-p)^{N-J}$. This is the model under which the mixture law of Section 4.2 is derived and "
 "under which $p^\\star=1-2^{-1/N}$ is the probability-mass boundary; it is realised by the generators of Section 7, "
 "and it is the idealisation that the structured-source experiment of Section 8.7 and the real-speech measurements of "
 "Section 9.2 probe."),

("§11：把 exact 归给边界，把 r(p) 归给半解析近似",
 "The semi-analytic reference-scale ratio locates the drift exactly, at the knee $\\pi_0=1/2$ given by "
 "$p^\\star=1-2^{-1/N}$.",

 "Its probability-mass boundary is exact — the median can lie on the noise scale only while the noise-only component "
 "alone carries at least half of the mass, $\\pi_0=(1-p)^N\\ge1/2$, that is $p^\\star=1-2^{-1/N}$ — whereas the ratio "
 "itself is given by the semi-analytic form of Section 4.2, which is exact at noise-only and single-source points and "
 "a mean-eigenvalue approximation beyond them, agreeing with simulation to within $1.5\\%$ wherever its expansion "
 "applies."),

("Prop 7 名称正式化（knee 降为俗称）",
 "**Proposition 7 (mass boundary, the knee).**",
 "**Proposition 7 (probability-mass boundary; referred to below as the knee).**"),

("§4.3 标题正式化",
 "### 4.3 The knee, and why a fixed median-relative threshold cannot work",
 "### 4.3 The probability-mass boundary, and why a fixed median-relative threshold cannot work"),

("贡献 2：self-calibrating → false-admission-calibrated",
 "**2. A blind, self-calibrating gate.**",
 "**2. A blind, false-admission-calibrated gate.**"),

("贡献 2 末尾：限定 blind 的所指",
 "so that the rule requires no per-condition tuning (Section 8.6).",
 "so that the rule requires no per-condition tuning (Section 8.6). \"Blind\" here qualifies the noise floor and the "
 "multiple only: like every method compared in this paper, the rule is given the source number $N$."),

("Algorithm 1 题注：self-calibrating → false-admission-calibrated",
 "**Algorithm 1** NF-SSP gate: noise-floor-referenced, self-calibrating single-source-point selection.",
 "**Algorithm 1** NF-SSP gate: noise-floor-referenced, false-admission-calibrated single-source-point selection."),

("§5.1：首次出现处限定 blind",
 "### 5.1 Blind estimation of the noise floor",
 "### 5.1 Blind estimation of the noise floor\n\n"
 "The word *blind* in this section qualifies the noise floor and the multiple alone. No method in this paper estimates "
 "the source number $N$ — every pipeline compared here receives it — and what is estimated below is the scale against "
 "which the gate is set, not the order of the mixture."),

("Table 15 定义句：点明是观测值而非数学界",
 "Wherever we quote a \"best-observed reference\" we mean the smallest angle error attained in that configuration by "
 "the three tuned columns (median-opt, max-opt, top-K-opt), whose mean is $0.703°$; taken over all seven competing "
 "columns it would be $0.681°$.",

 "Wherever we quote a \"best-observed reference\" we mean the smallest angle error attained in that configuration by "
 "the three tuned columns (median-opt, max-opt, top-K-opt), whose mean is $0.703°$; taken over all seven competing "
 "columns it would be $0.681°$. It is the best value observed on a specified grid rather than a mathematical bound: "
 "nothing prevents a rule outside that grid from doing better, and the quantity is reported to say how far a derived "
 "scale falls short of a tuned one, not to bound any method in principle."),

("精简 1：去掉自证式框句（§10 Which objective）",
 "We checked whether that coefficient is in fact stable on this corpus, since if it were, a single-coefficient rule "
 "would be the right answer and our argument would be weaker. It is not stable:",
 "That coefficient is not stable on this corpus:"),

("精简 2：合并重复的收尾三句（§10 Which objective）",
 "This is the sense in which we read the negative result as consistent with the diagnosis rather than opposed to it, "
 "and it is why we present the derived threshold as a transferable calibration rather than as the most accurate gate. "
 "A reader who knows the target domain and can afford a validation set should therefore read our result as *a bound on "
 "how far a derived threshold falls short of a selected one* — $1.59\\times$ against the best-observed reference — "
 "rather than as a claim of superior accuracy. The two coincide only when the coefficient is stable across the "
 "configurations of interest.",
 "That is why we present the derived threshold as a transferable calibration rather than as the most accurate gate: it "
 "should be read as a bound on how far a derived threshold falls short of a selected one "
 "($1.59\\times$ against the best-observed reference), not as a claim of superior accuracy."),

("精简 3：压缩 §9.6 的重复列举",
 "Confirmed: the energy gate is the fragile component; a conventionally set threshold costs an order of magnitude; "
 "and a gate whose scale is derived rather than chosen removes that failure without tuning, to within $1.6\\times$ of "
 "the best-observed reference, and reports when it declines to act. Overturned: the mechanism behind the failure is "
 "not reference-scale drift but an operating point the median is simply too small to reach; the calibrated gate is not "
 "the most accurate rule on real speech, being marginally surpassed by a max-referenced gate and by a top-$5\\%$ "
 "ranking; and under impulsive interference its lower-tail model fails, though the failure is detected rather than "
 "silent.",

 "Confirmed: the energy gate is the fragile component, a conventionally set threshold costs an order of magnitude, "
 "and a derived scale removes that failure without tuning and reports when it declines to act. Overturned: the "
 "mechanism is not reference-scale drift but an operating point the median is too small to reach; the calibrated gate "
 "is not the most accurate rule here, being marginally surpassed by a max-referenced gate and a top-$5\\%$ ranking; "
 "and under impulsive interference its lower-tail model fails, though the failure is detected rather than silent."),
]


# R9（CSSP 审稿意见，Minor Revision）：题注里的 `false-admission-calibrated` 被进一步收紧为
# `null-calibrated`——意见指出 alpha 只是 noise-only null 下的 *nominal* 误收率，不是端到端保证。
# 于是本条目的 new 不再出现在稿中（而是被 R9 的措辞取代）。按"登记而非删除"保留条目、只跳过。
_SUPERSEDED_R9 = {
    "贡献 2：self-calibrating → false-admission-calibrated",
    "Algorithm 1 题注：self-calibrating → false-admission-calibrated",
    # R9 重写了 A2 的表述（改用协方差记号），于是本条的 old（含旧 A2 文本）失效；
    # A3 已在稿中，本条的目的已达成。
    "假定层：新增 A3（源独立激活）",
    # R9b 把该条目锚定的句子里的 $1.59\times$ 改成了 $1.63\times$（oracle 协议修正的连锁），
    # 锚点因此失效；合并收尾三句的效果已达成。
    "精简 2：合并重复的收尾三句（§10 Which objective）",
    # R9b 重写了 Table 15 题注里的 best-observed 均值、七列值与 penalty，
    # 本条锚定的定义句因此失效；「点明是观测值而非数学界」的效果已达成。
    "Table 15 定义句：点明是观测值而非数学界",
}


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    s = MS.read_text()
    before = s

    # ---- 0) 全局改名（可判定方向，幂等）
    n_rename = 0
    for old, new in RENAME:
        if old in s:
            n_rename += s.count(old)
            s = s.replace(old, new)
    print(f"全局改名：{'per-configuration bound -> best-observed reference，%d 处' % n_rename if n_rename else '已是新名，跳过'}")

    # ---- 1) 原子条目
    changed = skipped = 0
    missed: list[str] = []
    for tag, old, new in E:
        if tag in _SUPERSEDED_R9:
            print(f"  [R9 已承接] {tag}")
            skipped += 1
            continue
        if new in s:                      # 幂等：终态已在
            skipped += 1
            continue
        if old not in s:
            missed.append(tag)
            continue
        cnt = s.count(old)
        if cnt != 1:
            die(f"「{tag}」的旧文出现 {cnt} 次（要求唯一），先核对原文")
        s = s.replace(old, new)
        changed += 1
        print(f"  ✓ {tag}")

    print(f"\n已改 {changed} | 跳过 {skipped}（终态已在）| 未命中 {len(missed)}")
    for t in missed:
        print(f"  ! 未命中：{t}")
    if missed:
        die("有未命中的条目，整体不写入")

    if s == before:
        print("稿件无变化（已是最新状态）。")
        return

    # ---- 2) 结构终检
    # 注意：`false-admission-calibrated` 在 R9 里被进一步收紧为 `null-calibrated`
    # （意见要求 alpha 明说成 noise-only null 下的 nominal 率），故终检指向新词。
    for probe in ("**A3 (independent source activation).**",
                  "null-calibrated",
                  "probability-mass boundary",
                  "best-observed reference"):
        if probe not in s:
            die(f"终检失败：{probe!r} 不在文中")
    for probe in ("self-calibrating", "per-configuration bound"):
        if probe in s:
            die(f"终检失败：旧词 {probe!r} 仍有残留")
    if s.count("$") % 2:
        die(f"$ 不配对：共 {s.count('$')} 个")

    words_before = len(before.split())
    words_after = len(s.split())
    print(f"词数 {words_before} -> {words_after}（{words_after - words_before:+d}）")

    if a.dry:
        print("\n[dry] 未写入。")
        return
    MS.write_text(s)
    print(f"\n已写入 {MS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
