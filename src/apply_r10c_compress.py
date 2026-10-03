r"""C1：压缩 §11（Discussion and Limitations）的重复与自证式表述。

CSSP 第二轮意见 §9：`Writing | 整体较强，但略显过度防御和解释过多`，建议删约 15–20% 的
defensive discussion。**本轮只压"重复表述"与"为设计选择自辩"的句子，一条 limitation 事实都不删**
——那七条限制是第五、六、八轮审稿人明确追问过的内容（尤其单语料、结构化源、守卫误判率）。

九条，全部在 §11 内。数字（$3.9\%$、$17$/$270$、$1.0\%$/$3.4\%$/$3.5\%$/$0.005$–$0.035$、
$0.88°$/$1.12°$、$1.62\times$）一个不改，只改措辞。

用法：
    python3 src/apply_r10c_compress.py --dry
    python3 src/apply_r10c_compress.py
"""

import argparse
import pathlib
import sys

MS = pathlib.Path("paper/manuscript.md")

E = [
    # ---- *Third*：守卫段。事实全留，去掉一次重复的"但是"转折与结尾的重复提醒
    (
        "§11 *Third*：压守卫段的自证式表述",
        r'*Third*, the guard is a heuristic with a quantified margin rather than a statistical test. Across the 27 conditions of Table 4 it makes no material error, its least favourable case being decided against a $3.9\%$ margin; but at the resolution of individual seeds it is wrong about once in sixteen — $17$ of $270$ decisions at $\eta=0.035$ (Table 25) — and those errors run in both directions, so it should be read as a coarse detector of a failed premise rather than a reliable per-instance classifier. But its boundary sits between a largest "inactive-preferred" retention of $1.0\%$ and a smallest "active-preferred" retention of $3.4\%$; the threshold at $3.5\%$ sits in the resulting empty interval, and Appendix A.3 shows the decision to be unchanged for every value from $0.005$ to $0.035$, so the constant is not tuned to the reported grid. A user operating near that boundary should still treat the decision as uncertain.',
        r'*Third*, the guard is a heuristic with a quantified margin rather than a statistical test. Across the 27 conditions of Table 4 it makes no material error, its least favourable case being decided against a $3.9\%$ margin; at the resolution of individual seeds it is wrong about once in sixteen — $17$ of $270$ decisions at $\eta=0.035$ (Table 25) — in both directions, so it is a coarse detector of a failed premise rather than a reliable per-instance classifier. The constant is not tuned to the reported grid: the largest "inactive-preferred" retention is $1.0\%$ and the smallest "active-preferred" retention $3.4\%$, so $3.5\%$ falls in an empty interval, and Appendix A.3 shows the decision unchanged for every value from $0.005$ to $0.035$. Near that boundary the decision should be treated as uncertain.',
    ),
    # ---- *Second*：建议句去重
    (
        "§11 *Second*：压稠密源建议句",
        r'A user who needs accurate mixing-matrix estimation from dense sources should use a full-rank covariance or non-negative factorisation model [12] rather than a direction-clustering pipeline, with or without our gate.',
        r'A user who needs accurate estimation from dense sources should use a full-rank covariance or non-negative factorisation model [12], with or without our gate.',
    ),
    # ---- *Fifth*：去重复的"部署前须知"口吻
    (
        "§11 *Fifth*：压传感器数结论句",
        r'This is predicted by Corollary 1 and should be understood before deploying the method in a many-sensor array, where the classical pipeline is far less brittle to begin with.',
        r'This is predicted by Corollary 1, and in a many-sensor array the classical pipeline is far less brittle to begin with.',
    ),
    # ---- *Sixth*：机制句收紧
    (
        "§11 *Sixth*：收紧机制句",
        r'There the reliability of an admitted point rises monotonically with its energy, so the binding question is not how many noise-only points pass but how reliable the direction of a point is; a calibration to a false-admission rate answers the former, and the residual gap of Section 9.4 is attributable to the difference.',
        r'There the reliability of an admitted point rises with its energy, so the binding question is not how many noise-only points pass but how reliable an admitted direction is; that is the residual gap of Section 9.4.',
    ),
    # ---- *Sixth*：purity 权重是负结果、细节已在附录，正文压成一句
    (
        "§11 *Sixth*：压 purity 权重的负结果",
        r'; weighting by purity rather than energy alone, which we implemented and tested, does not close that difference, being indistinguishable from the collinearity weight on real speech and worse than the plain gate on the ladder.',
        r'; weighting by purity rather than energy alone does not close that difference.',
    ),
    # ---- Which objective：开头去重
    (
        "§11 Which objective：压开头",
        r'The first is maximum empirical accuracy: given data resembling the target, choose whichever rule scores best on it — which here is a max-referenced gate with one coefficient ($0.88°$ against $1.12°$). The second is a calibration that transfers: a threshold derived from quantities of the problem rather than selected on data. The present method optimises the second, and on the first it is beaten, narrowly, by rules that require a coefficient to be chosen.',
        r'The first is maximum empirical accuracy: given data resembling the target, choose whichever rule scores best on it — here a max-referenced gate with one coefficient ($0.88°$ against $1.12°$). The second is a calibration that transfers: a threshold derived from quantities of the problem rather than selected on data. The present method optimises the second and is beaten, narrowly, on the first.',
    ),
    # ---- Which objective：结尾的"不要再当成 accuracy 声明"是重复自辩
    (
        "§11 Which objective：压结尾的自辩",
        r'That is why we present the derived threshold as a transferable calibration rather than as the most accurate gate: it should be read as a bound on how far a derived threshold falls short of a selected one ($1.62\times$ against the best-observed reference), not as a claim of superior accuracy.',
        r'The derived threshold is therefore presented as a transferable calibration rather than the most accurate gate; its shortfall ($1.62\times$ against the best-observed reference) is the price of that transfer.',
    ),
    # ---- What the study establishes：收紧
    (
        "§11 What the study establishes：收紧",
        r'**What the study establishes.** The reference scale conventionally used to set the energy gate is not invariant to sparsity; its misalignment is computable in semi-analytic form; the knee that separates the two behaviours is at $\pi_0=1/2$, that is $p^\star=1-2^{-1/N}$; and a gate referenced to a blindly estimated noise floor, calibrated once to a false-admission rate, reproduces the performance of an oracle-tuned threshold while transferring across regimes, sensor counts, source counts and SNRs, and across the two mask-consuming clustering front ends tested.',
        r'**What the study establishes.** The reference scale conventionally used to set the energy gate is not invariant to sparsity; its misalignment is computable in semi-analytic form; the boundary that separates the two behaviours is the noise-mass half-probability point, $p^\star=1-2^{-1/N}$; and a gate referenced to a blindly estimated noise floor, calibrated once to a false-admission rate, reproduces an oracle-tuned threshold while transferring across regimes, sensor counts, source counts and SNRs, and across the two mask-consuming front ends tested.',
    ),
    # ---- Why the fix：标题去掉 "anyway"，正文去重复
    (
        "§11 Why the fix：标题与开头",
        r'**Why the fix is worth making anyway.** The strongest argument here is not accuracy. It is that the conventional threshold failed silently, by a factor of forty, in the regime where the two-stage paradigm is supposed to be at its best, and that no user inspecting the pipeline could have seen it.',
        r'**Why the fix is worth making.** The strongest argument is not accuracy but silence: the conventional threshold failed by a factor of forty in the regime where the two-stage paradigm is supposed to be at its best, and no user inspecting the pipeline could have seen it.',
    ),
]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def sec11_words(text: str) -> int:
    i, j = text.index("## 11. Discussion"), text.index("## 12. Conclusion")
    return len(text[i:j].split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args()
    p = pathlib.Path(a.manuscript) if a.manuscript else MS
    s = p.read_text()
    before = sec11_words(s)
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
        if old in new:
            die(f"条目 {tag!r} 的 new 包含完整的 old —— 会破坏幂等")
        s = s.replace(old, new, 1)
        print(f"  ✓ {tag}")
        changed += 1

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")
    after = sec11_words(s)
    print(f"\n§11 词数：{before} → {after}（−{before - after}，{(before - after) / before:.1%}）")
    if a.dry:
        print(f"演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # ------------------------------------------------------------ 终检
    t = p.read_text()
    print("\n=== 终检：limitation 事实必须一条不少 ===")
    for probe in ("Seven limitations are reported without qualification",
                  r"$17$ of $270$ decisions at $\eta=0.035$",
                  r"a $3.9\%$ margin",
                  'the largest "inactive-preferred" retention is ' + r"$1.0\%$",
                  r"$0.005$ to $0.035$",
                  r"$8$ of its $40$ decisions on structured sources, a rate of $20\%$",
                  r"a factor of two rather than twenty",
                  r"$1.12°$ to $0.72°$, better in fourteen of fifteen",
                  r"disables the gate in half the instances",
                  r"rests on one corpus",
                  r"$0.001$–$0.05$, a factor of $50$",
                  r"adds $0.17\%$ to the runtime"):
        ok = probe in t
        print(f"  {'OK  ' if ok else 'FAIL'} {probe[:66]}")
        if not ok:
            die("终检失败：有 limitation 事实丢失")
    for gone in ("worth making anyway", "That is why we present the derived threshold",
                 "But its boundary sits between", "which we implemented and tested, does not close"):
        print(f"  {'OK  ' if gone not in t else 'FAIL'} 旧的防御式措辞已消失：{gone[:40]!r}")
        if gone in t:
            die("终检失败：旧措辞仍在")
    print("\n✓ 全部终检通过")


if __name__ == "__main__":
    main()
