"""R14b：正文压字（方案 C 的第二步，不动任何数字、不动任何论断）。

R14 已把 §10 整节搬进附录；这里只削"为防误读而重复的辩护性文字"，三处：

  1. §7 的"comparators that matter"段（330 -> ~215 词）：删掉对同一结构理由的两遍解释，
     保留结论、保留指向 A.12 的正面对比与"radius tuned per regime, an oracle the calibrated
     gate never gets"这句要害；
  2. §7 Records 段的"Successive revisions add …"清单句（95 -> ~45 词）：它逐条复述附录
     A.1–A.25 的标题，改为一句话指向附录本身；
  3. §9.5 的 467 词结果段（-> ~400 词）：删"Two caveats belong with these numbers"这类
     框架句与同义反复，**每个数字、每个 p 值、每个跨域对照全部保留**。

不碰 §2（"a factor of $87$"、Table 14 的两档区间是 verify_r6/verify_revision 的探针所在）、
不碰 §11 七条 limitation（verify_r10 在此有四个探针）。

幂等：第二次运行报"全跳过、未命中 0"。
用法：
    python3 src/apply_r14b_compress.py --dry
    python3 src/apply_r14b_compress.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")
TAG = "R14b"

SEC7_COMPARATORS_NEW = (
    "The comparison is between energy-gate rules with everything else held fixed, so the comparators "
    "that matter are pipelines that consume an SSP mask. Density-based clusterers (DBSCAN, OPTICS, "
    "density-peak), neighbourhood-confidence detectors of the TIFROM family, and transforms that "
    "sharpen sparsity before detection act either on the set a gate has already admitted or in place "
    "of the detection stage, so the reference scale of the gate does not enter their input at all; "
    "that is an argument about what this comparison isolates and not about accuracy, and the accuracy "
    "comparison is made anyway in Appendix A.12, which enters the three density fronts head to head — "
    "on all TF points, and separately on the collinearity-admitted set with the energy criterion "
    "removed — with their radius tuned per regime, an oracle the calibrated gate never gets. They lead "
    "in the dense regime and trail in the sparse ones, and we report that rather than asserting the "
    "structural argument. The two families that do consume a mask are included, with the mask fixed "
    "and only the front end varied (Section 8.5). Among the rules that decide *which points an SSP "
    "test sees*, referencing that decision to the noise floor removes the need to select a "
    "coefficient; we do not claim that a pipeline which bypasses the gate cannot do better, and "
    "Section 8.6 reports a setting in which a hand-set fixed multiple indeed is better.")

SEC7_RECORDS_OLD = ("Successive revisions add density front ends and a robust-scale comparison, "
                    "head-to-head density clustering, the complex- and convolutive-mixing study, "
                    "the $\\alpha$ sensitivity, the banded floor, order selection, the gate-plus-weight "
                    "design of Section 9.5, the source-number mismatch study, the recorded room, an "
                    "eigenstructure sweep of $4{,}400$ column subsets and a second corpus of $960$ runs.")
SEC7_RECORDS_NEW = ("Successive revisions add the studies Appendix A collects by name — density front "
                    "ends and clustering, complex and convolutive mixing, the banded floor, order "
                    "selection, the source-number mismatch, the recorded room, an eigenstructure sweep "
                    "and a second corpus.")

SEC95_NEW = (
    "The mechanism of Section 9.5 suggests a better device than a hard gate: if a point's reliability "
    "is a smooth function of its energy, the natural response is to weight each direction by that "
    "reliability rather than to discard points. Weighting the spherical $k$-means by point energy "
    "lowers the mean error on real speech from $1.12°$ to $0.72°$ ($p=6\\times10^{-5}$, paired over "
    "the $15\\times8$ configuration–seed cells), better than the unweighted gate in fourteen of the "
    "fifteen configurations. Augmenting the weight by $|\\cos|^4$ gives a further $0.68°$, but that "
    "increment is **not** significant ($0.72°\\to0.68°$, $p=0.23$): the entire gain is carried by the "
    "energy weight, and the weighted variant is then within noise of the per-configuration "
    "hard-threshold bound of $0.73°$ (Table 15). The gains concentrate where the admitted set is most "
    "heterogeneous: under babble noise ($2.62°\\to0.79°$), at $N=6$ ($1.85°\\to1.00°$) and in the "
    "two-dense-source configuration ($1.56°\\to0.90°$).\n\n"
    "First, the energy weight alone fails at SNR $0$ dB ($7.32°$, against $1.41°$ for the gate): when "
    "signal and noise energies are comparable the weight cannot separate them, and the hard removal "
    "performed by the gate is still required — it is the *combination* of gate and weight that we "
    "recommend. Second, the synthetic ladder gives no gain at all, and the sign there is the one the "
    "amplitude–purity relation predicts: the weighted mean is $2.49°$ against $2.12°$ for the plain "
    "gate, worse in four of the six regimes, with the only significant single contrast at $p=0.10$ "
    "($0.32°$ against $0.58°$, $p=0.03$), the pooled difference not significant ($p=0.19$), and the "
    "largest mean gap at $p=0.40$ smaller than its own seed-to-seed spread. In the i.i.d. model the "
    "highest-energy points are preferentially superpositions (Table 16), so raising their weight pulls "
    "the cluster centres toward mixed directions — the opposite of what happens on real speech. The "
    "repair this mechanism names, a weight on the *purity* of a point estimated blindly as its "
    "collinearity with the nearest cluster centre, was implemented and is reported as a tested "
    "negative in both domains: on the ladder it recovers part of the penalty ($2.49°\\to2.42°$ at the "
    "eighth power) but reaches the plain gate in no regime, and on real speech it is statistically "
    "indistinguishable from the collinearity weight it was meant to improve on — $0.676°$ against "
    "$0.677°$ at the second power, $0.704°$ and $0.689°$ at the fourth and eighth, none of the three "
    "paired contrasts significant ($p\\ge0.51$ over the $15\\times8$ cells). We therefore report the "
    "weighted variant as established on real speech and unsupported on the i.i.d. ladder — the purity "
    "weight included — rather than as a uniformly better method.")


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def replace_paragraph(s: str, anchor: str, new: str, tag: str) -> tuple[str, int, int]:
    """把 anchor 所在自然段整段换成 new。返回 (新文本, 旧词数, 新词数)。

    new 若跨多个自然段（如 §9.5 的压缩版含一个段间空行），落地后 anchor 只会命中其
    第一段，此时旧文本是新文本的前缀 —— 这算"已落地"，不是幂等破坏。
    """
    if anchor not in s:
        return s, -1, -1
    i = s.index(anchor)
    j = s.index("\n\n", i)
    old = s[i:j]
    if new.strip() == old.strip():
        return s, len(old.split()), len(new.split())
    if new.strip().startswith(old.strip()):
        # 已落地：anchor 只命中新文本的第一段，报告新文本的真实词数
        return s, len(new.split()), len(new.split())
    if old in new:
        die(f"{tag}：new 包含完整的 old")
    return s[:i] + new + s[j:], len(old.split()), len(new.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    p = MS
    s = p.read_text()
    changed = skipped = missing = 0

    # ---------------------------------------------------------------- 1. §7 辩护段
    s, wo, wn = replace_paragraph(
        s, "The comparison is between energy-gate rules", SEC7_COMPARATORS_NEW, "§7 comparators")
    if wo < 0:
        missing += 1
        print("  ✗ 未命中：§7 'comparators that matter' 段")
    elif wn < wo:
        changed += 1
        print(f"  ✓ §7 辩护段 {wo} -> {wn} 词")
    else:
        skipped += 1
        print(f"  – §7 辩护段已是压缩版（{wo} 词）")

    # ---------------------------------------------------------------- 2. §7 Records 清单句
    if SEC7_RECORDS_OLD in s:
        if SEC7_RECORDS_NEW in s:
            skipped += 1
            print("  – §7 Records 清单句已是压缩版")
        else:
            s = s.replace(SEC7_RECORDS_OLD, SEC7_RECORDS_NEW, 1)
            changed += 1
            print(f"  ✓ §7 Records 清单句 {len(SEC7_RECORDS_OLD.split())} -> "
                  f"{len(SEC7_RECORDS_NEW.split())} 词")
    elif SEC7_RECORDS_NEW in s:
        skipped += 1
        print("  – §7 Records 清单句已是压缩版")
    else:
        missing += 1
        print("  ✗ 未命中：§7 Records 的 'Successive revisions add …' 句")

    # ---------------------------------------------------------------- 3. §9.5 结果段
    s, wo, wn = replace_paragraph(
        s, "The mechanism of Section 9.5 suggests a better device", SEC95_NEW, "§9.5")
    if wo < 0:
        missing += 1
        print("  ✗ 未命中：§9.5 的结果段")
    elif wn < wo:
        changed += 1
        print(f"  ✓ §9.5 结果段 {wo} -> {wn} 词")
    else:
        skipped += 1
        print(f"  – §9.5 结果段已是压缩版（{wo} 词）")

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # ------------------------------------------------------------ 终检：数字一个不能少
    t = p.read_text()
    print("\n=== 终检：压缩前后必须在场的字面值 ===")
    probes = [
        "fourteen of the fifteen configurations",
        "$1.12°$ to $0.72°$", "$p=6\\times10^{-5}$", "$0.72°\\to0.68°$", "$p=0.23$",
        "$0.73°$", "under babble noise ($2.62°\\to0.79°$)", "($1.85°\\to1.00°$)",
        "($1.56°\\to0.90°$)", "$7.32°$", "$1.41°$", "$2.49°$", "$2.12°$",
        "four of the six regimes", "$0.32°$", "$p=0.03$", "$p=0.19$",
        "Table 16", "$2.49°\\to2.42°$", "$0.676°$", "$0.677°$", "$0.704°$", "$0.689°$",
        "$p\\ge0.51$", "an oracle the calibrated gate never gets",
        "Section 8.6 reports a setting in which a hand-set fixed multiple indeed is better",
        "1{,}344",
    ]
    bad = [q for q in probes if q not in t]
    for q in probes:
        print(f"  {'OK  ' if q in t else 'FAIL'} {q[:64]}")
    if bad:
        die(f"终检失败：{len(bad)} 个探针消失")
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    for lab, seq in (("表", caps), ("附录小节", secs)):
        ok = seq == list(range(1, len(seq) + 1))
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}：共 {len(seq)}，1..{max(seq)} 连续无重复")
        if not ok:
            die(f"{lab}编号异常")
    print(f"\n✓ {TAG} 全部终检通过（正文词数 "
          f"{sum(1 for _ in t.split('## Appendix A.')[0].split())}）")


if __name__ == "__main__":
    main()
