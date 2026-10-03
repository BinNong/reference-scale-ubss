r"""C2：第二个真实语料的落稿。

服务器网络实测：`openslr.org` ✓、`openairlib.net` ✓，`zenodo.org` / `archive.org` /
`raw.githubusercontent.com` ✗。故第二语料取 **OpenSLR SLR70**（Nigerian English，CC BY-SA 4.0，
男声 454 MB + 女声 760 MB = 31 说话人 / 3359 条 / 48 kHz），按说话人重组为
`data/slr70_corpus/<spk>/*.wav` 后，用**与 §9.1 完全相同的协议**重跑
（同 15 配置、同 8 评测 seed、同 2 个 held-out 调参 seed、同混合矩阵/聚类器/ℓ1/指标）。

四处改动（表放附录末尾，**不触发表号重排**）：

  1. §9.6：补一句指向 §A.20 的跨语料结论；
  2. §11 *Seventh*：由"只有一个语料"改写为"两个语料，但都不是阵列"；
  3. 附录新增 **§A.20 + Table 45**（插在 §A.19 之后、"## Figures" 之前）；
  4. §7 可用性清单：点名 `r10_real_slr70.json` 与 `r10_real_slr70_grid.json`。

用法：
    python3 src/apply_r10d_second_corpus.py --dry
    python3 src/apply_r10d_second_corpus.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")

SEC_A20 = r"""### A.20 A second corpus

The protocol of Section 9.1 is repeated unchanged on a second corpus, so that the finding can be separated from the material it was found on. That corpus is the Nigerian English set released as OpenSLR SLR70 (CC BY-SA 4.0): $3{,}359$ utterances from $31$ speakers, $19$ female and $12$ male, recorded through consumer hardware at $48$ kHz rather than as close-microphone read English at $16$ kHz, and resampled to $16$ kHz on load. Nothing else changes — the same fifteen configurations, the same eight evaluation seeds, the same two held-out tuning seeds, the same mixing matrix, the same clusterer, the same debiased $\ell_1$ recovery and the same metrics — so the two rows of Table 45 differ only in the speech they are computed from. The records are `r10_real_slr70.json` ($960$ runs) and `r10_real_slr70_grid.json` (the tuning grids).

**Table 45.** The real-speech comparison on two corpora. Mean mixing-matrix angle error in degrees over the fifteen configurations of Section 9.1; "best-observed" is the per-configuration minimum over the three tuned columns, and the two penalties are per-configuration ratios against it averaged over configurations, as defined in Table 15.

| Corpus | NF-SSP | conventional | best-observed | NF penalty | conventional penalty | NF better in |
|---|---|---|---|---|---|---|
| LibriSpeech dev-clean | $1.12°$ | $8.26°$ | $0.69°$ | $1.62\times$ | $13.1\times$ | $15/15$ |
| SLR70 Nigerian English | $1.15°$ | $8.89°$ | $0.61°$ | $1.86\times$ | $15.2\times$ | $14/15$ |

The two corpora behave the same way in the quantities that carry the claim rather than in their digits. The conventional reference scale costs an order of magnitude on each — $13.1\times$ and $15.2\times$, with worst configurations at $32\times$ and $31\times$ — so the failure it produces is not a property of LibriSpeech. The calibrated gate transfers at essentially the same absolute error, $1.15°$ against $1.12°$, without any change of constant, and it beats the conventional default in fourteen of the fifteen configurations on the new corpus against fifteen of fifteen on the old. One quantity does move: the penalty against the best-observed reference rises from $1.62\times$ to $1.86\times$. The price of transferring without a selected coefficient is therefore not exactly corpus-independent, which is what a second corpus was needed to establish."""

E = [
    (
        "§9.6：补跨语料结论并指向 §A.20",
        r"the weight we implemented realises it on real speech but not on the synthetic ladder (Section 9.5)."
        + "\n\n" + "## 10. Robustness of the calibration",
        r"the weight we implemented realises it on real speech but not on the synthetic ladder (Section 9.5). "
        + r"A second corpus — $3{,}359$ Nigerian English utterances recorded through consumer hardware, drawn from "
        + r"OpenSLR SLR70 — reproduces the comparison rather than merely the numbers: the conventional reference "
        + r"scale again costs an order of magnitude ($15.2\times$ against the best-observed reference, worst "
        + r"configuration $31\times$), the calibrated gate again lands within $1.86\times$ of it at $1.15°$, and it "
        + r"again beats the conventional default in fourteen of the fifteen configurations (Appendix A.20 and Table 45)."
        + "\n\n" + "## 10. Robustness of the calibration",
    ),
    (
        "§11 *Seventh*：由单语料改写为两语料",
        r"*Seventh*, the real-speech evaluation rests on one corpus. All $960$ real-speech runs draw on LibriSpeech dev-clean, and although the fifteen configurations vary the window length, the source count, the SNR, the denseness of two sources and the noise type, they cannot speak to corpora with different recording conditions, languages, or array geometries. The claim the real data supports is narrow and we state it as such: on this corpus the conventional reference scale fails by an order of magnitude and the derived one does not. A second corpus is the natural next step and we did not have access to one for this study.",
        r"*Seventh*, the real-speech evaluation rests on two corpora but not on many. The $1{,}920$ real-speech runs draw on LibriSpeech dev-clean (read English, close microphone) and on SLR70's Nigerian English (crowd-sourced, consumer hardware), so two language varieties and two recording regimes are covered, and the fifteen configurations vary the window length, the source count, the SNR, the denseness of two sources and the noise type. Neither corpus has a microphone array with a real propagation geometry, and the claim the real data supports stays narrow: on both corpora the conventional reference scale fails by an order of magnitude and the derived one does not (Appendix A.20).",
    ),
    (
        "附录：新增 §A.20 + Table 45（插在 §A.19 之后）",
        r"The record is `r10_gram_spectrum.json`." + "\n\n" + "## Figures",
        r"The record is `r10_gram_spectrum.json`." + "\n\n" + SEC_A20 + "\n\n" + "## Figures",
    ),
    (
        "§7 可用性清单：点名第二个语料的两个归档",
        r"two hundred mixing matrices). Code, the fixed configurations,",
        r"two hundred mixing matrices). The second-corpus study of Appendix A.20 adds "
        r"`r10_real_slr70.json` ($960$ runs on OpenSLR SLR70's Nigerian English set, under the protocol "
        r"of Section 9.1) and its tuning grids in `r10_real_slr70_grid.json`. Code, the fixed configurations,",
    ),
]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args()
    p = pathlib.Path(a.manuscript) if a.manuscript else MS
    s = p.read_text()
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
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # ------------------------------------------------------------ 终检：
    # 表号/附录小节只断言**连续无重复**（不写死张数）——写死 N 正是本项目反复误报的原因。
    t = p.read_text()
    print("\n=== 终检 ===")
    for probe in ("### A.20 A second corpus",
                  r"**Table 45.** The real-speech comparison on two corpora",
                  r"SLR70 Nigerian English | $1.15°$",
                  r"r10_real_slr70.json",
                  r"The $1{,}920$ real-speech runs"):
        ok = probe in t
        print(f"  {'OK  ' if ok else 'FAIL'} {probe[:60]}")
        if not ok:
            die("终检失败")
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", t)]
    for lab, seq in (("表", caps), ("附录小节", secs), ("图", figs)):
        ok = seq == list(range(1, len(seq) + 1))
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}：共 {len(seq)}，1..{max(seq)} 连续无重复")
        if not ok:
            die(f"{lab}编号不连续或有重复：{seq}")
    if "rests on one corpus" in t:
        die("终检失败：§11 *Seventh* 仍写'只有一个语料'")
    print("\n✓ 全部终检通过")


if __name__ == "__main__":
    main()
