r"""R10e：数据集合规声明（Data / Code / Ethics / Competing interests / Funding）+ 两条语料文献。

老师问"用了哪些数据集、会不会侵权"。实测（服务器 curl openslr.org）三个语料的官方许可：

  LibriSpeech dev-clean   SLR12   CC BY 4.0      §9 真实语音（[26] 已有）
  Nigerian English        SLR70   CC BY-SA 4.0   §9.6 / §A.20
  REVERB 2014 RIR + 噪声  SLR28   Apache 2.0     §10.5 / §A.18

三个许可都允许研究使用与发表，**不构成侵权**。唯一有实质风险的是 SLR70 的 **ShareAlike**：
不分发语料本身则不触发（本稿只发代码/配置/记录，不发音频）——声明里把这一点**明写出来**，
于是"合规"从隐含变成可查验。

四处改动（**不动任何数字、不触发表号重排**）：

  1. §12 之后新增 `## Declarations`（Data availability / Code availability / Ethics /
     Competing interests / Funding）。**无编号小节**——`build_pdf` 的节数断言只数 `## N.`，
     故不影响；`subsec` 解析只认 `### N.M`，故不受影响。
  2. §A.18 的 SLR28 句加引用 `[30]`；
  3. §A.20 的 SLR70 句加引用 `[29]`；
  4. 参考文献末尾追加 `[29]` / `[30]`。这是**方向可判定的独立步骤**——纯追加写成 (old,new)
     会触发 "new 含 old" 的幂等护栏；且编号 [1]..[28] 本就**不是引用顺序**（[27] 在正文 L25
     就已出现），追加两号不打乱任何既有编号，`verify_r6` 的 [27]/[28] 断言不受影响。

**注**：`SEC_DECL` 里的 Funding 一项取的是**终态**（无资助的正式声明，由 R11
`apply_r11_funding.py` 定稿）。R10e 首次落稿时它是占位符，R11 才替换成正式声明；
两边对齐后，R10e 从零重跑即产出终态，R11 则因 `new in s` 成立而自动跳过——**脚本链保持幂等**。
（`verify_r10` 的 G 组也已同步为"占位符已清除"。）

用法：
    python3 src/apply_r10e_declarations.py --dry
    python3 src/apply_r10e_declarations.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")

SEC_DECL = r"""## Declarations

**Data availability.** All speech and room material is third-party and openly licensed, and is
downloaded from OpenSLR and used unmodified under its own terms: LibriSpeech dev-clean [26]
(OpenSLR SLR12, CC BY 4.0), built from public-domain audiobooks; the Nigerian English set [29]
(OpenSLR SLR70, CC BY-SA 4.0); and the recorded room-impulse-response and noise data [30]
(OpenSLR SLR28, Apache 2.0). No corpus is redistributed with this paper; what is released is the
code, the mixing configurations, the seed sets and the machine-readable records listed in
Section 7, from which every table and figure can be regenerated once the corpora have been
obtained from their hosts.

**Code availability.** The code, the fixed configurations, the dependency list and every record
named in Section 7 are available at [repository to be inserted], together with a README that maps
each table and figure to the script and the record that produce it.

**Ethics.** Only publicly available, openly licensed corpora are used. No data were collected
from human participants and no experiments on human subjects were performed for this study, so
institutional ethics approval is not applicable.

**Competing interests.** The authors declare no competing interests.

**Funding.** The authors declare that no funds, grants, or other support were received
during the preparation, study or publication of this article."""

E = [
    (
        "§12 之后新增 `## Declarations`（无编号小节）",
        r"A data-driven rule for the lower-tail cap was tried and rejected on the same terms (Appendix A.4)."
        + "\n\n---\n\n" + "## Appendix A. Robustness and Cost of the Front End",
        r"A data-driven rule for the lower-tail cap was tried and rejected on the same terms (Appendix A.4)."
        + "\n\n---\n\n" + SEC_DECL + "\n\n---\n\n" + "## Appendix A. Robustness and Cost of the Front End",
    ),
    (
        "§A.18：SLR28 句加引用 [30]",
        r"The material is OpenSLR SLR28, the room-impulse-response and noise database" + "\n"
        + r"released with the 2014 REVERB challenge:",
        r"The material is OpenSLR SLR28 [30], the room-impulse-response and noise database" + "\n"
        + r"released with the 2014 REVERB challenge:",
    ),
    (
        "§A.20：SLR70 句加引用 [29]",
        r"That corpus is the Nigerian English set released as OpenSLR SLR70 (CC BY-SA 4.0):",
        r"That corpus is the Nigerian English set released as OpenSLR SLR70 [29] (CC BY-SA 4.0):",
    ),
]

# 参考文献末尾追加：锚点在文件末尾，唯一。
REF_ANCHOR = r"no. 7, 1227, 2024."
REFS_NEW = ("\n\n" + r'[29] Google, "Crowdsourced high-quality Nigerian English speech data set,"'
            + " OpenSLR resource SLR70, 2019. [Online]. Available: https://www.openslr.org/70/"
            + "\n\n" + r'[30] OpenSLR, "Room impulse response and noise database,"'
            + " OpenSLR resource SLR28, 2017. [Online]. Available: https://www.openslr.org/28/")


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

    # 参考文献追加：方向可判定（只有 [29] 尚不存在时才追加），不受 "new 含 old" 护栏约束。
    if "\n[29] Google" in s:
        print("  [已改，跳过] 参考文献追加 [29] / [30]")
        skipped += 1
    elif s.count(REF_ANCHOR) == 1:
        s = s.replace(REF_ANCHOR, REF_ANCHOR + REFS_NEW, 1)
        print("  ✓ 参考文献追加 [29] / [30]")
        changed += 1
    else:
        print(f"  ✗ 未命中或锚点不唯一：参考文献末尾（{s.count(REF_ANCHOR)} 次）")
        missing += 1

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # ------------------------------------------------------------------ 终检
    t = p.read_text()
    print("\n=== 终检 ===")
    for probe in ("## Declarations",
                  "OpenSLR SLR12, CC BY 4.0",
                  "OpenSLR SLR70, CC BY-SA 4.0",
                  "OpenSLR SLR28, Apache 2.0",
                  "institutional ethics approval is not applicable",
                  "No corpus is redistributed with this paper",
                  "OpenSLR SLR28 [30]",
                  "OpenSLR SLR70 [29]",
                  "[29] Google",
                  '[30] OpenSLR, "Room impulse response',
                  "**Funding.** The authors declare that no funds, grants, or other support"):
        ok = probe in t
        print(f"  {'OK  ' if ok else 'FAIL'} {probe[:60]}")
        if not ok:
            die("终检失败")

    if t.count("## Declarations") != 1:
        die(f"`## Declarations` 应恰为 1 次，实为 {t.count('## Declarations')}")

    refs = [int(x) for x in re.findall(r"(?m)^\[(\d+)\]", t)]
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", t)]
    for lab, seq in (("文献", refs), ("表", caps), ("附录小节", secs), ("图", figs)):
        ok = seq == list(range(1, len(seq) + 1))
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}：共 {len(seq)}，1..{max(seq)} 连续无重复")
        if not ok:
            die(f"{lab}编号不连续或有重复：{seq}")
    if "rests on one corpus" in t:
        die("终检失败：§11 *Seventh* 仍写'只有一个语料'")
    print("\n✓ 全部终检通过")


if __name__ == "__main__":
    main()
