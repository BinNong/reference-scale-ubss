"""R14d：把 Assumptions（A1/A2/A3）从 §6 挪到 §3，消掉一处前向引用。

问题：三个编号假设写在 **§6 开头**，而 §4.2 的命题 6（及其证明、三分层陈述）和 §5.1 的
命题 8 都在用 A2 —— 早 200 行、早四个节，且首次出现处**没有前向指针**。更关键的是 §3
（问题建模）从没规定过噪声的*结构*（各向同性、实虚部独立），只给了 σ²=E|n_m|²。

做法：**整块拆成三条，分别贴到各自该在的模型旁边**——
  A1（混合矩阵结构）-> §3.1 末尾，紧挨 A 与 n 首次出现处；
  A2（噪声各向同性）-> §3.1 末尾（同一段域：n 的性质）；
  A3（源激活独立）  -> §3.3 末尾，紧挨"每个源以概率 p 独立激活"那句。
不新增编号小节（新增小节会让 §3.4 被 LaTeX 自动重编号，正是本项目反复踩的坑），
所以三条都以段首粗体标签的形式并入现有小节。

标签原文不动（"**A1.**" / "**A2 (isotropic TF-domain noise).**" / "**A3 (…)**"），
只去掉段首的 "**Assumptions.**" 引导词并拆段——全文只按 "A2" 这个名字引用，
没有任何一处写 "Section 6 的 assumptions"，所以拆搬不破坏任何交叉引用。

用法：
    python3 src/apply_r14d_assumptions.py --dry
    python3 src/apply_r14d_assumptions.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")
TAG = "R14d"

BLOCK_START = "**Assumptions.**"
A1_LABEL = "**A1."
A2_LABEL = "**A2 (isotropic TF-domain noise).**"
A3_LABEL = "**A3 (independent source activation).**"

SEC31_END = "### 3.2 Exact real-valued reformulation"
SEC33_END = "### 3.4 Difficulty measure"
SEC6_HEADING = "## 6. The Classical Criterion: What Remains Valid"


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    p = MS
    s = p.read_text()
    changed = skipped = missing = 0

    def split_block(t: str) -> tuple[str, str, str]:
        """取出 Assumptions 段，切成 A1/A2/A3 三条（去掉段首引导词）。"""
        if BLOCK_START not in t:
            die("找不到 '**Assumptions.**' 段")
        i = t.index(BLOCK_START)
        j = t.index("\n\n", i)
        blk = t[i:j]
        for lab in (A1_LABEL, A2_LABEL, A3_LABEL):
            if blk.count(lab) != 1:
                die(f"标签 {lab!r} 在块内出现 {blk.count(lab)} 次，预期 1 次")
        k1, k2, k3 = blk.index(A1_LABEL), blk.index(A2_LABEL), blk.index(A3_LABEL)
        if not (k1 < k2 < k3):
            die("A1/A2/A3 顺序不对")
        p1 = blk[k1:k2].strip()
        p2 = blk[k2:k3].strip()
        p3 = blk[k3:].strip()
        for lab, part in ((A1_LABEL, p1), (A2_LABEL, p2), (A3_LABEL, p3)):
            if len(part.split()) < 20:
                die(f"{lab} 那段太短（{len(part.split())} 词），切分可能出错")
        return p1, p2, p3

    if BLOCK_START in s:
        p1, p2, p3 = split_block(s)
        # 1) 摘掉 §6 里那段
        i = s.index(BLOCK_START)
        j = s.index("\n\n", i)
        s = s[:i] + s[j + 2:]
        # 2) A1 + A2 落到 §3.1 末尾（§3.2 标题之前）
        if s.count(SEC31_END) != 1:
            die(f"{SEC31_END!r} 不唯一")
        s = s.replace(SEC31_END, f"{p1}\n\n{p2}\n\n{SEC31_END}", 1)
        # 3) A3 落到 §3.3 末尾（§3.4 标题之前）
        if s.count(SEC33_END) != 1:
            die(f"{SEC33_END!r} 不唯一")
        s = s.replace(SEC33_END, f"{p3}\n\n{SEC33_END}", 1)
        changed += 1
        print(f"  ✓ Assumptions 段拆分：A1/A2 → §3.1 末尾，A3 → §3.3 末尾；§6 只留 Terminology+命题")
    elif A1_LABEL in s.split(SEC31_END)[0] and A3_LABEL in s.split(SEC33_END)[0]:
        skipped += 1
        print(f"  – {TAG} 已落地：A1/A2 在 §3.1 末尾、A3 在 §3.3 末尾")
    else:
        missing += 1
        print("  ✗ 未命中：'**Assumptions.**' 段")

    # ------------------------------------------------------------ 终检
    if not missing:
        body = s.split("## Appendix A.")[0]
        bad = []
        if BLOCK_START in s:
            bad.append("'**Assumptions.**' 段仍在")
        i31 = body.index(SEC31_END)
        i33 = body.index(SEC33_END)
        i41 = body.index("## 4. The Reference Scale")
        # A1/A2 必须落在 §3.2 标题之前，A3 落在 §3.4 标题之前、§4 之前
        i33h = body.index("### 3.3 ")
        for lab, lo, hi in ((A1_LABEL, 0, i31), (A2_LABEL, 0, i31), (A3_LABEL, i33h, i33)):
            if not (lo <= body.index(lab) < hi):
                bad.append(f"{lab} 落点不在预期区间")
        # 首次出现顺序：A2 必须在 §4.2 之前
        i42 = body.index("### 4.2 ")
        if body.index(A2_LABEL) > i42:
            bad.append("A2 仍在 §4.2 之后")
        if re.search(r"Assumptions? A2 \(Section 6\)", s):
            bad.append("残留前向指针")
        probes = [
            r"\operatorname{Cov}(\mathbf{n}_r)=\operatorname{Cov}(\mathbf{n}_i)=\frac{\sigma^2}{2}\mathbf{I}_M",
            r"\mathbf{n}_r\perp\mathbf{n}_i",
            r"\pi_J=\binom{N}{J}p^{J}(1-p)^{N-J}",
            "Circularly symmetric complex Gaussian noise realises the assumption exactly",
            "full *column* rank is unattainable",
            "the idealisation that the structured-source experiment of Section 8.6",
        ]
        for q in probes:
            if q not in s:
                bad.append(f"探针消失：{q[:44]}")
        if bad:
            die("终检失败：" + "；".join(bad))
        print("\n=== 终检 ===")
        print(f"  OK   A1/A2 在 §3.1 末尾、A3 在 §3.3 末尾")
        print(f"  OK   A2 首现位置 {body.index(A2_LABEL)} < §4.2 位置 {i42}")
        print(f"  OK   六条内容探针全部在场（协方差、二项律、实虚正交等）")

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    t = p.read_text()
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", t)]
    for lab, seq in (("表", caps), ("附录小节", secs), ("图", figs)):
        ok = seq == list(range(1, len(seq) + 1))
        print(f"  {'OK  ' if ok else 'FAIL'} {lab}：共 {len(seq)}，1..{max(seq)} 连续无重复")
        if not ok:
            die(f"{lab}编号异常")
    print(f"\n✓ {TAG} 全部终检通过")


if __name__ == "__main__":
    main()
