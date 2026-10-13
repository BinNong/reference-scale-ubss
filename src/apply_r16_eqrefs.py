#!/usr/bin/env python3
r"""第十六轮：给跨段/跨节回顾性指代补上显式的公式编号引用。

背景（2026-10-13 核查）：20 个显示公式在成品里都有编号 (1)–(20)，但正文**零处引用**。
核查结论是"就地并置 + 符号名指代"本身是合规的数学写作，不是缺陷；不过**结论段与几处
跨节指代**确实值得补编号——审稿人讨论具体式子时可以说 "Eq. (13)"，不必自己数。

设计原则（避免过度引用）：
  * **只补跨段/跨节的回顾性指代**，就地引导并立即使用的公式一律不加（加了反而啰嗦）。
  * 摘要（L11–L51）不加——编号在公式之后出现，摘要里无法正向引用。
  * 一律写成 `Eq. (n)` / `Eqs. (n) and (m)`，**不写裸 `(n)`**：
    正文里的 `(n)` 是数学记号（$\mathcal{O}(1)$、$\mathrm{Exp}(1)$、$AR(1)$），
    裸写既与它们混淆，也让 `verify_pdf` 的悬空引用检测认不出来。

编号依据：`build_pdf.number_equations()` 按出现顺序给第 k 个 `$$` 块编号 k，
故"第 k 个块 = (k)"。本脚本引用的号与公式内容的对应关系已在注释里逐条写明。

幂等：10 条是**句中插入**；另 2 条（Algorithm 1 题注、结论段的 (16)）必须**追加/后置**，
    因为 R9 有两条目的 `new` 正是那两句——详见条目内注释。这 2 条的 `old` 因此延伸到
    插入点之后（其中一条跨过段落边界），确保 old 不是 new 的子串。
用法：
    python3 src/apply_r16_eqrefs.py --dry
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

# (标签, old, new)  —— old 必须唯一；new 不得包含完整的 old（句首插句尾才需注意）
E = [
    # §4.1 收束句：$r(p)$ 由 (9) 定义（(9) = e_{f,t} > t_e r(p) nu,  r(p)=m_e/nu）
    ("§4.1 r(p) 指回 (9)",
     r"The whole question is therefore the behaviour of $r(p)$. If $r$ were constant,",
     r"The whole question is therefore the behaviour of $r(p)$ in Eq. (9). "
     r"If $r$ were constant,"),

    # §4.2 数值验证段：小参数展开就是 (12)（(12) = F_J(x) 的部分矩展开）
    ("§4.2 展开指回 (12)",
     r"(z/\theta)^J/J!$ on which it rests no longer applies",
     r"(z/\theta)^J/J!$ (Eq. 12) on which it rests no longer applies"),

    # §4.3 边界讨论：该段讨论的就是 (13)（(13) = p* = 1 - 2^{-1/N}）
    ("§4.3 边界讨论指回 (13)",
     r"It is a statement about probability mass, not an identity for the median.",
     r"The boundary of Eq. (13) is a statement about probability mass, "
     r"not an identity for the median."),

    # §5.3 保留准则：该准则即 (17)（(17) = n_keep >= max(n_min^abs, eta n_base)）
    ("§5.3 保留准则指回 (17)",
     r"The criterion is a guard against gross silent failure",
     r"The criterion of Eq. (17) is a guard against gross silent failure"),

    # §5.4 Algorithm 1 题注：跨节指回 §5.1 的 (14)(15) 与 §5.2 的 (16)
    #   (14) = sigma^2 = 2 e_(q) / Q_q(chi^2_2M)   (15) = F_0(x_q) < q   (16) = tau(M,alpha)
    # ⚠️ 这里必须**追加在句末之后**，不能插进句子里：R9 有条目把整段题注（以
    #    "...lower-tail floor estimate." 结尾）当作它的 `new`。只要我把引用插进那句中间，
    #    R9 的 `new` 就不再逐字存在、`--dry` 报未命中——而"nominal"这个词明明已在稿里。
    #    追加到句末之后，R9 的 `new` 仍是我的 `new` 的**字面子串**，它的 `if new in s`
    #    分支自动走"跳过"。`old` 因此要**跨过段落边界**（尾句 + "\n\n" + 下一段开头），
    #    否则 old 会是 new 的前缀、每次运行都再追加一遍（本脚本开头有护栏会拦下）。
    ("§5.4 算法题注指回 (16)(14)(15)",
     r"by the error of the lower-tail floor estimate." + "\n\n" + "```",
     r"by the error of the lower-tail floor estimate. The multiple is given by Eq. (16) "
     r"and the floor estimate by Eqs. (14) and (15)." + "\n\n" + "```"),

    # §7 基线定义：median-energy gate 的正式写法是 (8)（(8) = e_{f,t} > t_e m_e）
    ("§7 基线定义指回 (8)",
     r"energy gate at $0.02\times$ the median energy, spherical $k$-means",
     r"energy gate at $0.02\times$ the median energy (Eq. 8), spherical $k$-means"),

    # §8.1：自检关门的机制来自 §5.3 的 (17)
    ("§8.1 自检关机指回 (17)",
     r"At the dense extreme the self-check disables the gate",
     r"At the dense extreme the self-check of Eq. (17) disables the gate"),

    # ---- §12 结论：跨全文回顾，价值最高的一处 ----
    ("§12 概率质量边界指回 (13)",
     r"Its probability-mass boundary is exact",
     r"Its probability-mass boundary (Eq. 13) is exact"),

    # 半解析形式 = Proposition 6 的 (10)(11)（(10) 混合律 e=...，(11) F(x)=sum pi_J F_J）
    ("§12 半解析形式指回 (10)(11)",
     r"the semi-analytic form of Section 4.2, which is exact at noise-only",
     r"the semi-analytic form (Eqs. 10 and 11) of Section 4.2, "
     r"which is exact at noise-only"),

    ("§12 chi-square 分位关系指回 (14)",
     r"with the chi-square quantile relation, and calibrating",
     r"with the chi-square quantile relation (Eq. 14), and calibrating"),

    # ⚠️ 同 Algorithm 1 题注：R9 有条目以
    #    "calibrating the multiple to a nominal false-admission rate under the noise-only null
    #     yields a rule with one meaningful constant" 为 `new`。把引用插在 "constant" **之后**、
    #    "that transfers" **之前**，R9 的 `new` 仍是我这句的字面子串，它自动走"跳过"。
    #    `old` 必须延伸到插入点之后（到 "that transfers"），否则 old 是 new 的前缀、会重复插入。
    ("§12 假准入倍率指回 (16)",
     r"calibrating the multiple to a nominal false-admission rate under the "
     r"noise-only null yields a rule with one meaningful constant that transfers",
     r"calibrating the multiple to a nominal false-admission rate under the "
     r"noise-only null yields a rule with one meaningful constant (Eq. 16) "
     r"that transfers"),

    ("§12 保留自检指回 (17)",
     r"A retention self-check lets the method decline to act",
     r"A retention self-check (Eq. 17) lets the method decline to act"),
]


def die(msg: str) -> None:
    print("✗ " + msg)
    raise SystemExit(1)


def main() -> None:
    dry = "--dry" in sys.argv
    if not MS.exists():
        die(f"找不到 {MS}——本脚本须在仓库根运行（内部用相对路径）")
    s = MS.read_text()
    orig = s

    # 护栏：new 不得包含完整 old（句尾追加会破坏幂等）
    for tag, old, new in E:
        if old in new:
            die(f"[{tag}] new 包含完整 old，会重复插入——改成句中插入或拆分条目")

    changed = skipped = missing = 0
    for tag, old, new in E:
        n = s.count(old)
        if n == 0:
            if new in s:
                skipped += 1
                print(f"  跳过 {tag}（已应用）")
            else:
                missing += 1
                print(f"  ✗ 未命中 {tag}")
            continue
        if n > 1:
            die(f"[{tag}] 锚点不唯一（{n} 处）——加长锚点")
        s = s.replace(old, new, 1)
        changed += 1
        print(f"  ✓ {tag}")

    if missing:
        die(f"{missing} 条未命中——**不做任何写入**（避免改一半）")

    # ---------------- 终检 ----------------
    print("\n=== 终检 ===")
    body = s[: s.index("## References")]
    n_dollar_pairs = len(re.findall(r"(?m)^[ \t]*\$\$[ \t]*$", body)) // 2

    # 1) 引用号集合：与 verify_pdf 用同一套正则（逐字一致，别只在这里另写一套）
    ref_pat = re.compile(
        r"\b(?:Eqs?\.|[Ee]quations?)\s*\(?\s*\d{1,2}"
        r"(?:\s*\)?\s*(?:[\u2013,\-]|and)\s*\(?\s*\d{1,2})*\)?")
    cited = set()
    for m in ref_pat.finditer(body):
        cited.update(int(x) for x in re.findall(r"\d{1,2}", m.group(0)))
    cited = sorted(cited)
    over = [n for n in cited if not 1 <= n <= n_dollar_pairs]
    if over:
        die(f"终检失败：正文引用了不存在的公式号 {over}（共 {n_dollar_pairs} 个）")
    print(f"  OK  引用公式号 {cited}（共 {n_dollar_pairs} 个公式，无悬空）")

    # 2) 引用的号必须确实指向预定的公式（防"号对但内容指错"）
    expect = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    if set(cited) != expect:
        die(f"终检失败：引用集合 {cited} != 预期 {sorted(expect)}")
    print(f"  OK  引用集合与预定一致：{sorted(expect)}")

    # 3) 结构：表/图/附录小节/文献编号连续无重复（不写死张数）
    caps = [int(x) for x in re.findall(r"(?m)^\*\*Table (\d+)\.\*\*", s)]
    figs = [int(x) for x in re.findall(r"(?m)^\*\*Fig\. (\d+)\.\*\*", s)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+)", s)]
    refs = [int(x) for x in re.findall(r"(?m)^\[(\d+)\]", s)]
    for name, seq in (("表题注", caps), ("图题注", figs),
                      ("附录小节", secs), ("文献", refs)):
        if seq != list(range(1, len(seq) + 1)):
            die(f"终检失败：{name}编号不连续或有重复：{seq}")
    print(f"  OK  表 {len(caps)} / 图 {len(figs)} / 附录 {len(secs)} / "
          f"文献 {len(refs)} 编号均连续")

    # 4) $$ 配对（20 对 -> 公式编号 1..20 才能对上）
    if n_dollar_pairs != 20:
        die(f"终检失败：显示公式 {n_dollar_pairs} 对，预期 20（编号会整体漂移）")
    print(f"  OK  显示公式 {n_dollar_pairs} 对")

    # 5) 正文无中文
    hz = re.findall(r"[\u4e00-\u9fff]", body)
    if hz:
        die(f"终检失败：正文有 {len(hz)} 个中文字符")
    print("  OK  正文无中文")

    # 6) 摘要词数（引用都在 §4 之后，摘要应保持不变）
    #    终止边界必须是 **Keywords:**，不是 "## 1." —— 后者会把关键词行也算进摘要，
    #    实测 250 -> 272 全是关键词（本项目在"计数边界"上反复踩过，见技能检查 6/15）。
    m = re.search(r"## Abstract\s*\n+(.*?)\n+\s*\*\*Keywords:\*\*", s, re.S)
    if not m:
        die("终检失败：抓不到摘要段（## Abstract ... **Keywords:**）")
    abstract = re.sub(r"[*$`]", "", m.group(1)).strip()
    n_words = len(abstract.split())
    print(f"  ·   摘要词数 {n_words}（CSSP 上限 250）")
    if n_words > 250:
        die(f"终检失败：摘要 {n_words} 词超过 250")

    if dry:
        print(f"\n（演练）将改 {changed} 处；未写入。")
        return

    MS.write_text(s)
    print(f"\n已写入 {MS}：改 {changed} 处、跳过 {skipped} 处。")
    print(f"字节 {len(orig)} -> {len(s)}（+{len(s) - len(orig)}）")


if __name__ == "__main__":
    main()
