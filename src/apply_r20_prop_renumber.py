r"""R20：把 Proposition 编号改成随文档出现顺序。

现状（2026-10-15 核查）：编号**不随出现顺序**，文档里依次是

    §4.2  Proposition 6      §4.3  Proposition 7
    §5.1  Proposition 8      §5.2  Proposition 9
    §6    Proposition 1, 2, 3, 4, 5      ← §6「The Classical Criterion」

即出现顺序为 6,7,8,9,1,2,3,4,5。编号本身无断号、无重号，全部交叉引用
数值自洽（§1 引 P2/P3/P4/P5、§2 引 P6/P7、§5.3 引 P7–P8、附录 A.3 引 P6、
A.4 引 P4、A.13 引 P6、A.26 引 P5 …），所以**这不是引用错误，是编号约定问题**：
Springer 的惯例是编号随首次出现顺序，审稿人容易把"6 在 1 前面"当成笔误。

成因：R14e（2026-10-06）补 Proposition 5 之前，编号本来就是断档态
`1,2,3,4,6,7,8,9` —— §4–§5（新方法）占 6–9、§6（古典准则回顾）占 1–4。
R14e 只是把 §6 那条 `*Scaling argument.*` 升级成 Proposition 5 填上空位，
于是 §6 凑成整齐的 1–5，让断档看起来像有意分组。**断档在先。**

本脚本改成随出现顺序：

    旧 6,7,8,9（§4.2, §4.3, §5.1, §5.2）→ 新 1,2,3,4
    旧 1,2,3,4,5（§6）                     → 新 5,6,7,8,9

## 为什么不能用逐条 (old, new) 替换

这是一个**无不动点的置换**：1→5, 2→6, 3→7, 4→8, 5→9, 6→1, 7→2, 8→3, 9→4。
逐条替换必然二次命中——先把 6 改成 1，再处理 1→5 时会把刚生成的 1 又改成 5，
再处理 5→9 时把它改成 9……一轮下来全乱，而且每跑一次就乱一次。
所以这里用**单遍全局重写**（`re.sub` 带映射函数，一次成型、不迭代），
一次运行完成全部 9 个定义与 23 处引用。

## 幂等判据：定义序，不是文字

置换类改动无法用"`new` 是否逐字仍在"判幂等（`new` 本身含新号，
且新号在旧稿里本就存在）。改用**内容型判据**：文档里 9 个 Proposition 定义的
出现顺序恰为 1..9 即视为已应用。已应用就整篇跳过、零写入。

用法：
    python3 src/apply_r20_prop_renumber.py --dry
"""

import pathlib
import re
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

# 旧号 → 新号（按文档出现顺序重排；无不动点，必须单遍重写）
MAP = {6: 1, 7: 2, 8: 3, 9: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9}

# 改前的定义序（§4.2 起）与改后的定义序
OLD_SEQ = [6, 7, 8, 9, 1, 2, 3, 4, 5]
NEW_SEQ = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 定义行：**Proposition N (title).**
DEF = re.compile(r"(?m)^\*\*Proposition (\d+)\b")

# 引用片段：Proposition N / Propositions N and M / Propositions N–M（en-dash）
# 分隔符含 `)` 是为了吃 "Propositions (7)–(8)" 这种带括号的区间。
SPAN = re.compile(
    r"\b(Propositions?)"
    r"(\s+\d+(?:(?:\s*\)?\s*(?:[–,\-]|and)\s*)\d+)*)"
)


def die(msg: str) -> None:
    print("FAIL " + msg)
    sys.exit(1)


def defs(t: str) -> list[int]:
    """文档里 Proposition 定义的出现顺序。"""
    return [int(m.group(1)) for m in DEF.finditer(t)]


def renumber(t: str) -> str:
    """单遍重写所有 Proposition 编号（不做迭代——迭代会死循环）。"""
    def one_span(m: re.Match) -> str:
        return m.group(1) + re.sub(
            r"\d+", lambda d: str(MAP[int(d.group(0))]), m.group(2))

    return SPAN.sub(one_span, t)


def main() -> None:
    dry = "--dry" in sys.argv
    s = MS.read_text()

    # 护栏：映射必须是 1..9 的一个排列，且无不动点（否则单遍重写不安全）
    if sorted(MAP) != list(range(1, 10)) or sorted(MAP.values()) != list(range(1, 10)):
        die("护栏：MAP 不是 1..9 的排列")
    fixed = [k for k, v in MAP.items() if k == v]
    if fixed:
        die(f"护栏：MAP 存在不动点 {fixed}，逐条替换才安全，单遍重写不需要")

    before = s
    seq = defs(before)
    n_occ_before = len(SPAN.findall(before))

    if seq == NEW_SEQ:
        print("跳过：Proposition 定义序已是 1..9（R20 已应用），零写入")
    elif seq == OLD_SEQ:
        # 无论 dry 与否都在内存里应用，终检验的是"将要成为"的状态
        # （R19 踩过：dry 时不替换，终检读磁盘，报假的 FAIL）。
        s = renumber(before)
        if not dry:
            MS.write_text(s)
        print(f"已改：{n_occ_before} 处 Proposition 编号按出现顺序重排"
              f"（{OLD_SEQ} → 1..9）")
    else:
        die(f"未命中：定义序为 {seq}，既不是改前 {OLD_SEQ} 也不是改后 1..9。"
            f"（若此前有别的脚本改过 Proposition 标签，先人工核对再动手。）")

    # ---- 终检：断言改后的最新状态（dry 时校验内存中的 s）----
    print()
    ok = defs(s) == NEW_SEQ
    print(f"  {'OK  ' if ok else 'FAIL'} 定义序 = 1..9（实为 {defs(s)}）")
    if not ok:
        die("终检失败：Proposition 定义序不是 1..9")

    # 引用号必须落在 1..9，且九个号都被用到
    nums = sorted({int(x) for m in SPAN.finditer(s)
                   for x in re.findall(r"\d+", m.group(2))})
    ok = nums == list(range(1, 10))
    print(f"  {'OK  ' if ok else 'FAIL'} 被引用的编号集合 = 1..9（实为 {nums}）")
    if not ok:
        die(f"终检失败：引用编号集合异常 {nums}")

    # 悬空引用：不得出现 Proposition 0 或 10 以上
    bad = [n for n in re.findall(r"Propositions?\s+(\d+)", s) if not 1 <= int(n) <= 9]
    ok = not bad
    print(f"  {'OK  ' if ok else 'FAIL'} 无悬空编号（0 或 ≥10）：{bad or '无'}")
    if not ok:
        die(f"终检失败：出现悬空 Proposition 编号 {bad}")

    # 出现次数守恒（重编号不该增删任何一处提及）
    n_occ_after = len(SPAN.findall(s))
    ok = n_occ_after == n_occ_before
    print(f"  {'OK  ' if ok else 'FAIL'} 提及处数守恒 "
          f"{n_occ_before} → {n_occ_after}")
    if not ok:
        die(f"终检失败：提及处数由 {n_occ_before} 变成 {n_occ_after}")

    # 逐条抽查九个定义标签（标题随编号一起搬过，一个都不能错位）
    for probe, what in (
        (r"**Proposition 1 (semi-analytic form for $r(p)$).**", "§4.2 半解析形式"),
        (r"**Proposition 2 (probability-mass boundary; referred to below as the knee).**",
         "§4.3 概率质量边界"),
        (r"**Proposition 3 (noise-dominated lower-tail relation).**", "§5.1 下尾关系"),
        (r"**Proposition 4 (calibrated multiple).**", "§5.2 标定倍率"),
        (r"**Proposition 5 (scale ambiguity).**", "§6 尺度歧义"),
        (r"**Proposition 6 (exactness without noise).**", "§6 无噪声精确性"),
        (r"**Proposition 7 (null distribution).**", "§6 零分布"),
        (r"**Proposition 8 (contamination ratio).**", "§6 污染比"),
        (r"**Proposition 9 (first-order perturbation bound on the direction error).**",
         "§6 一阶扰动"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what} 的标签不在稿中")

    # §4.2/§4.3/§5.2 里带节号的指代必须跟着改
    for probe, what in (
        ("Proposition 1 (Section 4.2) uses", "§3.3 对 Prop 1 的前向指代"),
        ("Proposition 2 (Section 4.3", "Table 1 对 Prop 2 的指代"),
        ("Propositions 2–3 also delimit the method", "§5.3 的区间引用"),
        ("by Propositions 6 and 7 the conditional admission", "§6 Prop 8 的证明"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what} 未随之改号")

    # 结构不变：表题注 / 图题注 / 附录小节编号仍连续（不写死张数，只判连续无重复）
    caps = [int(x) for x in re.findall(r"(?m)^\*\*Table (\d+)\.", s)]
    figs = [int(x) for x in re.findall(r"(?m)^\*\*Fig\. (\d+)\.", s)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+)", s)]
    for name, seq2 in (("表题注", caps), ("图题注", figs), ("附录小节", secs)):
        ok = bool(seq2) and seq2 == list(range(1, len(seq2) + 1))
        print(f"  {'OK  ' if ok else 'FAIL'} {name}编号连续（共 {len(seq2)} 个）")
        if not ok:
            die(f"终检失败：{name}编号不连续或有重复")

    # 摘要词数（口径与 R16/R17/R19 同：只取摘要正文、不含标题行）
    body = s[: s.index("## References")]
    m_abs = re.search(r"## Abstract\s*\n+(.*?)\n+\s*\*\*Keywords:\*\*", body, re.S)
    if not m_abs:
        die("终检失败：抓不到摘要段（## Abstract ... **Keywords:**）")
    n_words = len(m_abs.group(1).split())
    ok = n_words <= 250
    print(f"  {'OK  ' if ok else 'FAIL'} 摘要 {n_words} 词（CSSP 上限 250）")
    if not ok:
        die(f"终检失败：摘要 {n_words} 词超过 250")

    print()
    print("R20 完成" if not dry else "R20 演练完成（未写盘）")


if __name__ == "__main__":
    main()