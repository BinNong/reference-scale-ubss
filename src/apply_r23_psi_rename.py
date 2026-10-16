r"""R23：把 §4.2–§5.1 的 CDF 家族从 $F$ 重命名为 $\Psi$（用户选定的符号冲突解法）。

背景（用户问"公式 (11) 中的 F 又是什么"）：公式 (11) 的 $F$ 是**逐点能量的混合
CDF**——$F(x)=\sum_J\pi_JF_J(x)$ 对活跃源数 $J$ 加权，$F_J$ 是条件 CDF，$F_0$ 是
纯噪声点的 CDF，后文还有 $F_s$（信号分量混合 CDF）、$F_{\Gamma(J,\theta)}$（Gamma 的
CDF）、$F_e$（$F_0$ 的小参数幂律形式）。但字母 $F$ 在稿中还有两个完全不同的含义：

  ① §7 协议与 Algorithm 1：$F=33$ 是**频点数目**（$FT$＝频点×帧数，$L=2FT$）；
  ② §A.12 掩码质量表：$F_1$ 是 **F1 score**。

用户已定夺：**重命名 CDF 一侧**（32 处，完整关在 §4.2–§5.1 这 85 行里），频点约定
$F=33$ 及其全部 $FT$ 用法不动（后者已扩散到全文约 10 处、算法规格与 Fig. 11 坐标轴）。

改动一（重命名，span 限定，六步有序替换）：
    span = 「### 4.2 A semi-analytic form」到「### 5.2 Calibrating the multiple」
    F_{\Gamma(J,\theta)} → \Psi_{\Gamma(J,\theta)}   （2 处）
    F_J                 → \Psi_J                     （8 处）
    F_0                 → \Psi_0                     （10 处）
    F_s                 → \Psi_s                     （5 处）
    F_e                 → \Psi_e                     （1 处）
    裸 F（正则）          → \Psi                       （6 处）
  ⚠️ 顺序必须先长后短；裸 F 的正则用 `(?<![A-Za-z0-9_])F(?![A-Za-z0-9_])`，
     它天然排除同段内的 "TF point"、"CDF"、"Fig."、"For"、"First"、"FT=2112"，
     以及 span 外的 $F_1$（F1 score）、$F=33$、$(F,T)$、参考文献里的作者姓氏。

改动二（就地指引，仿 R22）：公式 (11) 后原本只有 "and $r(p)$ is the solution of
$F(m_e)=1/2$..."，读者在这里第一次遇见符号却看不到它是什么。补成：
    …where $\Psi$ is the mixture CDF of the per-point energy ($\Psi_J$ conditional
    on exactly $J$ active sources, $\Psi_0$ noise-only), and $r(p)$ is the solution
    of $\Psi(m_e)=1/2$, divided by $\nu=M\sigma^2$.

改动三（挂回公式号）：Proposition 2 的 "Write $F=\sum_{J\ge0}\pi_JF_J$ for the
mixture CDF…" 补 "(Eq. 11)"，与全文既有的 Eq. 引用写法一致。

幂等：三步各自的 old 在改后即消失；`new` 均不含完整 `old`（护栏 `if old in new: die()`）。

用法：
    python3 src/apply_r23_psi_rename.py --dry
"""

import pathlib
import re
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

SPAN_HEAD = "### 4.2 A semi-analytic form"
SPAN_TAIL = "### 5.2 Calibrating the multiple"

# (旧, 新, 期望处数, 说明)；前六项在 span 内做，顺序必须保持
RENAMES = [
    (r"F_{\Gamma(J,\theta)}", r"\Psi_{\Gamma(J,\theta)}", 2, "Gamma CDF"),
    (r"F_J", r"\Psi_J", 8, "条件 CDF"),
    (r"F_0", r"\Psi_0", 10, "纯噪声 CDF"),
    (r"F_s", r"\Psi_s", 5, "信号分量混合 CDF"),
    (r"F_e", r"\Psi_e", 1, "小参数幂律形式"),
]
BARE_OLD_RE = re.compile(r"(?<![A-Za-z0-9_])F(?![A-Za-z0-9_])")
BARE_NEW = r"\Psi"
BARE_EXPECT = 6

# 改动二 / 三的锚点（改后态）。
# ⚠️ 指引句的锚点必须取到插入点**之前**：若只锚 "and $r(p)$ is the solution of …"，
#    它正是新串的后缀，第二次运行会再插一遍（护栏 `if old in new: die()` 已实测拦下）。
#    故锚点带上公式 (11) 的收尾 `$$` + 空行，新串以 "where" 开头，old 不再是 new 的子串。
# ⚠️ 这些串含真实换行，必须用**原始三引号**（raw 串里的 `\n` 是两个字，不会变成换行）。
GLOSS_OLD = r"""$$

and $r(p)$ is the solution of $\Psi(m_e)=1/2$, divided by $\nu=M\sigma^2$."""
# 新串在 `$$` + 空行之后保持**单行**（稿件惯例：一段一行；多行虽不影响 pandoc 合并，
# 但会破坏本文件的自洽性，也让按行分析的脚本多看一层）
GLOSS_NEW = (
    "$$\n\nwhere $\\Psi$ is the mixture CDF of the per-point energy ($\\Psi_J$ "
    "conditional on exactly $J$ active sources, $\\Psi_0$ noise-only), and $r(p)$ "
    "is the solution of $\\Psi(m_e)=1/2$, divided by $\\nu=M\\sigma^2$."
)
XREF_OLD = r"Write $\Psi=\sum_{J\ge0}\pi_J\Psi_J$ for the mixture CDF"
XREF_NEW = r"Write $\Psi=\sum_{J\ge0}\pi_J\Psi_J$ (Eq. 11) for the mixture CDF"

# 终检探针
P_PSI_FAMILY = r"\Psi_{2M}"          # 不应存在（防把 χ²_{2M} 之类的下标误改）
P_EQ11 = r"\Psi(x)\;=\;\sum_{J=0}^{N}\pi_J\,\Psi_J(x)"
P_GLOSS = r"$\Psi$ is the mixture CDF of the per-point energy"
P_XREF = r"(Eq. 11) for the mixture CDF"
P_FREQ = r"$F=33$"                    # 频点约定必须原样
P_FT = r"$FT=2112$"
P_F1 = r"$F_1$"


def die(msg: str) -> None:
    print("FAIL " + msg)
    sys.exit(1)


def sub_span(s: str, old: str, new: str, expect: int, tag: str, dry: bool) -> str:
    """span 内的字面替换；处数不符即失败（防锚点漂移）。"""
    i0 = s.index(SPAN_HEAD)
    i1 = s.index(SPAN_TAIL)
    head, span, tail = s[:i0], s[i0:i1], s[i1:]
    n = span.count(old)
    if n == 0:
        if new in span:
            print(f"跳过[{tag}]：改后文本已在稿中")
        else:
            die(f"未命中[{tag}]：span 内既无旧串也无新串")
        return s
    if n != expect:
        die(f"处数不符[{tag}]：期望 {expect}，实为 {n}（锚点可能已漂移，先核查再改）")
    span = span.replace(old, new)
    if not dry:
        print(f"已改[{tag}]：{n} 处")
    else:
        print(f"将改[{tag}]：{n} 处")
    return head + span + tail


def sub_bare(s: str, dry: bool) -> str:
    i0 = s.index(SPAN_HEAD)
    i1 = s.index(SPAN_TAIL)
    head, span, tail = s[:i0], s[i0:i1], s[i1:]
    n = len(BARE_OLD_RE.findall(span))
    if n == 0:
        if BARE_NEW in span:
            print("跳过[裸 F]：改后文本已在稿中")
        else:
            die("未命中[裸 F]：span 内既无旧串也无新串")
        return s
    if n != BARE_EXPECT:
        die(f"处数不符[裸 F]：期望 {BARE_EXPECT}，实为 {n}")
    # ⚠️ 替换串含反斜杠，必须用 lambda：re.sub 的 repl 是模板，`\P` 会被当转义报错
    span = BARE_OLD_RE.sub(lambda _m: BARE_NEW, span)
    if not dry:
        print(f"已改[裸 F]：{n} 处")
    else:
        print(f"将改[裸 F]：{n} 处")
    return head + span + tail


def swap(s: str, old: str, new: str, tag: str, dry: bool) -> str:
    if old in new:
        die(f"护栏[{tag}]：old 是 new 的前缀，会重复插入")
    n = s.count(old)
    if n == 0:
        if new in s:
            print(f"跳过[{tag}]：改后文本已在稿中")
        else:
            die(f"未命中[{tag}]：锚点既不是原状也不是改后状")
        return s
    if n > 1:
        die(f"锚点不唯一[{tag}]（{n} 处）")
    s = s.replace(old, new, 1)
    if not dry:
        MS.write_text(s)
        print(f"已改[{tag}]")
    else:
        print(f"将改[{tag}]")
    return s


def main() -> None:
    dry = "--dry" in sys.argv
    s = MS.read_text()

    if SPAN_HEAD not in s or SPAN_TAIL not in s:
        die("找不到 span 边界标题（§4.2 / §5.2）")

    # ---- 改动一：span 内有序重命名 ----
    for old, new, expect, tag in RENAMES:
        s = sub_span(s, old, new, expect, tag, dry)
    s = sub_bare(s, dry)

    # ---- 改动二、三：就地指引 + 公式号回引（依赖改后态，故在重命名之后）----
    s = swap(s, GLOSS_OLD, GLOSS_NEW, "公式 (11) 后就地指引", dry)
    s = swap(s, XREF_OLD, XREF_NEW, "Prop 2 回引 Eq. 11", dry)

    # ---- 终检 ----
    print()
    for probe, what in (
        (P_EQ11, "公式 (11) 本体已改为 Ψ 家族"),
        (P_GLOSS, "就地指引子句"),
        (P_XREF, "Proposition 2 回引 Eq. 11"),
        (P_FREQ, "频点约定 $F=33$ 原样未动"),
        (P_FT, "$FT=2112$ 原样未动"),
        (r"\mathbf{G}", "Gram 矩阵记号未动（本次不碰 G）"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what}")

    # span 内不得再有 F 家族；span 外的频点/F1 用法必须健在
    i0, i1 = s.index(SPAN_HEAD), s.index(SPAN_TAIL)
    span = s[i0:i1]
    left = re.findall(r"F_\{J\}|F_J|F_0|F_s|F_e|F_\{\\Gamma|(?<![A-Za-z0-9_])F(?![A-Za-z0-9_])", span)
    ok = not left
    print(f"  {'OK  ' if ok else 'FAIL'} span 内 F 家族零残留（实为 {len(left)}）")
    if not ok:
        die(f"终检失败：span 内仍有 F 家族 {left[:5]}")

    for probe, what in (
        (r"$F_1$", "F1 score 记号健在"),
        (r"C^{M x F x T}", "Algorithm 1 输入规格的 F 健在"),
        (r"{0,1}^{F x T}", "Algorithm 1 输出规格的 F 健在"),
        (r"$(F,T)$", "§A.27 的 (F,T) 健在"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what}不在稿中")

    ok = P_PSI_FAMILY not in s
    print(f"  {'OK  ' if ok else 'FAIL'} 未产生 \\Psi_{{2M}} 这类误改下标")
    if not ok:
        die("终检失败：出现 \\Psi_{2M}")

    # 公式总数不变（未新增 $$ 对）
    n_eq = len(re.findall(r"\$\$", s)) // 2
    ok = n_eq == 20
    print(f"  {'OK  ' if ok else 'FAIL'} 公式仍为 20 个（实为 {n_eq}）")
    if not ok:
        die(f"终检失败：公式数变为 {n_eq}")

    print()
    print("R23 完成" if not dry else "R23 演练完成（未写盘）")


if __name__ == "__main__":
    main()
