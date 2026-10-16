r"""R24：把公式 (5) 缺掉的推导步补上（用户：公式 (5) 我不是很明白怎么来的）。

背景：公式 (5) 是 §3.3 的"本地 SNR"关系

    $$\mathrm{SNR}_{\text{local}}=\mathrm{SNR}-10\log_{10}\big(1-(1-p)^N\big)\ \ \text{dB}$$

稿中原来的写法是**直接断言**："The mean signal energy at a point where at least one
source is active is a factor $1/P(\text{act})$ above the plane average, where
$P(\text{act})=1-(1-p)^N$, so …"——跳过了唯一需要解释的一步：**为什么**活跃点上的
均值能量是平面平均的 $1/P(\text{act})$ 倍。作者本人读到这里也没能跟上，审稿人更会卡。

缺的步其实一句话：平面平均把**总信号功率摊到全部 $FT$ 个点上**，而信号只占其中
$P(\text{act})=1-(1-p)^N$ 那一部分（$N$ 个源各自独立、概率 $p$，空点概率 $(1-p)^N$），
所以承载信号的点上，均值能量 = 平面平均 ÷ 该占比；噪声处处都在、不参与缩放。

改动（只重写这一句，公式本体一字不动）：

    The mean signal energy at a point where at least one source is active is a
    factor $1/P(\text{act})$ above the plane average, where $P(\text{act})=1-(1-p)^N$, so
→
    The plane average spreads the total signal power evenly over all $FT$ points,
    while the sources occupy only the fraction $P(\text{act})=1-(1-p)^N$ of them —
    each of the $N$ sources being independently active with probability $p$, a point
    is empty with probability $(1-p)^N$ — so the mean signal energy at a point that
    carries at least one source is the plane average divided by that fraction, a
    factor $1/P(\text{act})$ above it. The noise is present at every point and is not
    rescaled, so

后文原有的"— the factor $M$ cancels, because the plane average and the local value
are both sums over the same $M$ components"与数值例（$p=0.02$、$N=4$、20 dB → 31.1 dB）
都保持不变，且与新句衔接自然。

幂等：old 锚在要被替换的整句上，new 换了措辞，old 不是 new 的子串（护栏会验）。
⚠️ 不能把锚点取成 "where the average in the numerator is taken over all $FT$ TF points
and all $M$ components. This convention matters for what follows. The mean signal
energy…"——那句的前半段在新串里原样保留，old 会成为 new 的前缀而被护栏拦下。

连带登记：`apply_referee_fixes.py` 的 "M10a Prop.6 的混合密度：P(act) → Np" 一条的
改后态正是被本句替换掉的文本，须登记 `_SUPERSEDED_R24`，否则它会把一次"跳过"变成
"未命中"（该脚本的判据是有未命中就不写入）。

用法：
    python3 src/apply_r24_snr_derivation.py --dry
"""

import pathlib
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

OLD = (
    "The mean signal energy at a point where at least one source is active is a "
    r"factor $1/P(\text{act})$ above the plane average, where $P(\text{act})=1-(1-p)^N$, so"
)
NEW = (
    "The plane average spreads the total signal power evenly over all $FT$ points, "
    r"while the sources occupy only the fraction $P(\text{act})=1-(1-p)^N$ of them "
    r"— each of the $N$ sources being independently active with probability $p$, a "
    r"point is empty with probability $(1-p)^N$ — so the mean signal energy at a "
    r"point that carries at least one source is the plane average divided by that "
    r"fraction, a factor $1/P(\text{act})$ above it. The noise is present at every "
    "point and is not rescaled, so"
)

# 终检探针（字面子串）
P_DERIV = r"The plane average spreads the total signal power evenly over all $FT$ points"
P_PACT = r"$P(\text{act})=1-(1-p)^N$"
P_EMPTY = r"a point is empty with probability $(1-p)^N$"
P_FACTOR = r"a factor $1/P(\text{act})$ above it"
P_NOISE = r"The noise is present at every point and is not rescaled, so"
P_EQ5 = r"\mathrm{SNR}_{\text{local}}=\mathrm{SNR}-10\log_{10}\big(1-(1-p)^N\big)"
P_MCANCEL = r"— the factor $M$ cancels, because the plane average and the local value"
P_NUM = r"At $p=0.02$, $N=4$ and $\mathrm{SNR}=20$ dB this is $31.1$ dB"
P_ONESOURCE = r"carries on average $1/(Np)$ of the plane average rather than $1/P(\text{act})$"


def die(msg: str) -> None:
    print("FAIL " + msg)
    sys.exit(1)


def main() -> None:
    dry = "--dry" in sys.argv
    s = MS.read_text()

    if OLD in NEW:
        die("护栏：old 是 new 的前缀，会重复插入")

    n = s.count(OLD)
    if n == 0:
        if NEW in s:
            print("跳过：推导句已在稿中（R24 已应用）")
        else:
            die("未命中：§3.3 那句既不是原状也不是改后状")
    elif n > 1:
        die(f"锚点不唯一（{n} 处）")
    else:
        s = s.replace(OLD, NEW, 1)
        if not dry:
            MS.write_text(s)
            print("已改：§3.3 补上公式 (5) 的推导步")
        else:
            print("将改：§3.3 补上公式 (5) 的推导步")

    print()
    for probe, what in (
        (P_DERIV, "推导步：平面平均把总功率摊到全部 FT 个点"),
        (P_PACT, "P(act) 的定义仍在"),
        (P_EMPTY, "空点概率 (1-p)^N 的来源（独立性）"),
        (P_FACTOR, "结论：活跃点均值 = 平面平均 ÷ 占比"),
        (P_NOISE, "噪声不参与缩放（公式 (5) 只改信号项）"),
        (P_EQ5, "公式 (5) 本体未被改动"),
        (P_MCANCEL, "M 相消的说明句未动"),
        (P_NUM, "数值例 31.1 dB 未动"),
        (P_ONESOURCE, "恰好一个源情形的 1/(Np) 对比未动"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what}")

    dbl = s.count(P_DERIV)
    ok = dbl == 1
    print(f"  {'OK  ' if ok else 'FAIL'} 推导句只出现 1 次（实为 {dbl}）")
    if not ok:
        die("终检失败：推导句重复出现")

    ok = "The mean signal energy at a point where at least one source is active is a" not in s
    print(f"  {'OK  ' if ok else 'FAIL'} 旧的断言式写法已无残留")
    if not ok:
        die("终检失败：旧句子仍在稿中")

    # 公式总数不变
    n_eq = len([1 for i in range(0, len(s) - 1) if s[i:i + 2] == "$$"]) // 2
    ok = n_eq == 20
    print(f"  {'OK  ' if ok else 'FAIL'} 公式仍为 20 个（实为 {n_eq}）")
    if not ok:
        die(f"终检失败：公式数变为 {n_eq}")

    print()
    print("R24 完成" if not dry else "R24 演练完成（未写盘）")


if __name__ == "__main__":
    main()
