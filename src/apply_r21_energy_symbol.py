r"""R21：给公式 (10) 里的裸符号 $e$ 补上定义（逐点能量的省略下标写法）。

背景（用户读 PDF 时发现）：公式 (10) 写作

    $$e = \frac{\sigma^2}{2}\chi^2_{2M} + \Gamma(J, \frac{M\sigma^2}{Np}10^{SNR/10}),$$

其中 $e$ 之前**从未以裸符号出现过**——§4.1 只定义了带下标的
$e_{f,t}=\|\mathbf{x}(f,t)\|^2$。但稿中裸 $e$ 其实早已在用，只是没人说过它就是
$e_{f,t}$：

    §4.1 正文   $m_e=\mathrm{median}(e)$ / $e_{\max}$      （第 135 行）
    公式 (10)    裸 $e$（本回合的引子）                    （第 155 行）
    Table 1 题注 $r(p)=\mathrm{median}(e)/\nu$            （第 188 行）
    Prop. 4      $e>\tau\nu$                               （第 268 行）
    Table 14 表头、Table 30 题注、Table 43 题注、Fig. 1 题注

⇒ 与 R17（$f$/$t$）、R19（$S_n$）同一类缺口：符号就地用了，来源没交代。

改动一（主改动，一句，公式本体一字不动）：§4.1 定义 $e_{f,t}$ 的那句里
声明"不特指某个点时省略下标写作 $e$"：

    Let the **per-point energy** be $e_{f,t}=\|\mathbf{x}(f,t)\|^2$, and let
→
    Let the **per-point energy** be $e_{f,t}=\|\mathbf{x}(f,t)\|^2$; where the
    point is not singled out we drop the subscript and write $e$, and let

改动二（消歧，一处）：§7 Baselines 里唯一一个非能量的 $e$ —— 欧拉数
`$e^{-0.02}=98\%$`。$e$ 一经成为已定义的逐点能量符号，这个写法就成了真正
的符号冲突（全稿仅此一处用欧拉数，改成 `\exp(-0.02)` 语义不变）：

    since $e^{-0.02}=98\%$ of noise-only points survive it
→
    since $\exp(-0.02)=98\%$ of noise-only points survive it

幂等：两处 old 都取到插入点之后（", and let" / "since"），old 均不是 new 的
前缀；护栏 `if old in new: die()` 兜底。

用法：
    python3 src/apply_r21_energy_symbol.py --dry
"""

import pathlib
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

E1_OLD = r"be $e_{f,t}=\|\mathbf{x}(f,t)\|^2$, and let"
E1_NEW = (
    r"be $e_{f,t}=\|\mathbf{x}(f,t)\|^2$; where the point is not singled out "
    r"we drop the subscript and write $e$, and let"
)

E2_OLD = r"since $e^{-0.02}=98\%$ of noise-only points survive it"
E2_NEW = r"since $\exp(-0.02)=98\%$ of noise-only points survive it"

# 终检探针（字面子串；`$` 在正则里是行尾锚点，故一律不用正则）
P_DEF = r"we drop the subscript and write $e$, and let"
P_EQ10 = r"e \;=\; \frac{\sigma^{2}}{2}\,\chi^{2}_{2M}"      # 公式 (10) 本体原样还在
P_EQ9 = r"e_{f,t} > \underbrace{t_e\, r(p)}_{\text{effective multiple}}"
P_PROP4 = r"$e>\tau\nu$ is equivalent to"
P_EULER_GONE = r"since $e^{-0.02}=98\%$"
P_EXP_NEW = r"since $\exp(-0.02)=98\%$"


def die(msg: str) -> None:
    print("FAIL " + msg)
    sys.exit(1)


def apply_one(s: str, tag: str, old: str, new: str, dry: bool) -> str:
    if old in new:
        die(f"护栏[{tag}]：old 是 new 的前缀，会重复插入")
    n = s.count(old)
    if n == 0:
        if new in s:
            print(f"跳过[{tag}]：改后文本已在稿中（R21 已应用）")
        else:
            die(f"未命中[{tag}]：锚点既不是原状也不是改后状")
    elif n > 1:
        die(f"锚点不唯一[{tag}]（{n} 处）")
    else:
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

    s = apply_one(s, "E1 定义裸 e", E1_OLD, E1_NEW, dry)
    s = apply_one(s, "E2 欧拉数消歧", E2_OLD, E2_NEW, dry)

    # ---- 终检：断言改后的最新状态（dry 时校验内存中的 s）----
    print()
    for probe, what in (
        (P_DEF, "§4.1 的裸 $e$ 定义子句"),
        (P_EQ10, "公式 (10) 本体未被改动"),
        (P_EQ9, "公式 (9) 本体未被改动"),
        (P_PROP4, "Proposition 4 的 $e>\\tau\\nu$ 未被改动"),
        (P_EXP_NEW, "§7 欧拉数写作 $\\exp(-0.02)$"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what}")

    dbl = s.count(P_DEF)
    ok = dbl == 1
    print(f"  {'OK  ' if ok else 'FAIL'} 定义子句只出现 1 次（实为 {dbl}）")
    if not ok:
        die("终检失败：定义子句重复出现")

    eul = s.count(P_EULER_GONE)
    ok = eul == 0
    print(f"  {'OK  ' if ok else 'FAIL'} 欧拉数裸写 $e^{{-0.02}}$ 已无残留（实为 {eul}）")
    if not ok:
        die("终检失败：欧拉数裸写仍在稿中")

    # 裸 $e$ 现在必须有定义来源：定义句 + 公式 (9)/(10) + Prop. 4 同时在场
    for probe, what in (
        (r"$e_{f,t}=\|\mathbf{x}(f,t)\|^2$", "带下标的 $e_{f,t}$ 定义仍在"),
        (r"$m_e=\mathrm{median}(e)$", "§4.1 的 median(e) 简写仍在"),
        (r"$r(p)=\mathrm{median}(e)/\nu$", "Table 1 题注的 median(e) 仍在"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what}不在稿中")

    print()
    print("R21 完成" if not dry else "R21 演练完成（未写盘）")


if __name__ == "__main__":
    main()
