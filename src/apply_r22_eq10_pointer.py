r"""R22：在公式 (10) 处就地给裸符号 $e$ 一个指引（用户翻到公式 (10) 没看到解释）。

背景：R21 把裸 $e$ 的简写声明加在了 **§4.1 的定义句**（成品第 7 页），而公式 (10)
在 **Proposition 1** 里（成品第 8 页）。用户读 PDF 时直奔公式 (10)，翻到那一页
"没有变化"——**解释确实存在，但不在读者遇见符号的地方**。

这不是 R21 写错了位置（定义就该在定义处），而是长距离复用符号时缺一个就地指引：
Proposition 1 与 §4.1 之间隔着一个公式 (9)、一整段论证和一个小节标题。

改动（一句，公式本体与命题陈述一字不动）：把 Proposition 1 公式后的 where 从句
从"两个性质并列"改成"两个性质 + 符号出处"——

    where the $\chi^2$ and Gamma variates are independent and the second term
    is absent for $J=0$.
→
    where the $\chi^2$ and Gamma variates are independent, $e$ is the per-point
    energy of Section 4.1, and the second term is absent for $J=0$.

幂等：old 取到插入点之后的 "and the second term"，old 不是 new 的前缀；
护栏 `if old in new: die()` 兜底。

用法：
    python3 src/apply_r22_eq10_pointer.py --dry
"""

import pathlib
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

OLD = (
    r"where the $\chi^2$ and Gamma variates are independent and the second term "
    r"is absent for $J=0$."
)
NEW = (
    r"where the $\chi^2$ and Gamma variates are independent, $e$ is the per-point "
    r"energy of Section 4.1, and the second term is absent for $J=0$."
)

# 终检探针（字面子串）
P_POINTER = r"$e$ is the per-point energy of Section 4.1"
P_EQ10 = r"e \;=\; \frac{\sigma^{2}}{2}\,\chi^{2}_{2M}"
P_DEF_R21 = r"we drop the subscript and write $e$, and let"
P_GAMMA_CLAUSE = r"\Gamma\!\Big(J,\ \frac{M\sigma^{2}}{Np}\,10^{\mathrm{SNR}/10}\Big),"


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
            print("跳过：指引子句已在稿中（R22 已应用）")
        else:
            die("未命中：Proposition 1 的 where 从句既不是原状也不是改后状")
    elif n > 1:
        die(f"锚点不唯一（{n} 处）")
    else:
        s = s.replace(OLD, NEW, 1)
        if not dry:
            MS.write_text(s)
            print("已改：公式 (10) 后补上 $e$ 的就地指引")
        else:
            print("将改：公式 (10) 后补上 $e$ 的就地指引")

    print()
    for probe, what in (
        (P_POINTER, "就地指引子句（$e$ = §4.1 的逐点能量）"),
        (P_EQ10, "公式 (10) 本体未被改动"),
        (P_GAMMA_CLAUSE, "公式 (10) 的 Gamma 项未被改动"),
        (P_DEF_R21, "R21 在 §4.1 的定义句仍在（两处互补，不互相替代）"),
        (r"**Proposition 1 (semi-analytic form for $r(p)$).**", "Proposition 1 标题行未动"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what}")

    dbl = s.count(P_POINTER)
    ok = dbl == 1
    print(f"  {'OK  ' if ok else 'FAIL'} 指引子句只出现 1 次（实为 {dbl}）")
    if not ok:
        die("终检失败：指引子句重复出现")

    print()
    print("R22 完成" if not dry else "R22 演练完成（未写盘）")


if __name__ == "__main__":
    main()
