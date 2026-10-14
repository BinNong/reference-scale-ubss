r"""R19：给公式 (6) 依赖的 $S_n$ 补上定义（第 $n$ 个源在 TF 点的复振幅）。

背景：§3.4 的 participation ratio 写作

    PR(f,t) = (\sum_n e_n)^2 / \sum_n e_n^2,   e_n = |S_n(f,t)|^2,

其中 $e_n$ 就地定义清楚（第 $n$ 个源的能量），但它把读者推给 $S_n(f,t)$ ——
而这个记号**全稿只出现这一次**，从未被引入。§3.1 引入的是粗体小写
$\mathbf{s}(f,t)\in\mathbb{C}^N$（源向量），§3.2 用的是粗体大写
$\mathbf{S}=\mathbf{S}_r+j\mathbf{S}_i$（源矩阵的实/虚部），$S_n$ 是
**非粗体大写**的第三种形态；§3.3 只用文字 "source $n$ is active" 说过激活，
没把文字与记号挂上。⇒ 与 R17（$f$/$t$）同一类缺口：符号就地用了，来源没交代。

改动（一句，公式本体一字不动）：在 §3.3 引入激活概率的那句里，把 $S_n$
显式挂到 $\mathbf{s}$ 上 ——

    ... with complex Gaussian amplitude; the sparsity of the sources ...
→
    ... with complex Gaussian amplitude, writing $S_n(f,t)$ for the $n$-th
    component of $\mathbf{s}(f,t)$; the sparsity of the sources ...

幂等：old 取到插入点**之后**的 "the sparsity"，故 old 不是 new 的前缀
（若 old 只到 "amplitude;"，它正是新串前缀，第二次运行会再插一遍 ——
本脚本护栏 `if old in new: die()` 会当场拦下）。

用法：
    python3 src/apply_r19_source_component.py --dry
"""

import pathlib
import re
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

OLD = (
    "At each TF point, source $n$ is active independently with probability $p$, "
    "with complex Gaussian amplitude; the sparsity"
)
NEW = (
    "At each TF point, source $n$ is active independently with probability $p$, "
    "with complex Gaussian amplitude, writing $S_n(f,t)$ for the $n$-th component "
    r"of $\mathbf{s}(f,t)$; the sparsity"
)

# 终检用的探针（一律字面子串，不用正则 —— `$` 在正则里是行尾锚点）
P_FORMULA6 = r"e_n=|S_n(f,t)|^2"          # 公式 (6) 本体必须原样还在
P_DEF = r"writing $S_n(f,t)$ for the $n$-th component of $\mathbf{s}(f,t)$"
P_NOT_DOUBLED = "writing $S_n(f,t)$ for the $n$-th component of $\\mathbf{s}(f,t)$; the sparsity of the sources are"


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
            print("跳过：定义子句已在稿中（R19 已应用）")
        else:
            die("未命中：§3.3 那句既不是原状也不是改后状")
    elif n > 1:
        die(f"锚点不唯一（{n} 处）")
    else:
        # 无论 dry 与否都在内存里应用，终检验的是"将要成为"的状态（R17 踩过：
        # dry 时不在内存里替换，终检读磁盘，报假的 FAIL）。
        s = s.replace(OLD, NEW, 1)
        if not dry:
            MS.write_text(s)
        print("已改：§3.3 补上 $S_n$ 的定义子句")

    # ---- 终检：必须断言改后的最新状态（dry 时校验内存中的 s）----
    print()
    for probe, what in (
        (P_DEF, "定义子句（$S_n$ = $\\mathbf{s}$ 的第 $n$ 个分量）"),
        (P_FORMULA6, "公式 (6) 本体未被改动"),
        ("participation ratio", "§3.4 的 participation ratio 标题段"),
        (r"$\mathrm{PR}=1$ under exact WDO", "PR=1 ↔ exact WDO 的解释句"),
    ):
        ok = probe in s
        print(f"  {'OK  ' if ok else 'FAIL'} {what}")
        if not ok:
            die(f"终检失败：{what} 不在稿中")

    # 不得被插了两次
    dbl = s.count(P_DEF)
    ok = dbl == 1
    print(f"  {'OK  ' if ok else 'FAIL'} 定义子句只出现 1 次（实为 {dbl}）")
    if not ok:
        die("终检失败：定义子句重复出现")

    # 摘要词数（口径必须与 build_pdf_cssp 一致：只取摘要正文、不含标题行。
    #   早先我用 body.index("## Abstract") 起算，把标题行的 "##" 与 "Abstract"
    #   也数进去，虚高 2 词报 252 —— 与 R17 踩过的是同一个坑。）
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
    print("R19 完成" if not dry else "R19 演练完成（未写盘）")


if __name__ == "__main__":
    main()
