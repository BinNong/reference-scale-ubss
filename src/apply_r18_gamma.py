r"""第十八轮：把 §5.1 结构无关门限的标量 $T$ 改名为 $\gamma$。

**为什么改**：同一个字母 $T$ 在稿中表示两个不同的量——

  * §5.1（L135）：结构无关门限的标量，"A structureless gate is a scalar $T$
    and the test is $e_{f,t}>T$"；
  * §7（L391）与 §A.27（L1618）：**时间帧的数目**，"$F=33$, $T=64$"、
    "taking $(F,T)$ from $(33,64)$ to ..."。

同名不同义是审稿人一抓一个准的硬伤，且 §7 的 $T$ 定义在后面，读者在 §5.1 先见到
门限的 $T$，会把两者当成一回事。

**为什么选 $\gamma$**：先查了全部已占用的数学符号——
  * $\tau$ **已被本文核心量占用**（标定倍率 $\tau(M,\alpha)=Q_{1-\alpha}(\chi^2_{2M})/2M$，
    门限写成 $\tau\hat\nu$，全文 13 处）；
  * $\theta$ 已被小参数展开 $F_{\Gamma(J,\theta)}$ 与 Gram 条件数占用；
  * $\lambda$ 已被附录 A.19 的 Gram 谱占用；
  * $\alpha$、$\eta$、$\rho$、$\kappa$、$\nu$、$\sigma$、$\pi$、$q$ 均已占用。
$\gamma$ 全文零出现，且符合本文"希腊字母表标量、拉丁字母表索引/计数"的惯例
（$f,t$ 索引，$F,T,M,N,J$ 计数，$\tau,\nu,\sigma,\alpha$ 标量）。

**改动范围（已逐处核实）**：仅 L135 一行、共 3 处。后文立刻把门限改写成
$t_e$（ practitioner 的选择）与 $\tau$（标定倍率），故 $T$ 不外溢。
无脚本断言 `$T$`，无图例含该符号（11 张矢量图已逐张 pdftotext 核查），不必重生成图。

**刻意不做的**：不顺手把 $F$/$T$ 的"数目"含义在 §3.1 补定义——因为发现 **$F$ 也撞车**
（§4 的能量 CDF $F,F_0,F_J,F_s,F_e$ 是理论核心、约 31 处；§7 的 $F=33$ 是频点数）。
那一处的修法（改 CDF 的字母还是改计数的字母）是判断题，留给作者定夺，见当日记忆。

幂等：两条 old 都不是各自 new 的子串，重跑走"跳过"分支。

用法：
    python3 src/apply_r18_gamma.py --dry
    python3 src/apply_r18_gamma.py
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

E = [
    # 门限的定义句 + 判据（一句话覆盖 2 处）
    ("§5.1 门限标量定义与判据",
     r"A structureless gate is a scalar $T$ and the test is $e_{f,t}>T$.",
     r"A structureless gate is a scalar $\gamma$ and the test is $e_{f,t}>\gamma$."),
    # "任何实用选择"那句（第 3 处）
    ("§5.1 任何实用选择那句",
     r"*Any* practical choice of $T$ is expressed relative to",
     r"*Any* practical choice of $\gamma$ is expressed relative to"),
]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只演练，不写盘")
    a = ap.parse_args()

    s = MS.read_text()
    n_changed = n_skipped = 0

    for tag, old, new in E:
        n = s.count(old)
        if n == 0:
            if new in s:
                print(f"  –  {tag}：已完成（new 已在）")
                n_skipped += 1
            else:
                die(f"{tag}：old 不在稿中")
            continue
        if n > 1:
            die(f"{tag}：old 出现 {n} 次，锚点不唯一")
        if old in new:
            die(f"{tag}：old 是 new 的子串，会重复插入（原子性陷阱）")
        s = s.replace(old, new, 1)
        print(f"  ✓  {tag}")
        n_changed += 1

    if not a.dry and n_changed:
        MS.write_text(s)

    print(f"\n{'演练' if a.dry else '应用'}完成：已改 {n_changed} | 跳过 {n_skipped} | 未命中 0")

    # ---- 终检：断言改后的最新状态（校验内存中的 s，dry 不写盘）----
    t = s

    # 1) 门限已是 $\gamma$，且全稿再无 $T$ 作门限
    # 注意：3 处里只有 2 处是独立的 $\gamma$，第三处写在 $e_{f,t}>\gamma$ 里，
    #      故必须数 \gamma 本身，不能数 $\gamma$（第一版就这么误报过）。
    n_gamma = t.count(r"\gamma")
    ok = n_gamma == 3
    print(rf"  {'OK  ' if ok else 'FAIL'} 门限 $\gamma$ 共出现 3 次（实为 {n_gamma}）")
    if not ok:
        die(rf"终检失败：门限 \gamma 的出现次数不是 3（实为 {n_gamma}）")

    # 2) 全稿不再有裸 $T$（§7 的 $T=64$ 与 $(F,T)$ 不含 $T$ 子串，故此项可作判据）
    n_T = t.count("$T$")
    ok = n_T == 0
    print(f"  {'OK  ' if ok else 'FAIL'} 全文无裸 $T$（实为 {n_T}）")
    if not ok:
        die("终检失败：仍有裸 $T$")

    # 3) 帧数 $T=64$ 与 $(F,T)$ 未被误伤（这正是要保留的用法）
    for probe, what in ((r"$F=33$, $T=64$", "§7 的频点/帧数定义"),
                        (r"taking $(F,T)$ from $(33,64)$", "§A.27 的 (F,T) 扫描")):
        ok = probe in t
        print(f"  {'OK  ' if ok else 'FAIL'} {what}未受影响")
        if not ok:
            die(f"终检失败：{what}被误伤")

    # 4) 核心量 $\tau$ 一字未动（标定倍率，13 处）
    n_tau = len(re.findall(r"\\tau\b", t))
    ok = n_tau >= 13
    print(rf"  {'OK  ' if ok else 'FAIL'} 标定倍率 $\tau$ 仍有 {n_tau} 处（≥13）")
    if not ok:
        die(r"终检失败：$\tau$ 被误伤")

    # 5) 公式 (1)–(20) 与摘要词数不受影响
    n_eq = len(re.findall(r"(?m)^[ \t]*\$\$[ \t]*$", t)) // 2
    ok = n_eq == 20
    print(f"  {'OK  ' if ok else 'FAIL'} 显示公式仍为 20 个（实为 {n_eq}）")
    if not ok:
        die("终检失败：显示公式个数变了")
    body = t[:t.index("## References")]
    m_abs = re.search(r"## Abstract\s*\n+(.*?)\n+\s*\*\*Keywords:\*\*", body, re.S)
    if not m_abs:
        die("终检失败：解析摘要失败")
    n_words = len(m_abs.group(1).split())
    ok = n_words <= 250
    print(f"  {'OK  ' if ok else 'FAIL'} 摘要 {n_words} 词（CSSP 上限 250）")
    if not ok:
        die(f"终检失败：摘要 {n_words} 词超过 250")

    # 6) 公式引用仍无悬空（R16 的判据不能被本轮破坏）
    ref_pat = re.compile(
        r"\b(?:Eqs?\.|[Ee]quations?)\s*\(?\s*\d{1,2}"
        r"(?:\s*\)?\s*(?:[\u2013,\-]|and)\s*\(?\s*\d{1,2})*\)?")
    cited = set()
    for m2 in ref_pat.finditer(t):
        cited.update(int(x) for x in re.findall(r"\d{1,2}", m2.group(0)))
    over = sorted(n for n in cited if not 1 <= n <= n_eq)
    ok = not over
    print(f"  {'OK  ' if ok else 'FAIL'} 公式引用无悬空（引用号 {sorted(cited)}）")
    if not ok:
        die(f"终检失败：引用号 {over} 超出 1..{n_eq}")

    print("\n终检全部通过")


if __name__ == "__main__":
    main()
