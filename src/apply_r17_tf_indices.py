r"""第十七轮：把公式 (1) 里从未定义的索引 $f$、$t$ 显式写清楚。

**问题（用户提出）**：§3.1 开头只写 "In the TF domain," 就给出

    x(f,t) = A s(f,t) + n(f,t)

其中 $f$ 与 $t$ **在全稿从未被定义**。后文只用 "TF point" 统称，读者只能自己猜。
$f$/$t$ 另见于 §3.3（PR(f,t)）、§4.1（e_{f,t}）与 Algorithm 1 的伪代码行。

**改法**：在首次出现处（公式 (1) 之前）补半句定义，后文沿用，不再重复。

⚠️ **刻意不引入 $T$ 作为"帧数"**：$T$ 在 §5.1 已被用作**结构无关门限的标量**
（"A structureless gate is a scalar $T$ and the test is $e_{f,t}>T$"）。
同名不同义是审稿人一抓一个准的硬伤，故本脚本只定义索引本身，不定义 $F\times T$。

    （顺带发现、**本脚本不动**：§3.3 与 Fig. 11 题注用了 "$FT$" 表示总点数，
     即 $F$ 与 $T$ 的乘积，而两者同样未定义——且 $T$ 与门限标量撞名。
     修它要决定"改名叫别的字母"还是"重新命名门限标量"，属判断题，留待作者定夺。）

幂等：old 是句中前缀，new 只在其后追加，old 不是 new 的子串；重跑走"跳过"分支。

用法：
    python3 src/apply_r17_tf_indices.py --dry
    python3 src/apply_r17_tf_indices.py
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"

E = [
    # ⚠️ old 必须跨过插入点（带上后面的 "\n\n$$"）：插入点在 "In the TF domain," 之后，
    #    若 old 只到逗号，它就是 new 的前缀，第二次运行会再插一遍（幂等失效）——
    #    本项目踩过多次的原子性陷阱，故 new 里 "frames," 之后必须紧跟原文的 "\n\n$$"。
    ("§3.1 定义 TF 索引 f、t",
     "In the TF domain,\n\n$$",
     "In the TF domain, where $f$ indexes frequency bins and $t$ indexes time "
     "frames,\n\n$$"),
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
                die(f"{tag}：old 不在稿中（{old!r}）")
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

    # ---- 终检：必须断言改后的最新状态 ----
    # ⚠️ 校验内存里的 `s`（将要成为的状态），不是重读磁盘——dry 运行不写盘，
    #    读磁盘会拿到改前状态而假失败（本轮就这么错过一次）。
    t = s

    # 1) 定义确实在公式 (1) 之前
    i_def = t.find("where $f$ indexes frequency bins and $t$ indexes time frames")
    i_eq = t.find("\\mathbf{x}(f,t)=\\mathbf{A}")
    ok = 0 < i_def < i_eq
    print(f"  {'OK  ' if ok else 'FAIL'} 定义位于公式 (1) 之前（L{t[:i_def].count(chr(10))+1} < L{t[:i_eq].count(chr(10))+1}）")
    if not ok:
        die("终检失败：定义不在公式 (1) 之前")

    # 2) 全稿只定义一次（重复定义是另一种噪声）
    n_def = t.count("indexes frequency bins")
    ok = n_def == 1
    print(f"  {'OK  ' if ok else 'FAIL'} 索引定义只出现一次（{n_def} 次）")
    if not ok:
        die("终检失败：索引定义重复出现")

    # 3) 公式 (1) 本体一字未动
    ok = "\\mathbf{x}(f,t)=\\mathbf{A}\\,\\mathbf{s}(f,t)+\\mathbf{n}(f,t)" in t
    print(f"  {'OK  ' if ok else 'FAIL'} 公式 (1) 本体未改动")
    if not ok:
        die("终检失败：公式 (1) 本体被动了")

    # 4) $T$ 仍只作门限标量用（不得被本脚本引入成帧数）
    m = re.search(r"A structureless gate is a scalar \$T\$", t)
    ok = m is not None
    print(f"  {'OK  ' if ok else 'FAIL'} §5.1 的门限标量 $T$ 未受影响")
    if not ok:
        die("终检失败：门限标量 $T$ 的句子不见了")

    # 5) 摘要词数（口径与 build_pdf_cssp.py 完全一致：摘要正文、不含标题行，
    #    且不做 markdown 字符剥离——用 `len(d["abstract"].split())` 同一算法。
    #    早先我按 `body.index("## Abstract")` 起算，把标题行的 "##" 与 "Abstract"
    #    也数进去，虚高 2 词；那是我的 bug，不是稿件的问题。)
    body = t[:t.index("## References")]
    m_abs = re.search(r"## Abstract\s*\n+(.*?)\n+\s*\*\*Keywords:\*\*", body, re.S)
    if not m_abs:
        die("终检失败：解析摘要失败")
    n_words = len(m_abs.group(1).split())
    ok = n_words <= 250
    print(f"  {'OK  ' if ok else 'FAIL'} 摘要 {n_words} 词（CSSP 上限 250）")
    if not ok:
        die(f"终检失败：摘要 {n_words} 词超过 250")

    # 6) 公式引用仍无悬空（上一轮 R16 的判据，不能被本轮破坏）
    n_eq = len(re.findall(r"(?m)^[ \t]*\$\$[ \t]*$", t)) // 2
    ref_pat = re.compile(
        r"\b(?:Eqs?\.|[Ee]quations?)\s*\(?\s*\d{1,2}"
        r"(?:\s*\)?\s*(?:[\u2013,\-]|and)\s*\(?\s*\d{1,2})*\)?")
    cited = set()
    for m2 in ref_pat.finditer(t):
        cited.update(int(x) for x in re.findall(r"\d{1,2}", m2.group(0)))
    over = sorted(n for n in cited if not 1 <= n <= n_eq)
    ok = not over
    print(f"  {'OK  ' if ok else 'FAIL'} 公式引用无悬空（{n_eq} 个公式，引用号 {sorted(cited)}）")
    if not ok:
        die(f"终检失败：引用号 {over} 超出 1..{n_eq}")

    print("\n终检全部通过")


if __name__ == "__main__":
    main()
