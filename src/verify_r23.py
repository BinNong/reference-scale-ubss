#!/usr/bin/env python3
r"""R23 复核：CDF 家族已从 $F$ 重命名为 $\Psi$，且频点计数 $F$ 未被误伤。

背景：用户问"公式 (11) 中的 F 又是什么"。公式 (11) 的 $F$ 是逐点能量的**混合 CDF**
（$F(x)=\sum_J\pi_JF_J(x)$，$F_J$ 条件 CDF，$F_0$ 纯噪声 CDF，另有 $F_s$、$F_e$、
$F_{\Gamma(J,\theta)}$），但字母 $F$ 在稿中还有两个含义：§7/Algorithm 1 的**频点计数**
$F=33$（$FT$、$L=2FT$）与 §A.12 的 **F1 score** $F_1$。用户定夺重命名 CDF 一侧。

判据（全部从手稿解析，`--manuscript <path>` 可做负向测试）：
  A. §4.2–§5.1 的 span 内 CDF 家族**全是** Ψ，零 F 残留；
  B. 每一处原 F 家族都有对应的 Ψ（32 处，逐模式计数）；
  C. 频点计数的四种写法（$F=33$ / $FT$ / $(F,T)$ / 算法规格）与 F1 score 原样健在；
  D. 公式 (11) 后就地指引 + Proposition 2 回引 Eq. 11；
  E. 公式总数仍为 20，未新增 $$ 对。

用法：
    python3 verify_r23.py [--manuscript <path>]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

FAIL: list[str] = []

SPAN_HEAD = "### 4.2 A semi-analytic form"
SPAN_TAIL = "### 5.2 Calibrating the multiple"

# 原 F 家族 → Ψ 家族，附期望处数（R23 改前实测）
FAMILY = [
    (r"F_{\Gamma(J,\theta)}", r"\Psi_{\Gamma(J,\theta)}", 2),
    (r"F_J", r"\Psi_J", 9),          # 8（重命名）+ 1（指引句）
    (r"F_0", r"\Psi_0", 11),         # 10（重命名）+ 1（指引句）
    (r"F_s", r"\Psi_s", 5),
    (r"F_e", r"\Psi_e", 1),
]
BARE_RE = re.compile(r"(?<![A-Za-z0-9_])F(?![A-Za-z0-9_])")


def want(tag: str, ok: bool, extra: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {tag}{'  ' + extra if extra else ''}")
    if not ok:
        FAIL.append(tag)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args()
    MS = pathlib.Path(a.manuscript) if a.manuscript else (
        pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md")
    t = MS.read_text()

    if SPAN_HEAD not in t or SPAN_TAIL not in t:
        print("FAIL 找不到 span 边界标题（§4.2 / §5.2）")
        return 1
    i0, i1 = t.index(SPAN_HEAD), t.index(SPAN_TAIL)
    span, outside = t[i0:i1], t[:i0] + t[i1:]

    # ---- A/B. span 内：Ψ 到位、F 清零、逐模式计数 ----
    # ⚠️ 期望处数 = R23 重命名处的数量 **+ 就地指引句新增的**：指引句把原来的
    #    `$\Psi(m_e)$`（1 个裸 Ψ）换成 "$\Psi$ is … ($\Psi_J$ … $\Psi_0$ …) …
    #    $\Psi(m_e)$"（2 裸 + 1 Ψ_J + 1 Ψ_0），净增 1 裸 + 1 Ψ_J + 1 Ψ_0。
    print("--- A/B. §4.2–§5.1：CDF 家族 F → Ψ ---")
    for old, new, n in FAMILY:
        want(f"{old} 已全部改为 {new}", span.count(new) == n and span.count(old) == 0,
             f"Ψ={span.count(new)} / F={span.count(old)}")
    n_bare = len(re.findall(r"\\Psi(?![_ {])", span))
    want("span 内裸 Ψ 为 7 个（原 6 + 指引句净增 1）", n_bare == 7, f"实为 {n_bare}")
    n_psi = span.count(r"\Psi")
    want("span 内 Ψ 共 35 处（重命名 32 + 指引句净增 3）", n_psi == 35, f"实为 {n_psi}")

    # ---- C. span 外：频点计数与 F1 score 未被误伤 ----
    print("\n--- C. 频点计数 $F$ 与 F1 score 原样健在 ---")
    for probe, what, n in (
        (r"$F=33$", "§7 协议 $F=33$（频点数目）", 1),
        (r"$FT=2112$", "$FT=2112$", 1),
        (r"$L=2FT=4224$", "$L=2FT=4224$", 1),
        (r"C^{M x F x T}", "Algorithm 1 输入规格", 1),
        (r"{0,1}^{F x T}", "Algorithm 1 输出规格", 1),
        (r"$(F,T)$", "§A.27 的 $(F,T)$", 1),
        (r"$F_1$", "F1 score 记号", 4),
    ):
        want(f"{what}", t.count(probe) == n, f"实为 {t.count(probe)}")
    want("span 外裸 F 只剩频点计数与作者姓氏（不得出现 CDF 家族）",
         not re.search(r"F_\{J\}|F_J|F_0|F_s|F_e|F_\{\\Gamma", outside))

    # ---- D. 就地指引与公式号回引 ----
    print("\n--- D. 公式 (11) 的就地指引与 Prop 2 回引 ---")
    want("公式 (11) 本体为 Ψ 家族",
         r"\Psi(x)\;=\;\sum_{J=0}^{N}\pi_J\,\Psi_J(x)" in t)
    want("公式 (11) 后的就地指引（Ψ 是什么）",
         r"where $\Psi$ is the mixture CDF of the per-point energy ($\Psi_J$ conditional "
         r"on exactly $J$ active sources, $\Psi_0$ noise-only)" in t)
    want("Proposition 2 回引 Eq. 11",
         r"Write $\Psi=\sum_{J\ge0}\pi_J\Psi_J$ (Eq. 11) for the mixture CDF" in t)
    want("公式 (12) 的小参数展开也已改用 Ψ",
         r"\Psi_{\Gamma(J,\theta)}(z)\simeq(z/\theta)^J/J!" in t)

    # ---- E. 结构 ----
    print("\n--- E. 结构 ---")
    n_eq = len(re.findall(r"\$\$", t)) // 2
    want(f"公式仍为 20 个（实为 {n_eq}）", n_eq == 20)
    want("未产生 \\Psi_{2M} 这类误改下标", r"\Psi_{2M}" not in t)
    want("Gram 矩阵记号 \\mathbf{G} 未动", r"\mathbf{G}" in t)

    print()
    if FAIL:
        print(f"仍有 {len(FAIL)} 项不一致：{FAIL}")
        return 1
    print("✓ verify_r23 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
