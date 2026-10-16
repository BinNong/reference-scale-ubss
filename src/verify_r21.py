#!/usr/bin/env python3
r"""R21 复核：公式 (10) 的裸符号 $e$ 已有定义（逐点能量的省略下标写法）。

用户读 PDF 时发现公式 (10)

    $$e = \\frac{\\sigma^2}{2}\\chi^2_{2M} + \\Gamma(J, \\frac{M\\sigma^2}{Np}10^{SNR/10}),$$

中的 $e$ 是新符号。核查结论：它不是新符号，而是 §4.1 已定义的
$e_{f,t}=\\|\mathbf{x}(f,t)\|^2$ 的省略下标写法——但稿中**从未声明过这件事**，
裸 $e$ 在摘要、Table 1/14/30/43 题注、Fig. 1 题注、Proposition 4 里已用了
九处。R21 在 §4.1 定义句里补上该声明，并把 §7 唯一一处欧拉数
$e^{-0.02}$ 改成 $\\exp(-0.02)$（$e$ 成为已定义符号后，那处是真正的符号冲突）。

判据全部从手稿解析：
  - 公式 (10) 的本体逐字未动（R21 只改散文，不改公式）；
  - 定义子句存在且唯一，且声明的正是 $e_{f,t}$ 的省略下标；
  - 稿中每一处裸 $e$ 的用法都仍在（median}(e) 的出现数不减少）；
  - 欧拉数裸写 $e^{...}$ 零残留；
  - 公式总数仍为 20（本轮不新增 $$ 对）。

用法：
    python3 verify_r21.py [--manuscript <path>]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

FAIL: list[str] = []

P_DEF = r"we drop the subscript and write $e$, and let"
# 手稿里 `$$` 独占一行，故公式本体不能与 `$$` 写在同一子串里
EQ10_BODY = (
    r"e \;=\; \frac{\sigma^{2}}{2}\,\chi^{2}_{2M}\;+\;"
    r"\Gamma\!\Big(J,\ \frac{M\sigma^{2}}{Np}\,10^{\mathrm{SNR}/10}\Big),"
)
EQ9_BODY = r"e_{f,t} > \underbrace{t_e\, r(p)}_{\text{effective multiple}}\ \nu,\qquad"


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

    # ---- A. 公式清单：定位公式 (10)，确认本体逐字未动 ----
    print("--- A. 公式 (10) 的本体与裸 $e$ 的定义 ---")
    eqs = re.findall(r"\$\$(.*?)\$\$", t, re.S)
    want(f"手稿共有 20 个公式（$$ 对）（实为 {len(eqs)}）", len(eqs) == 20)
    if len(eqs) < 10:
        print("公式不足 10 个，后续断言无法进行")
        return 1
    eq10 = eqs[9]
    want("公式 (10) 是 Proposition 1 的逐点能量混合律", EQ10_BODY in eq10)
    want(r"公式 (10) 左边是裸 $e$（LaTeX 作 $e\;=\;$，反斜杠在分号前）",
         re.match(r"^\s*e\s*\\;?=", eq10) is not None)
    want("公式 (10) 本体逐字未动（R21 只改散文）",
         f"$$\n{EQ10_BODY}\n$$" in t)
    eq9 = eqs[8]
    want("公式 (9) 本体未动", EQ9_BODY in t and EQ9_BODY in eq9)

    # ---- B. 定义子句 ----
    print("\n--- B. §4.1 的裸 $e$ 定义子句 ---")
    want("定义子句存在且唯一", t.count(P_DEF) == 1, f"实为 {t.count(P_DEF)}")
    want("定义句同时保留带下标的 $e_{f,t}$ 定义",
         r"Let the **per-point energy** be $e_{f,t}=\|\mathbf{x}(f,t)\|^2$; where the" in t)
    want("定义子句挂在 §4.1（公式 (10) 之前）",
         t.index(P_DEF) < t.index(EQ10_BODY))

    # ---- C. 稿中每一处裸 $e$ 的用法都仍在 ----
    print("\n--- C. 裸 $e$ 的既有用法未被动 ---")
    bare_uses = {
        "摘要": r"$r(p)=\mathrm{median}(e)/\nu$",
        "§4.1 median(e)": r"$m_e=\mathrm{median}(e)$",
        "§4.1 e_max": r"the maximum $e_{\max}$",
        "Table 1 题注": r"$r(p)=\mathrm{median}(e)/\nu$",
        "Proposition 4": r"$e>\tau\nu$ is equivalent to",
        "Table 14 表头": r"$r=\mathrm{median}(e)/\nu$",
        "Table 43 题注": r"$e>t_e\,\mathrm{median}(e)$",
        "Fig. 1 题注": r"$r(p)=\mathrm{median}(e)/\nu$",
    }
    for tag, probe in bare_uses.items():
        want(f"{tag} 的裸 $e$ 用法仍在", probe in t)
    n_med = t.count(r"\mathrm{median}(e)")
    want(r"median}(e) 共 7 处（摘要 1 + §4.1 正文 1 + 题注/表头 5）", n_med == 7, f"实为 {n_med}")

    # ---- D. 欧拉数消歧 ----
    print("\n--- D. §7 欧拉数改写为 \\exp ---")
    want("裸欧拉数 $e^{-0.02}$ 已无残留", r"$e^{-0.02}" not in t)
    want("改写为 $\\exp(-0.02)$", r"$\exp(-0.02)=98\%$" in t)
    want("全稿再无其它裸 $e^{...}$（欧拉数）",
         re.search(r"\$e\^", t) is None)

    # ---- E. R22：公式 (10) 处的就地指引（用户翻到公式 (10) 没看到解释）----
    print("\n--- E. 公式 (10) 处的就地指引（R22）---")
    want("Proposition 1 的 where 从句点名 $e$ 的出处",
         r"$e$ is the per-point energy of Section 4.1" in t)
    want("指引子句只出现 1 次", t.count(r"$e$ is the per-point energy of Section 4.1") == 1,
         f"实为 {t.count(r'$e$ is the per-point energy of Section 4.1')}")
    want("指引与 R21 的定义句互补（两处都在）",
         P_DEF in t and r"$e$ is the per-point energy of Section 4.1" in t)
    want("指引紧跟公式 (10) 之后（同一命题内）",
         t.index(EQ10_BODY) < t.index(r"$e$ is the per-point energy of Section 4.1")
         < t.index(r"whose partial moments $m_k$ follow from the chi-square CDF"))

    print()
    if FAIL:
        print(f"仍有 {len(FAIL)} 项不一致：{FAIL}")
        return 1
    print("✓ verify_r21 全部通过（含 R22 的就地指引）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
