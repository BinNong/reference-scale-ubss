r"""第十轮（对外）B1 的稿件落地：Appendix A.19 + Table 44。

回应 CSSP 第二轮意见 MC2 —— 它要求在**固定最小列夹角**下直接扫描活跃 Gram 的谱结构
（"minimum angle ≠ complete conditioning characterization"）。

三处改动（**表放附录末尾，不触发表号重排**）：

  1. §7 可用性清单：点名新归档 `r10_gram_spectrum.json`；
  2. §4.2：在 "Empirical" 段末补一句指引 §A.19 的结论；
  3. 附录新增 **§A.19** + **Table 44**（插在 §A.18 之后、"## Figures" 之前）。

数字全部由 `src/r10_gram_spectrum.py` 产出、写进 `results/r10_gram_spectrum.json`，
并由 `src/verify_r10.py` 从**手稿解析后**与归档复算值比对（本脚本不写死任何数字判据）。

条目可带可选的**第 4 个元素**：判"已完成"的短签名（默认 = 整段 new）。后一轮在条目 new 中间插字时，
完整 new 会失效而效果仍在，用签名即可避免假未命中（C2 就插了两处）。

用法：
    python3 src/apply_r10b_gram_spectrum.py --dry
    python3 src/apply_r10b_gram_spectrum.py
"""

import argparse
import pathlib
import re
import sys

MS = pathlib.Path("paper/manuscript.md")

SEC_A19 = r"""### A.19 Sensitivity to the eigenstructure of the active Gram

Section 4.2 replaces the weighted sum of exponentials $\|\mathbf{A}_J\mathbf{s}_J\|^2$ by its mean-eigenvalue Gamma form and argues that the resulting error is not primarily a proximity effect. Table 20 varies the *minimum column separation* and finds the algorithm insensitive to it; this appendix varies the quantity the approximation actually acts on — the spectrum of $\mathbf{A}_J^{\mathsf H}\mathbf{A}_J$ — while the minimum separation is held fixed at the $12°$ of Section 7.

**Protocol.** Two hundred mixing matrices are drawn from the generator of Section 7 ($\theta_{\min}=12°$, unit-norm columns), and for every $J$-column subset ($J=2,3,4$; all $\binom{4}{J}$ of them) the exact signal energy $(E_J/J)\sum_j\lambda_je_j$, with $e_j\sim\mathrm{Exp}(1)$ and $\lambda_j$ the eigenvalues of the Gram, is compared with the mean-eigenvalue form $(E_J/J)\sum_je_j$ by Monte Carlo — $4\times10^{5}$ draws per subset and $4{,}400$ subsets in total, single process, fixed seed. Because the columns are unit-norm, $\mathrm{tr}(\mathbf{A}_J^{\mathsf H}\mathbf{A}_J)=J$ and the mean eigenvalue is exactly one, so the two laws share their mean and the deviation is a pure function of the spectrum $\{\lambda_j\}$.

**Table 44.** Median relative deviation (per cent) of the signal energy under the mean-eigenvalue substitution, at the fixed $12°$ minimum column separation of Section 7, binned by the condition number $\kappa=\lambda_{\max}/\lambda_{\min}$ of the active Gram; the last column is the rank-deficient case $\lambda_{\min}=0$, which occurs whenever $J>M$. The $12°$ floor caps $\kappa$ at $90.5$ when $J=2=M$, so the $\kappa\ge100$ column is empty there.

| Active Gram | $\kappa<1.5$ | $1.5\le\kappa<3$ | $3\le\kappa<10$ | $10\le\kappa<100$ | $\kappa\ge100$ | $\lambda_{\min}=0$ |
|---|---|---|---|---|---|---|
| $J=2$, $M\in\{2,3\}$ | $0.2$ | $2.9$ | $10.7$ | $18.5$ | --- | --- |
| $J=3$, $M=3$ | $0.3$ | $1.9$ | $5.5$ | $8.8$ | $10.7$ | --- |
| $J=3$, $M=2$ | --- | --- | --- | --- | --- | $10.2$ |
| $J=4$ | --- | --- | --- | --- | --- | $11.2$ |

Two things follow. First, for $J=2$ the deviation is a *deterministic* function of the condition number — the Spearman rank correlation between $\kappa$ and the deviation is $1.000$ — because a $2\times2$ Gram with unit diagonal has eigenvalues $1\pm\cos\theta$, so the condition number and the pair angle carry the same information; the deviation runs from $0.2\%$ where the columns are nearly orthogonal to $18.5\%$ at the $12°$ corner. Second, that one-scalar description does **not** extend to $J\ge3$: for $J=3$, $M=3$ the same correlation falls to $0.649$, so the deviation is not a function of the conditioning alone and the whole spectrum is needed. Rank deficiency is a third and separate term: whenever $J>M$ the Gram carries a zero eigenvalue that the mean-eigenvalue form replaces by one, and the deviation is then $10$–$12\%$ at every separation.

What the method uses is not the signal energy but the observable $r(p)$. Re-evaluating the mixture law of Proposition 6 with the two signal laws gives a deviation of at most $0.003\%$ for $p\le0.10$ — the regime in which the median lies inside the noise component and the semi-analytic form is used to advantage — rising to $4.1\%$ at $p=0.20$ and $13.0\%$ at $p=0.40$ for $M=2$ ($2.8\%$ and $8.9\%$ at $M=3$). These reproduce the figures quoted in Section 4.2 from an independent Monte Carlo, and they are why the approximation is described there as affecting the quantity the method uses by less than $0.1\%$ throughout the range in which that quantity is used, rather than globally. The record is `r10_gram_spectrum.json`."""

E = [
    (
        "§7 可用性清单：点名 r10_gram_spectrum.json",
        r"as released in OpenSLR SLR28. Code, the fixed configurations,",
        r"as released in OpenSLR SLR28. The eigenstructure study of Appendix A.19 adds "
        r"`r10_gram_spectrum.json` ($4{,}400$ random column subsets at the fixed $12°$ minimum "
        r"separation of Section 7, two hundred mixing matrices). Code, the fixed configurations,",
        # ⚠️ 签名：C2（apply_r10d）在本条 new 的中间插了"第二语料"那句，完整 new 已不再逐字存在。
        #    判"已完成"改用短签名——只要该效果在，就跳过（不要因为别人在中间插字而报未命中）。
        "The eigenstructure study of Appendix A.19 adds",
    ),
    (
        "§4.2：在 Empirical 段末补指引 §A.19 的一句",
        r"and reaches $13\%$ only at $p=0.40$, where the small-argument expansion has already ceased to apply. We therefore retain",
        r"and reaches $13\%$ only at $p=0.40$, where the small-argument expansion has already ceased to apply. "
        r"Appendix A.19 sweeps the eigenstructure of the active Gram directly — at the *fixed* minimum separation "
        r"of Section 7, so that proximity is not the variable — and finds that the substitution error on the signal "
        r"energy is a deterministic function of the Gram's conditioning for $J=2$ but not of it alone for $J\ge3$, "
        r"while the effect on $r(p)$ stays within the figures just quoted. We therefore retain",
    ),
    (
        "附录：新增 §A.19 + Table 44（插在 §A.18 之后）",
        r"banded floor of Appendix A.14 as the repair rather than the guard." + "\n\n" + "## Figures",
        r"banded floor of Appendix A.14 as the repair rather than the guard." + "\n\n"
        + SEC_A19 + "\n\n" + "## Figures",
        # ⚠️ 同上：C2 把 §A.20 插在本条 new 的末尾与 "## Figures" 之间，完整 new 已不逐字存在。
        "### A.19 Sensitivity to the eigenstructure",
    ),
]


def die(msg: str) -> None:
    print(f"✗ {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args()
    p = pathlib.Path(a.manuscript) if a.manuscript else MS
    s = p.read_text()
    changed = skipped = missing = 0
    for item in E:
        tag, old, new = item[0], item[1], item[2]
        # 第 4 元素可选：判"已完成"的**短签名**（默认就是整段 new）。
        # 后一轮若在本条 new 的中间插字，完整 new 会失效，但效果仍在——用签名避免假未命中。
        sig = item[3] if len(item) > 3 else new
        n = s.count(old)
        if n == 0:
            if sig in s:
                print(f"  [已改，跳过] {tag}")
                skipped += 1
                continue
            print(f"  ✗ 未命中（原文不存在）：{tag}")
            missing += 1
            continue
        if n > 1:
            print(f"  ✗ 原文不唯一（{n} 次）：{tag}")
            missing += 1
            continue
        if old in new:
            die(f"条目 {tag!r} 的 new 包含完整的 old —— 会破坏幂等")
        s = s.replace(old, new, 1)
        print(f"  ✓ {tag}")
        changed += 1

    if missing:
        die(f"有 {missing} 条未命中，整体不写入")
    if a.dry:
        print(f"\n演练：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")
        return
    p.write_text(s)
    print(f"\n已写入 {p}：已改 {changed} | 跳过 {skipped} | 未命中 {missing}")

    # ------------------------------------------------------------ 终检
    t = p.read_text()
    print("\n=== 终检 ===")
    for probe in ("### A.19 Sensitivity to the eigenstructure",
                  r"**Table 44.** Median relative deviation",
                  r"r10_gram_spectrum.json",
                  r"Appendix A.19 sweeps the eigenstructure"):
        ok = probe in t
        print(f"  {'OK  ' if ok else 'FAIL'} {probe[:64]}")
        if not ok:
            die("终检失败")
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    print(f"  表题注 {len(caps)} 张（1..{max(caps)} 递增？{caps == list(range(1, max(caps) + 1))}）")
    print(f"  附录小节 A.{secs[0]}..A.{secs[-1]} 共 {len(secs)} 节（各一次？{secs == list(range(1, len(secs) + 1))}）")
    if caps != list(range(1, 45)):
        die("表题注不是 1..44")
    if secs != list(range(1, 20)):
        die("附录小节不是 A.1..A.19")
    print("\n✓ 全部终检通过")


if __name__ == "__main__":
    main()
