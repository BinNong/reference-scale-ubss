#!/usr/bin/env python3
"""第四轮审稿修订：结构调整（Minor 6 与新增表格）。

做四件事：
  1. 把 §10「Robustness and Cost of the Front End」整段移为**附录 A**（放在结论之后、参考文献之前），
     子节由 10.x 改为 A.x；
  2. §11 Discussion → §10、§12 Conclusion → §11（正文不留编号缺口；稿中无 Section 11/12 引用，已核）；
  3. 把正文里对它的交叉引用由 `Section 10.x` 改为 `Appendix A.x`；
  4. 在附录中插入新表 Table 27–30（参考尺度分类、守卫错误分布与消融、密度前端、鲁棒噪声底）。

用法:
  python3 apply_r4_structure.py --dry
  python3 apply_r4_structure.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

MS = pathlib.Path(__file__).resolve().parent.parent / "paper" / "manuscript.md"

def _join_caption(text: str) -> str:
    """把 \*\*Table N.\*\* 表题合并成**单行**。

    构建脚本按单行解析表题（第二轮的教训）：表题跨行时会报"caption 后面找不到表格"。
    这里对新写的四张表统一处理，避免手写时又分多行。
    """
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("**Table "):
            buf = [l]
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("|", "**", "#")):
                buf.append(lines[j].strip())
                j += 1
            out.append(" ".join(buf))
            i = j
            continue
        out.append(l)
        i += 1
    return "\n".join(out)


APP_TITLE = "Appendix A. Robustness and Cost of the Front End"
APP_INTRO = """*This appendix collects the robustness, sensitivity and cost material referenced from the main text,
together with the four tables added in revision: A.9 the stability of the selected coefficient, A.10 where
the retention criterion's errors lie, A.11 a third mask-consuming front end, A.12 a robust-scale comparison
of the noise-floor estimator, and A.13 the classes of reference scale in use. Table numbers continue from the
main text.*"""

TABLE_27 = _join_caption("""### A.13 Classes of reference scale

**Table 30.** Classes of reference scale for the energy gate. The table classifies the *forms* the literature
uses and names the works this paper associates with each in Section 2; it is not a census of the constants
those works adopt, which we have not attempted, and the assessment in the last two columns is ours.

| Reference scale | What it is a quantile of | Implied noise-floor multiple moves with $p$? | Where it appears |
|---|---|---|---|
| fixed fraction of the mixture maximum | the mixture's upper extreme | weakly: the maximum tracks the signal scale, so the implied multiple stays inside a small range | [5,7]; the $c=0.05$ baseline here |
| fixed percentile of the mixture (median, 90th, ...) | the mixture's own distribution | strongly: the implied multiple *is* $r(p)$ of Proposition 6, which moves by $87\\times$ (Table 1) | [6,8,9]; the conventional default here |
| fixed multiple of the noise floor | the noise component alone | no, by construction | this work |
| absolute magnitude, no reference scale | --- | not applicable: it requires a re-chosen constant per recording | avoided in practice |
| no gate, collinearity alone | --- | not applicable: the contamination ratio $\\kappa$ of Proposition 4 grows as $(1-p)/p$ | the ungated rows of Tables 19–20 |
""")

TABLE_28 = _join_caption("""### A.10 Where the retention criterion's errors lie

**Table 27.** Where the retention criterion's 17 material errors lie, and what its three clauses each
contribute. Decisions are counted per seed over the 27 conditions of Table 4 (270 decisions), with the
material-gain criterion of Section 5.3. Families are those of the protocol of Section 7.

| Family | decisions | material errors | rate |
|---|---|---|---|
| sparse ladder, $p=0.02$–$0.40$ | $150$ | $4$ | $2.7\\%$ |
| dense ($p=1$) | $60$ | $2$ | $3.3\\%$ |
| structured sources (chirp, sinusoid, AM–FM, transient) | $40$ | $8$ | $20\\%$ |
| Laplace and block-sparse | $20$ | $3$ | $15\\%$ |
| **all** | $270$ | $17$ | $6.3\\%$ |

Of the $17$, $4$ sit at $\\mathrm{SNR}\\le5$ dB; the rest sit at the nominal $20$ dB, so the errors are not
concentrated at the corner of the sparsity–SNR plane. Clauses:

| Clause | decisions in which it is the *sole* failing one (synthetic) | (real speech) |
|---|---|---|
| retention $n_{\\rm keep}\\ge\\max(n_{\\min},\\eta\\,n_{\\rm base})$ | $68$ | $0$ |
| dispersion $\\mathrm{spread}\\le0.5$ | $0$ | $5$ |
| $n_{\\rm base}\\ge n_{\\min}$ | $0$ | $0$ |
| **errors with / without the dispersion clause** | $17\\,/\\,17$ | --- |

The dispersion clause is therefore dead weight on the synthetic grid and the only live clause on real
speech, where its five firings are the babble and noise-free configurations of Section 9 — the two whose
noise is not a stationary white floor. We keep it for that reason and not as a redundant guard.
""")

TABLE_29_AND_PROSE = _join_caption("""### A.11 A third mask-consuming front end: density pre-screening

Section 8.5 tests the transfer of the rule on a spherical $k$-means and on a fuzzy $c$-means front end. A
density-based detector is the third class a referee would ask about, and it behaves differently in a way
that is worth measuring rather than asserting. We use the construction the DBSCAN family uses in this
literature [8,9] — DBSCAN pre-screening to discard low-density directions, followed by spherical $k$-means —
because DBSCAN does not itself return a fixed number of cluster centres. Both gates share the same
`eps` and `min_samples`, so only the energy criterion differs.

**Table 28.** DBSCAN pre-screening ($\\texttt{min\\_samples}=10$) followed by spherical $k$-means, ten seeds, on the
synthetic ladder. The two gates share every front-end parameter. "Retained" is the fraction of admitted
points DBSCAN keeps; the last column counts the regimes (of six) in which the calibrated gate is better.

| `eps` | DBSCAN retained | classical gate | NF gate | change | regimes where NF is better |
|---|---|---|---|---|---|
| $0.02$ | $74\\%$ | $3.51°$ | $3.92°$ | $-11\\%$ | $4/6$ |
| $0.05$ | $91\\%$ | $6.54°$ | $2.11°$ | $+68\\%$ | $4/6$ |
| $0.10$ | $98\\%$ | $8.45°$ | $2.17°$ | $+74\\%$ | $5/6$ |

The pattern is the empirical form of the structural argument in Section 2. At $\\texttt{eps}=0.10$ and $0.05$
the pre-screen is nearly inactive (it retains $98\\%$ and $91\\%$ of the admitted points), so the energy
criterion still carries the detection and the gap between the two gates is the one we measure elsewhere: at
$\\texttt{eps}=0.10$ the conventional gate gives $15.36°$ against $0.36°$ at $p=0.02$ and $13.00°$ against
$0.32°$ at $p=0.05$. At $\\texttt{eps}=0.02$ the pre-screen itself begins to reject, retaining $74\\%$, and the
conventional gate's error collapses from $15.36°$ to $0.52°$ at $p=0.02$: the pipeline no longer depends on
the gate because the density stage has taken over the rejection, and the comparison becomes a wash. In the
dense regime the pre-screen actively hurts, keeping only $29\\%$ of the points at $\\texttt{eps}=0.02$ — those
in the densest directions, which are the multi-source ones — and the error rises to $18.08°$. This is why we
treat density detectors as a different mechanism rather than as head-to-head comparators, and the numbers
here are the demonstration rather than the assertion.
""")

TABLE_30_AND_PROSE = _join_caption("""### A.12 A robust scale estimator in place of the lower tail

The lower tail assumes that the noise power is the same at every TF point. Where it is not, the estimator
reads the *quiet* points and underestimates the floor. The natural repair to test is a robust scale
estimator that uses the centre rather than the tail of the distribution, since such an estimator averages
over the plane. We compare the lower-tail estimator of Section 5.1 with the two standard robust scales, the
median absolute deviation of the real part and its $25\\%$ quantile, scaled so that each returns $\\sigma$ on
Gaussian noise (verified: on pure Gaussian noise over $40$ draws all three return $1.00\\pm0.04$). Each is used
in the same gate, with the same $\\tau$ and the same base mask; only $\\hat\\nu$ changes.

**Table 29.** Three noise-floor estimators in the same gate. "stationarity" is the across-frequency
interquartile dispersion of the noise power over the frequency axis — a direct test of the premise the lower
tail needs, and it is computed on the noise itself. Errors are mean angle errors in degrees.

| Noise | stat. | $\\hat\\sigma^2/\\sigma^2$ lower tail | MAD | $q_{25}$ | error: lower tail | MAD | $q_{25}$ |
|---|---|---|---|---|---|---|---|
| Gaussian, $p=0.05$ | $0.14$ | $1.09$ | $1.56$ | $1.49$ | $0.40$ | $0.39$ | $0.40$ |
| Gaussian, $p=0.40$ | $0.14$ | $3.07$ | $51.9$ | $23.2$ | $1.39$ | $10.13$ | $3.93$ |
| Laplace, $p=0.40$ | $0.22$ | $1.60$ | $53.0$ | $23.0$ | $1.19$ | $9.86$ | $4.42$ |
| impulsive, $p=0.40$ | $0.15$ | $0.93$ | $52.9$ | $18.6$ | $1.02$ | $11.19$ | $5.11$ |
| colored, $p=0.40$ | $0.12$ | $3.05$ | $52.5$ | $23.1$ | $1.31$ | $9.65$ | $6.00$ |
| uniform, $p=0.40$ | $0.09$ | $5.24$ | $52.8$ | $23.0$ | $2.61$ | $10.76$ | $4.25$ |
| real speech, Gaussian | $0.10$ | $1.61$ | $2.55$ | $2.34$ | $0.56$ | $0.45$ | $0.47$ |
| real speech, babble | $5.80$ | $0.015$ | $0.62$ | $0.25$ | $1.22$ | $0.59$ | $0.69$ |
| real speech, noise-free | --- | --- | --- | --- | $0.61$ | $0.37$ | $0.44$ |

Two conclusions, and the second is the answer to the question this comparison was run for. First, the
robust scales are *not* safer: they fail where more than half the points carry signal, which is exactly the
dense and impulsive regimes, and the failure is catastrophic ($52$–$53\\times$ high at $p=0.40$ against
$0.9$–$5.2\\times$ for the lower tail). Robustness to heavy tails and robustness to contamination are
different properties and the second is the one that matters here. Second, on babble the ordering reverses
for the reason the stationarity column shows: babble's power varies across frequency by a factor of five
(the statistic is $5.80$ against $0.10$ for the Gaussian recordings), so the lower tail reads the quiet bins
and underestimates the floor by $66\\times$, while the centre-based estimators are within a factor of
$1.6$–$4$. Substituting them cuts the babble error from $1.22°$ to $0.59°$.

That is a repair in the one configuration where the premise fails, and it is not one we propose to adopt,
because it is bought by the collapse above. What we take from it is the diagnosis: under babble the failure
is not that the estimator is insufficiently robust, it is that the noise is not stationary across the plane,
which no scale estimator can repair. The same statistic is computable at run time from the estimator's own
level curve, so it is available as a second diagnostic; we report the measurement and leave the modelling of
a TF-varying floor to future work.
""")

TABLE_A9 = """### A.9 Stability of the selected coefficient

The real-speech comparison leaves a question that decides how the negative result should be read: if a
single coefficient works well for a max-referenced gate, perhaps coefficients in general are stable on this
corpus, and the drift of Section 4 is an artefact of the synthetic ladder. The oracle grid answers it. Over
the fifteen configurations the per-configuration optimum of the max-referenced coefficient runs over
$0.001$–$0.1$, a factor of $100$, and that of the median-referenced multiple over $5$–$1000$, a factor of
$200$ (`real_grid.json`). Neither is stable. What distinguishes the two is flatness rather than stability:
holding the max-referenced coefficient at $0.05$ costs $0.92°$ on average against $0.49°$ for the
per-configuration optimum, a penalty of $0.43°$, while holding the conventional median-referenced default at
its $0.02$ costs an order of magnitude against the $8.26°$ to $0.70°$ gap of Section 9.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    s = MS.read_text()
    lines = s.split("\n")

    if APP_TITLE in s:
        print("  [跳过] 附录已存在")
        return

    # ---- 1. 取出 §10 整段 ----
    i = next(k for k, l in enumerate(lines)
             if l.startswith("## 10. Robustness and Cost of the Front End"))
    j = next(k for k, l in enumerate(lines) if l.startswith("## 11. Discussion"))
    block = lines[i:j]
    print(f"  §10 共 {j - i} 行（L{i+1}–L{j}）")

    # ---- 2. 重编号子节 10.x -> A.x，并把顶层标题改为附录 ----
    block = [re.sub(r"^### 10\.(\d+) ", r"### A.\1 ", l) for l in block]
    block[0] = f"## {APP_TITLE}"
    # 标题后插附录引言；新小节按 27–30 的顺序依次附在 A.8 之后
    block = block[:1] + ["", APP_INTRO, ""] + block[1:]
    if not any(l.startswith("### A.9 ") for l in block):
        block = block + ["", TABLE_A9, "", TABLE_28, "", TABLE_29_AND_PROSE, "",
                         TABLE_30_AND_PROSE, "", TABLE_27]
    print(f"  附录块 {len(block)} 行，含 A.1–A.13")

    # ---- 3. 组装：删除原 §10，把附录放到结论之后、Figures 之前 ----
    rest = lines[:i] + lines[j:]
    # 重编号 §11 -> §10，§12 -> §11（稿中无 Section 11/12 引用，已核）
    rest = [re.sub(r"^## 11\. Discussion and Limitations$", "## 10. Discussion and Limitations", l)
            for l in rest]
    rest = [re.sub(r"^## 12\. Conclusion$", "## 11. Conclusion", l) for l in rest]
    kf = next(k for k, l in enumerate(rest) if l.strip() == "## Figures")
    out = rest[:kf] + block + ["", "---", ""] + rest[kf:]
    s2 = "\n".join(out)

    # ---- 4. 交叉引用 ----
    before = s2.count("Section 10")
    s2 = re.sub(r"Section 10\.(\d+)", r"Appendix A.\1", s2)
    s2 = re.sub(r"Section 10\b(?!\.)", "Appendix A", s2)
    print(f"  交叉引用 Section 10* -> Appendix A*：{before} 处")

    # ---- 4b. 表号与引用对齐（新表落在附录末尾，编号 27–30）----
    for old, new in (
        ("is worth stating plainly (Table 28)", "is worth stating plainly (Table 27)"),
        ("Appendix A.3 gives the ablation", "Appendix A.10 gives the ablation"),
        ("Table 27 sets the classes of reference scale", "Table 30 sets the classes of reference scale"),
    ):
        if old not in s2:
            sys.exit(f"引用对齐未命中: {old}")
        s2 = s2.replace(old, new, 1)
    print("  表号引用已对齐（守卫=Table 27，分类=Table 30）")

    # ---- 5. 校验 ----
    order = re.findall(r"^\*?(First|Second|Third|Fourth|Fifth|Sixth|Seventh)\*?", s2, re.M)
    secs = re.findall(r"^## (\d+)\. ", s2, re.M)
    tabs = sorted(int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", s2))
    print(f"  章节编号: {secs}")
    print(f"  表格编号: {tabs}")
    if secs != [str(x) for x in range(1, 12)]:
        sys.exit(f"章节编号异常: {secs}")
    if tabs != list(range(1, 31)):
        sys.exit(f"表格编号异常: {tabs}")
    if "## 10. Robustness" in s2:
        sys.exit("§10 未被移除")

    if not a.dry:
        MS.write_text(s2)
    print(f"  {'演练' if a.dry else '已写入'}：{len(out)} 行")


if __name__ == "__main__":
    # —— 已冻结（R8 表号重排）——
    # R8 把表号与附录小节号整体重排（旧 23/24/39 -> 新 19/20/21；附录 A.7..A.20 顺延为 A.5..A.17），
    # 本脚本锚定的是**重排前**的编号：重放会写入过时编号，个别条目还会把已存在的句子再插一遍。
    # 按"登记而非删除"的既有约定，条目原样保留，只把入口冻结。活跃的是
    # apply_r7b_fig_renumber.py / apply_r8_fixes.py / apply_r8b_restructure.py。
    print("[R8 已冻结] 表号与附录小节号已整体重排，本脚本锚定旧编号，跳过。")
