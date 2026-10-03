#!/usr/bin/env python3
"""第九轮（CSSP Minor Revision）的**数字更新**：真实语音 oracle 改为 held-out 调参种子后的全部连锁数字。

背景：意见指出摘要说 "best-observed reference that is allowed to inspect the test data"，
而 Table 15 题注说 "tuned per configuration on separate seeds"——两者矛盾。核查代码
（`src/experiments_real.py`）发现实际情况是第三种：调参用 `range(tune_seeds)` = 种子 {0,1}，
评测用 `range(seeds)` = 种子 {0..7}，**调参实例是评测实例的子集**（2/8），存在轻微泄漏。

已改为 `range(seeds, seeds + tune_seeds)`（种子 {8,9}），与评测集完全不相交，并重跑了
`results/real_results.json` 与 `results/real_grid.json`（960 条，用时 1922 s）。
试算与重跑都表明影响很小：best-observed 0.703° → 0.685°，比值 1.59x → 1.63x。

本脚本把**由此变化的每一个数字**落进手稿。所有数字均从归档现算，不写死。

用法：
    python3 src/apply_r9b_realnum.py --dry
    python3 src/apply_r9b_realnum.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper" / "manuscript.md"
RES = ROOT / "results"

# 15 个配置，顺序与稿中 Table 14/15 一致
CONF = [("R1_res", "w256"), ("R1_res", "w512"), ("R1_res", "w1024"), ("R1_res", "w2048"),
        ("R2_n", "N3"), ("R2_n", "N5"), ("R2_n", "N6"),
        ("R3_snr", "snr00"), ("R3_snr", "snr10"), ("R3_snr", "snr30"), ("R3_snr", "snr40"),
        ("R4_dense", "d1"), ("R4_dense", "d2"), ("R5_noise", "babble"), ("R5_noise", "clean")]
# 表体里的配置标签（与稿中一致）
LABEL = ["win 256", "win 512", "win 1024", "win 2048", "$N=3$", "$N=5$", "$N=6$",
         "SNR $0$ dB", "SNR $10$ dB", "SNR $30$ dB", "SNR $40$ dB",
         "1 dense source", "2 dense sources", "babble noise", "noise-free"]


def collect() -> dict:
    R = json.load(open(RES / "real_results.json"))["records"]
    G = json.load(open(RES / "real_grid.json"))
    cases = [f"{e}:{t}" for e, t in CONF]

    def col(meth, case):
        v = [r["A_angle_deg"] for r in R if r["case"] == case and r["method"] == meth]
        return float(sum(v) / len(v))

    opt_names = ("SCA-median-opt", "SCA-max-opt", "top-K-opt")
    cbest, best1, fixed05 = [], [], []
    for c in cases:
        cbest.append(min(G[f"{c}|max"], key=lambda k: G[f"{c}|max"][k]))
        best1.append(min(G[f"{c}|max"].values()))
        fixed05.append(G[f"{c}|max"]["0.05"])

    best3 = [min(col(m, c) for m in opt_names) for c in cases]
    best7 = [min(b, col("SCA-median", c), col("SCA-max", c), col("top-K", c))
             for b, c in zip(best3, cases)]
    nf_c = [col("NF-SSP", c) for c in cases]
    conv_c = [col("SCA-median", c) for c in cases]

    tes = sorted({G[f"{c}|star"]["te"] for c in cases})
    cs = sorted({G[f"{c}|star"]["c"] for c in cases})
    d = dict(
        # Table 14 的两列
        star_te=[G[f"{c}|star"]["te"] for c in cases],
        star_c=[G[f"{c}|star"]["c"] for c in cases],
        # Table 15 的三列 + 均值
        med_opt=[col("SCA-median-opt", c) for c in cases],
        max_opt=[col("SCA-max-opt", c) for c in cases],
        topk_opt=[col("top-K-opt", c) for c in cases],
        # 汇总量
        best3=best3,
        best3_mean=sum(best3) / 15,
        best7_mean=sum(best7) / 15,
        nf_mean=sum(nf_c) / 15,
        conv_mean=sum(conv_c) / 15,
        best3_lo=min(best3), best3_hi=max(best3),
        # 派生比值
        ratio_means=(sum(conv_c) / 15) / (sum(best3) / 15),
        pen_conv=sum(c / b for c, b in zip(conv_c, best3)) / 15,
        pen_conv_max=max(c / b for c, b in zip(conv_c, best3)),
        pen_nf=sum(n / b for n, b in zip(nf_c, best3)) / 15,
        # 系数范围
        te_lo=tes[0], te_hi=tes[-1], te_factor=tes[-1] / tes[0],
        c_lo=cs[0], c_hi=cs[-1], c_factor=cs[-1] / cs[0],
        # §A.7 的平坦度
        maxopt_mean=sum(col("SCA-max-opt", c) for c in cases) / 15,
        fixed05_mean=sum(fixed05) / 15,
        best1_mean=sum(best1) / 15,
    )
    d["pen_flat"] = d["fixed05_mean"] - d["best1_mean"]
    return d


def replace_table_columns(text: str, caption: str, cols: list[int],
                          values: list[list[float]], fmt: str) -> tuple[str, int]:
    """把 `**Table N.**` 之后那张表里指定列按行替换；values[i][j] 是第 i 行第 j 个待改列的**数值**。

    fmt 是单元格式样（如 "{:.2f}" / "{:g}"）；均值行由各列均值生成并加粗。
    """
    i = text.index(caption)
    j = text.index("\n\n", text.index("\n|", i))          # 表块结束
    blk = text[i:j]
    means = [sum(v) / len(v) for v in zip(*values)]
    out, row = [], 0
    for ln in blk.split("\n"):
        if ln.startswith("|") and not re.match(r"^\|[\s\-|:]+\|$", ln):
            cells = ln.split("|")
            is_mean = cells[1].strip().startswith("**mean")
            if not is_mean and cells[1].strip() == "Configuration":
                out.append(ln)
                continue
            if is_mean:
                vals = [f"**{fmt.format(m)}**" for m in means]
            elif row < len(values):
                vals = [fmt.format(x) for x in values[row]]
            else:
                out.append(ln)
                continue
            for k, ci in enumerate(cols):
                cells[ci] = f" {vals[k]} "
            ln = "|".join(cells)
            if not is_mean:
                row += 1
        out.append(ln)
    return text[:i] + "\n".join(out) + text[j:], row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    d = collect()
    s = MS.read_text()
    orig = s

    # ---------------- A. Table 14：optimal te / optimal c 两列 ----------------
    s, n14 = replace_table_columns(
        s, "**Table 14.**", [6, 7],
        [[float(te), float(c)] for te, c in zip(d["star_te"], d["star_c"])], "{:g}")
    print(f"  {'✓' if n14 == 15 else '✗'} Table 14 的两列（{n14}/15 行）")

    # ---------------- B. Table 15：三个 tuned 列 + 均值行 ----------------
    s, n15 = replace_table_columns(
        s, "**Table 15.**", [6, 7, 8],
        [[a_, b_, c_] for a_, b_, c_ in zip(d["med_opt"], d["max_opt"], d["topk_opt"])], "{:.2f}")
    print(f"  {'✓' if n15 == 15 else '✗'} Table 15 的三个 tuned 列（{n15}/15 行）")

    # ---------------- C. 其余数字（精确片段替换） ----------------
    E = [
        ("Table 15 题注：best-observed 均值与七列值",
         r"whose mean is $0.703°$; taken over all seven competing columns it would be $0.681°$.",
         rf"whose mean is ${d['best3_mean']:.3f}°$; taken over all seven competing columns it would be ${d['best7_mean']:.3f}°$."),
        ("Table 15 题注：penalty 与比值",
         r"the mean penalty of the conventional baseline against this bound is $12.8\times$ (ratio of means: $11.8\times$), and that of NF-SSP is $1.59\times$.",
         rf"the mean penalty of the conventional baseline against this bound is ${d['pen_conv']:.1f}\times$ (ratio of means: ${d['ratio_means']:.1f}\times$), and that of NF-SSP is ${d['pen_nf']:.2f}\times$."),
        ("Table 15 题注：separate seeds -> held out（协议名副其实）",
         r'"opt" denotes a coefficient tuned per configuration on separate seeds',
         r'"opt" denotes a coefficient tuned per configuration on a validation seed set held out from the evaluation seeds'),
        ("贡献 5：比值 + 协议措辞（该句在贡献列表，不在摘要）",
         r"within $1.59\times$ of a best-observed reference that is allowed to inspect the test data, while accounting for $0.17\%$ of the runtime.",
         rf"within ${d['pen_nf']:.2f}\times$ of a best-observed reference whose coefficients are tuned on seeds held out from the evaluation seeds, while accounting for $0.17\%$ of the runtime."),
        ("贡献 5：conventional 的 penalty 与上限",
         r"a conventionally set fixed threshold costs a mean factor of $12.8$ (up to $31$) against the same bound",
         rf"a conventionally set fixed threshold costs a mean factor of ${d['pen_conv']:.1f}$ (up to ${d['pen_conv_max']:.0f}$) against the same bound"),
        ("§9.3：best-observed 均值与 penalty",
         r"the two-stage pipeline attains a mean angle error of $8.26°$, against $0.70°$ for the best-observed reference. Counted per configuration the penalty averages $12.8\times$ and reaches $31\times$ at $N=3$;",
         rf"the two-stage pipeline attains a mean angle error of $8.26°$, against ${d['best3_mean']:.2f}°$ for the best-observed reference. Counted per configuration the penalty averages ${d['pen_conv']:.1f}\times$ and reaches ${d['pen_conv_max']:.0f}\times$ at $N=3$;"),
        # 下面两条是**目视**（渲染 §9.3 那一页）才发现的漏网：上面那条的片段到 "at $N=3$;" 为止，
        # 没有覆盖同句后半的 "the ratio of the two means" 与 §12 的 "12.8-fold penalty"。
        # 教训：以"句子片段"做替换时，**要先确认片段覆盖了句内所有受影响的数字**。
        ("§9.3：ratio of the two means",
         r"the ratio of the two means would be $11.8\times$",
         rf"the ratio of the two means would be ${d['ratio_means']:.1f}\times$"),
        ("§12：removes a silent 12.8-fold penalty",
         r"removes a silent $12.8$-fold penalty",
         rf"removes a silent ${d['pen_conv']:.1f}$-fold penalty"),
        ("§9.4：比值与 best-observed 范围",
         r"within $1.59\times$ of the best-observed reference ($1.12$–$2.95$).",
         rf"within ${d['pen_nf']:.2f}\times$ of the best-observed reference ($1.12$–${d['best3_hi']:.2f}$)."),
        ("§9.4：per-config oracle bound",
         r"against $0.70°$ for the per-configuration oracle bound of Table 15",
         rf"against ${d['best3_mean']:.2f}°$ for the per-configuration oracle bound of Table 15"),
        ("§9.5：hard-threshold bound",
         r"within noise of the per-configuration hard-threshold bound of $0.75°$ (Table 15)",
         rf"within noise of the per-configuration hard-threshold bound of ${sum(d['med_opt']) / 15:.2f}°$ (Table 15)"),
        ("§9.5：median multiple 的范围",
         r"on real speech every configuration prefers the large end of the range ($5$ to $1000$, Table 14)",
         rf"on real speech every configuration prefers the large end of the range (${d['te_lo']:g}$ to ${d['te_hi']:g}$, Table 14)"),
        ("§9.2：optimal multiple 的范围",
         r"the optimal multiple runs from $5$ to $1000$ (Table 14)",
         rf"the optimal multiple runs from ${d['te_lo']:g}$ to ${d['te_hi']:g}$ (Table 14)"),
    ]
    for tag, old, new in E:
        if old in new and old in s:
            print(f"  ✗ 条目设计有误（new 含 old）：{tag}")
            sys.exit(1)
        n = s.count(old)
        if n == 0:
            if new in s:
                print(f"  [已改，跳过] {tag}")
                continue
            pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
            if len(pat.findall(s)) == 1:
                s = pat.sub(lambda _: new, s, count=1)
                print(f"  ✓ {tag}（容忍换行）")
                continue
            print(f"  ✗ 未命中：{tag}")
            sys.exit(1)
        if n > 1:
            print(f"  ✗ 原文不唯一（{n} 次）：{tag}")
            sys.exit(1)
        s = s.replace(old, new, 1)
        print(f"  ✓ {tag}")

    # ---------------- D. §11 与 §A.7 的同一组数字（两处措辞略有不同） ----------------
    for tag, old, new in [
        ("§11：系数范围与倍数",
         r"the per-configuration optimum of the max-referenced rule runs over $0.001$–$0.1$, a factor of $100$, and that of the median-referenced rule over $5$–$1000$, a factor of $200$",
         rf"the per-configuration optimum of the max-referenced rule runs over ${d['c_lo']:g}$–${d['c_hi']:g}$, a factor of ${d['c_factor']:g}$, and that of the median-referenced rule over ${d['te_lo']:g}$–${d['te_hi']:g}$, a factor of ${d['te_factor']:g}$"),
        ("§11：平坦度与 penalty",
         r"holding $c=0.05$ costs $0.92°$ on average against $0.49°$ for the per-configuration optimum, a penalty of $0.43°$",
         rf"holding $c=0.05$ costs ${d['fixed05_mean']:.2f}°$ on average against ${d['best1_mean']:.2f}°$ for the per-configuration optimum, a penalty of ${d['pen_flat']:.2f}°$"),
        ("§11：1.59x -> 1.63x",
         r"($1.59\times$ against the best-observed reference)",
         rf"(${d['pen_nf']:.2f}\times$ against the best-observed reference)"),
        ("§A.7：系数范围与倍数",
         r"the fifteen configurations the per-configuration optimum of the max-referenced coefficient runs over $0.001$–$0.1$, a factor of $100$, and that of the median-referenced multiple over $5$–$1000$, a factor of $200$ (`real_grid.json`)",
         rf"the fifteen configurations the per-configuration optimum of the max-referenced coefficient runs over ${d['c_lo']:g}$–${d['c_hi']:g}$, a factor of ${d['c_factor']:g}$, and that of the median-referenced multiple over ${d['te_lo']:g}$–${d['te_hi']:g}$, a factor of ${d['te_factor']:g}$ (`real_grid.json`)"),
        ("§A.7：平坦度与 penalty",
         r"holding the max-referenced coefficient at $0.05$ costs $0.92°$ on average against $0.49°$ for the per-configuration optimum, a penalty of $0.43°$",
         rf"holding the max-referenced coefficient at $0.05$ costs ${d['fixed05_mean']:.2f}°$ on average against ${d['best1_mean']:.2f}°$ for the per-configuration optimum, a penalty of ${d['pen_flat']:.2f}°$"),
        ("§A.7：8.26° to 0.70° 的 gap",
         r"against the $8.26°$ to $0.70°$ gap of Section 9",
         rf"against the $8.26°$ to ${d['best3_mean']:.2f}°$ gap of Section 9"),
    ]:
        if old in new and old in s:
            print(f"  ✗ 条目设计有误（new 含 old）：{tag}")
            sys.exit(1)
        n = s.count(old)
        if n == 0:
            if new in s:
                print(f"  [已改，跳过] {tag}")
                continue
            pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
            hits = list(pat.finditer(s))
            if len(hits) == 1:
                s = s[:hits[0].start()] + new + s[hits[0].end():]
                print(f"  ✓ {tag}（容忍换行）")
                continue
            print(f"  ✗ 未命中：{tag}（{len(hits)} 次容忍匹配）")
            sys.exit(1)
        if n > 1:
            print(f"  ✗ 原文不唯一（{n} 次）：{tag}")
            sys.exit(1)
        s = s.replace(old, new, 1)
        print(f"  ✓ {tag}")

    print(f"\n词数 {len(orig.split())} -> {len(s.split())}")
    if a.dry:
        return
    if s == orig:
        print("稿件无变化。")
        return
    MS.write_text(s)
    print(f"已写入 {MS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
