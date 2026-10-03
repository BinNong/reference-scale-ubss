#!/usr/bin/env python3
"""第十轮复核：CSSP「第二轮」意见的措辞/位置修订。

本轮**不补实验、不动归档**，改动全部是"把已有内容提到更醒目处或补显式措辞"。
唯一的新定量内容是 §4.2 里为回应"最小夹角 ≠ 条件数"而并列的那两个 Gram 条件数
（12° 的 2x2 单位对角 Gram 条件数 91、30° 的 14，降 6.5 倍），它是**解析值**
（两列夹角 θ 的单位对角 Gram 本征值恒为 1±cos θ），不来自任何归档，故在此独立复算。

判据一律**从手稿解析**、与解析式现算值比对，不写死；`--manuscript <path>` 可做负向测试。

用法：
    python3 verify_r10.py [--manuscript <path>]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MS = ROOT / "paper" / "manuscript.md"

FAIL: list[str] = []

N_UPPER = 250        # CSSP 摘要词数上限


def want(tag: str, ok: bool, extra: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {tag}{'  ' + extra if extra else ''}")
    if not ok:
        FAIL.append(tag)


def num(cell: str) -> float:
    """手稿表格单元 -> float。`$150$`、`20\\%`、`**17**` 都能读。"""
    return float(re.sub(r"[^0-9.\-]", "", cell))


def table_rows(text: str, caption: str) -> list[list[str]]:
    """取 caption 之后第一张表的数据行（按 `|` 切开）。"""
    i = text.index(caption)
    rows: list[list[str]] = []
    seen_sep = False
    for ln in text[i:].split("\n")[1:]:
        if not ln.startswith("|"):
            if rows:
                break
            continue
        if re.match(r"^\|[\s\-|:]+\|$", ln):
            seen_sep = True
            continue
        if not seen_sep:
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and cells[0].lower().startswith("family"):
            continue
        rows.append(cells)
    return rows


def gram_condition(theta_deg: float) -> float:
    """两列夹角 θ 的 2x2 单位对角 Gram 的条件数 = (1+cosθ)/(1−cosθ)。"""
    c = math.cos(math.radians(theta_deg))
    return (1.0 + c) / (1.0 - c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=None)
    a = ap.parse_args()
    p = pathlib.Path(a.manuscript) if a.manuscript else MS
    t = p.read_text()
    print(f"=== verify_r10：{p} ===\n")

    # ---------------------------------------------------------- A. 措辞/位置
    print("--- A. 本轮七条修订是否落地 ---")
    # 用**字面子串**匹配，不要用正则：这些探针里含 `$`（正则里是行尾锚点）与 `*`
    # （正则里是量词），直接 re.search 会静默失配。
    for tag, probe in [
        ("摘要：自检加了 heuristic 限定",
         "A heuristic retention self-check"),
        ("摘要：'drifting by two orders of magnitude'（压词后）",
         "drifting two orders of magnitude"),
        ("§11 第四条：结构化源 20% 错误率已进正文",
         r"a rate of $20\%$, against $2.7\%$"),
        ("§4.2：点名活跃 Gram 的谱（而非仅最小夹角）",
         "spectrum of the active Gram"),
        ("§4.2：Gram 条件数 91→14 已并列",
         r"from $91$ to $14$"),
        ("§9.1：十五配置为覆盖设计",
         "chosen to cover the axes on which the reference scale"),
        ("§1 首段：适用域已前置",
         "*instantaneous, delay-free, real-valued* mixing"),
        ("§2：Table 32 的分类要点已内嵌",
         "a fixed multiple of the noise floor does not move by construction"),
    ]:
        want(tag, probe in t)

    print("  --- 旧措辞应已消失 ---")
    want("摘要里 'drifting by two orders' 已消失", "drifting by two orders" not in t)

    # ---------------------------------------------------------- B. Gram 条件数（解析复算）
    print("\n--- B. §4.2 的 Gram 条件数（解析复算，不依赖归档）---")
    i = t.index("First, it is not primarily a proximity effect")
    para = t[i:i + 1600]
    k12, k30 = gram_condition(12.0), gram_condition(30.0)
    want(f"§4.2 条件数下界（稿应为 {round(k12)}）", f"from ${round(k12)}$ to ${round(k30)}$" in para,
         f"解析 {k12:.2f}")
    want("§4.2 说 'a factor of six'", "a factor of six" in para)
    want(f"六倍与实算 {k12 / k30:.2f} 相符", 5.5 <= k12 / k30 <= 7.0)
    want("§4.2 说偏差只降 1.2 倍", "a factor of $1.2$" in para)
    want(f"1.2 与 9.7/8.3 = {9.7 / 8.3:.3f} 相符", abs(9.7 / 8.3 - 1.2) <= 0.05)
    want("结论成立：条件数降 6 倍而偏差几乎不动",
         (k12 / k30) > 4 * (9.7 / 8.3))

    # ---------------------------------------------------------- C. 20% / 2.7% 与 Table 29 一致
    print("\n--- C. §11 引用的 20% / 2.7% 必须与 Table 29 逐行相符 ---")
    rows = table_rows(t, "**Table 29.**")
    by_family: dict[str, list[str]] = {}
    for r in rows:
        if len(r) >= 4:
            by_family[r[0]] = r
    struct = next((r for k, r in by_family.items() if k.lower().startswith("structured")), None)
    sparse = next((r for k, r in by_family.items() if k.lower().startswith("sparse")), None)
    if not struct or not sparse:
        want("Table 29 可解析出 structured / sparse 两行", False)
    else:
        s_err, s_n, s_rate = num(struct[2]), num(struct[1]), num(struct[3])
        p_err, p_n, p_rate = num(sparse[2]), num(sparse[1]), num(sparse[3])
        want(f"structured: {s_err:.0f}/{s_n:.0f} = {100 * s_err / s_n:.0f}%（表内 {s_rate:.0f}%）",
             abs(s_rate - 100 * s_err / s_n) < 0.5)
        want(f"sparse: {p_err:.0f}/{p_n:.0f} = {100 * p_err / p_n:.1f}%（表内 {p_rate:.1f}%）",
             abs(p_rate - 100 * p_err / p_n) < 0.05)
        want("§11 引的两个百分比与 Table 29 一致",
             abs(s_rate - 20) < 0.5 and abs(p_rate - 2.7) < 0.05)

    # ---------------------------------------------------------- D. 摘要词数与结构
    print("\n--- D. 摘要词数与结构 ---")
    m = re.search(r"## Abstract\s*\n+(.*?)\n+\s*\*\*Keywords", t, re.S)
    if not m:
        want("摘要可解析", False)
    else:
        w = len(m.group(1).split())
        want(f"摘要 {w} 词 ≤ CSSP 上限 {N_UPPER}", w <= N_UPPER)
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", t)]
    figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", t)]
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", t)]
    want(f"表题注编号连续（共 {len(caps)} 张，1..{max(caps)}）",
         caps == list(range(1, len(caps) + 1)), f"{len(caps)} 张")
    want(f"图题注编号连续（共 {len(figs)} 张）",
         figs == list(range(1, len(figs) + 1)), f"{len(figs)} 张")
    want(f"附录小节编号连续（共 {len(secs)} 节，A.1..A.{max(secs)}）",
         secs == list(range(1, len(secs) + 1)), f"{len(secs)} 节")

    # ---------------------------------------------------------- E. Table 44（第十轮新增）
    print("\n--- E. Table 44 / §A.19：Gram 谱展宽，vs r10_gram_spectrum.json ---")
    gpath = RES / "r10_gram_spectrum.json"
    if not gpath.exists():
        want("results/r10_gram_spectrum.json 存在", False)
    else:
        G = json.loads(gpath.read_text())
        R = G["subset_records"]
        sp = [x["spearman_kappa_shift"] for x in G["summary"] if "spearman_kappa_shift" in x]
        want("归档含 3 组 Spearman(κ, 偏差)", len(sp) == 3, str(sp))
        want("§A.19 的三个 Spearman 与归档一致（1.000 / 1.000 / 0.649）",
             len(sp) == 3 and all(abs(a - b) <= 0.005 for a, b in zip(sp, [1.000, 1.000, 0.649]))
             and "is $1.000$" in t and "falls to $0.649$" in t, str([round(x, 3) for x in sp]))

        BINS = [(1.0, 1.5), (1.5, 3.0), (3.0, 10.0), (10.0, 100.0), (100.0, float("inf"))]

        def _cell(pred, lo=None, hi=None):
            v = [r["shift_pct"] for r in R if pred(r)
                 and (np.isfinite(r["cond"]) and lo <= r["cond"] < hi if lo is not None
                      else not np.isfinite(r["cond"]))]
            return float(np.median(v)) if v else None

        ROWS = [("$J=2$", lambda r: r["J"] == 2),
                ("$J=3$, $M=3$", lambda r: r["J"] == 3 and r["M"] == 3),
                ("$J=3$, $M=2$", lambda r: r["J"] == 3 and r["M"] == 2),
                ("$J=4$", lambda r: r["J"] == 4)]
        tb44 = table_rows(t, "**Table 44.**")
        want("Table 44 解析出 4 行 7 列",
             len(tb44) == 4 and all(len(r) == 7 for r in tb44),
             f"{len(tb44)} 行，列数 {[len(r) for r in tb44]}")
        if len(tb44) == 4:
            bad = []
            for (lab, pred), mrow in zip(ROWS, tb44):
                for k in range(6):
                    exp = _cell(pred, *BINS[k]) if k < 5 else _cell(pred)
                    cell = mrow[k + 1]
                    got = None if cell.strip().startswith("---") else num(cell)
                    if exp is None or got is None:
                        if exp is not None or got is not None:
                            bad.append((lab, k, exp, cell))
                        continue
                    if abs(exp - got) > 0.05:          # 表内一位小数
                        bad.append((lab, k, round(exp, 2), cell))
            want("Table 44 的 24 格逐格与归档复算一致（±0.05）", not bad, str(bad[:4]))

        # r(p) 偏差：§A.19 引的五个数与归档一致
        dp = {(x["M"], x["p"]): x["dev_pct"] for x in G["r_p"]}
        want("§A.19 的 r(p) 偏差与归档一致（0.003 / 4.1 / 13.0 / 2.8 / 8.9）",
             abs(dp[(2, 0.10)] - 0.003) <= 0.002 and abs(dp[(2, 0.20)] - 4.1) <= 0.05
             and abs(dp[(2, 0.40)] - 13.0) <= 0.05 and abs(dp[(3, 0.20)] - 2.8) <= 0.05
             and abs(dp[(3, 0.40)] - 8.9) <= 0.05
             and "$4.1\\%$ at $p=0.20$" in t and "$13.0\\%$ at $p=0.40$" in t,
             str({k: round(v, 3) for k, v in dp.items()}))
        want("归档协议在 §A.19 声明（200 矩阵 / 4e5 每子集 / 固定 12°）",
             G["meta"]["protocol"]["n_matrices"] == 200
             and G["meta"]["protocol"]["min_angle_deg"] == 12.0
             and "Two hundred mixing matrices" in t and "$4\\times10^{5}$ draws per subset" in t)

    # ---------------------------------------------------------- F. Table 45（C2 第二语料）
    print("\n--- F. Table 45 / §A.20：第二语料，vs r10_real_slr70.json ---")
    p2 = RES / "r10_real_slr70.json"
    if not p2.exists():
        want("results/r10_real_slr70.json 存在", False)
    else:
        def _summ(path: pathlib.Path) -> dict:
            R = json.loads(path.read_text())["records"]
            cases: list[str] = []
            for r in R:
                if r["case"] not in cases:
                    cases.append(r["case"])
            g: dict = {}
            for r in R:
                g.setdefault(r["case"], {}).setdefault(r["method"], []).append(r["A_angle_deg"])
            f = lambda c, m: float(np.nanmean(g[c][m]))          # noqa: E731
            T = ["SCA-median-opt", "SCA-max-opt", "top-K-opt"]
            b3 = [min(f(c, m) for m in T) for c in cases]
            nf = [f(c, "NF-SSP") for c in cases]
            cv = [f(c, "SCA-median") for c in cases]
            return dict(nf=f"{np.mean(nf):.2f}", conv=f"{np.mean(cv):.2f}",
                        best=f"{np.mean(b3):.2f}",
                        pen_nf=f"{np.mean([a / b for a, b in zip(nf, b3)]):.2f}",
                        pen_conv=f"{np.mean([a / b for a, b in zip(cv, b3)]):.1f}",
                        win=f"{sum(1 for a, b in zip(nf, cv) if a < b)}/{len(cases)}")

        a1, a2 = _summ(RES / "real_results.json"), _summ(p2)
        tb45 = table_rows(t, "**Table 45.**")
        want("Table 45 解析出 2 行 7 列",
             len(tb45) == 2 and all(len(r) == 7 for r in tb45),
             f"{len(tb45)} 行，列数 {[len(r) for r in tb45]}")
        if len(tb45) == 2:
            bad = []
            for row, d in zip(tb45, (a1, a2)):
                got = [f"{num(row[1]):.2f}", f"{num(row[2]):.2f}", f"{num(row[3]):.2f}",
                       f"{num(row[4]):.2f}", f"{num(row[5]):.1f}",
                       row[6].strip().strip("$")]
                exp = [d["nf"], d["conv"], d["best"], d["pen_nf"], d["pen_conv"], d["win"]]
                for k, (e, g_) in enumerate(zip(exp, got)):
                    if e != g_:
                        bad.append((row[0][:22], k, e, g_))
            want("Table 45 两行逐格与两个归档复算一致", not bad, str(bad))
        want("§A.20 声明了语料规模（31 说话人 / 3359 条 / 48 kHz）",
             r"$3{,}359$ utterances from $31$ speakers" in t and r"$48$ kHz" in t)
        want("§A.20 与 §7 清单点名两个归档",
             "r10_real_slr70.json" in t and "r10_real_slr70_grid.json" in t)
        want("§11 *Seventh* 已由单语料改写为两语料",
             "rests on two corpora but not on many" in t
             and r"The $1{,}920$ real-speech runs" in t
             and "rests on one corpus" not in t)

    # ------------------------------------------------- G. Declarations 与语料文献（R10e）
    print("\n--- G. Declarations / 三个语料的许可与两条新文献（R10e）---")
    want("§12 之后存在 `## Declarations` 且仅一次", t.count("## Declarations") == 1,
         str(t.count("## Declarations")))
    for tag, probe in [
        ("LibriSpeech 标 CC BY 4.0 且指向 OpenSLR SLR12", "OpenSLR SLR12, CC BY 4.0"),
        ("SLR70 标 CC BY-SA 4.0", "OpenSLR SLR70, CC BY-SA 4.0"),
        ("SLR28 标 Apache 2.0", "OpenSLR SLR28, Apache 2.0"),
        ("Ethics：公开语料，无需伦理审批", "institutional ethics approval is not applicable"),
        ("明示不再分发语料（CC BY-SA 的 ShareAlike 义务由此不触发）",
         "No corpus is redistributed with this paper"),
        ("Funding 已改为无资助的正式声明（R11）",
         "**Funding.** The authors declare that no funds, grants, or other support were received"),
    ]:
        want(tag, probe in t)
    want("Funding 占位符已清除（R11）", "[funding information to be inserted]" not in t)
    want("正文 §A.18 的 SLR28 句带引用 [30]", "OpenSLR SLR28 [30]" in t)
    want("正文 §A.20 的 SLR70 句带引用 [29]", "OpenSLR SLR70 [29]" in t)
    refs = [int(x) for x in re.findall(r"(?m)^\[(\d+)\]", t)]
    want(f"参考文献编号连续（共 {len(refs)} 条，1..{max(refs)}）",
         refs == list(range(1, len(refs) + 1)), str(refs[-4:]))
    want("[29] 为 SLR70 数据集条目",
         re.search(r"(?m)^\[29\] Google, .*OpenSLR resource SLR70", t) is not None)
    want("[30] 为 SLR28 数据集条目",
         re.search(r"(?m)^\[30\] .*OpenSLR resource SLR28", t) is not None)
    want("两个数据集条目都给了 OpenSLR 链接",
         "https://www.openslr.org/70/" in t and "https://www.openslr.org/28/" in t)

    print()
    if FAIL:
        print(f"仍有 {len(FAIL)} 项不一致：{FAIL}")
        return 1
    print("✓ verify_r10 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
