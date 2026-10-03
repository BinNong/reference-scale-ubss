#!/usr/bin/env python3
"""第九轮复核：CSSP Minor Revision 的 oracle 协议修正所波及的**全部数字**。

背景：意见指出"real-speech oracle 是 test-data 还是 separate seeds"三处描述互不相同；
核查代码发现调参种子 {0,1} 是评测种子 {0..7} 的子集（部分泄漏）。已改为
`range(seeds, seeds+tune_seeds)` = 种子 {8,9}（与评测集不相交）并重跑归档。
本脚本逐条复核由此变化的数字，并**逐行**核对两张受影响的表（既有脚本只验汇总量）。

判据一律**从手稿解析**、与归档现算的值比对，不写死——写死只验证"记忆"。
给 `--manuscript <path>` 可做负向测试。

用法：
    python3 verify_r9.py [--manuscript <path>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MS = ROOT / "paper" / "manuscript.md"

FAIL: list[str] = []
CONF = ["win 256", "win 512", "win 1024", "win 2048", "$N=3$", "$N=5$", "$N=6$",
        "SNR $0$ dB", "SNR $10$ dB", "SNR $30$ dB", "SNR $40$ dB",
        "1 dense source", "2 dense sources", "babble noise", "noise-free"]
CASES = [f"{e}:{t}" for e, t in
         [("R1_res", "w256"), ("R1_res", "w512"), ("R1_res", "w1024"), ("R1_res", "w2048"),
          ("R2_n", "N3"), ("R2_n", "N5"), ("R2_n", "N6"),
          ("R3_snr", "snr00"), ("R3_snr", "snr10"), ("R3_snr", "snr30"), ("R3_snr", "snr40"),
          ("R4_dense", "d1"), ("R4_dense", "d2"), ("R5_noise", "babble"), ("R5_noise", "clean")]]


def want(tag: str, ok: bool, extra: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {tag}{'  ' + extra if extra else ''}")
    if not ok:
        FAIL.append(tag)


def num(cell: str) -> float:
    """手稿表格单元 -> float，容忍 `**1.23**` 与 `$1.23$`。"""
    return float(re.sub(r"[^0-9.\-]", "", cell))


def table_rows(text: str, caption: str) -> list[list[str]]:
    """取 caption 之后第一张表的**数据行**（已按 `|` 切开）。"""
    i = text.index(caption)
    out = []
    for ln in text[i:].split("\n")[1:]:
        if not ln.startswith("|"):
            if out:
                break
            continue
        if re.match(r"^\|[\s\-|:]+\|$", ln):      # 分隔行
            continue
        out.append(ln.split("|"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=str(MS))
    a = ap.parse_args()
    text = pathlib.Path(a.manuscript).read_text()

    R = json.loads((RES / "real_results.json").read_text())["records"]
    G = json.loads((RES / "real_grid.json").read_text())

    def col(m: str, c: str) -> float:
        return st.mean([r["A_angle_deg"] for r in R if r["case"] == c and r["method"] == m])

    # ---------- Table 14：两个最优系数列，逐行 ----------
    print("\n[Table 14] optimal te / optimal c（逐行 vs real_grid.json 的 star）")
    rows = table_rows(text, "**Table 14.**")
    rows = [r for r in rows if r[1].strip() in CONF]
    want(f"Table 14 行数 = 15", len(rows) == 15, f"实为 {len(rows)}")
    bad14 = []
    for r in rows:
        c = CASES[CONF.index(r[1].strip())]
        star = G[f"{c}|star"]
        if abs(num(r[6]) - star["te"]) > 1e-9 or abs(num(r[7]) - star["c"]) > 1e-9:
            bad14.append((r[1].strip(), num(r[6]), star["te"], num(r[7]), star["c"]))
    want("Table 14 的 te/c 两列逐行与归档一致", not bad14, str(bad14[:3]))

    # ---------- Table 15：三个 tuned 列 + 均值行，逐行 ----------
    print("\n[Table 15] median-opt / max-opt / top-K-opt（逐行 vs real_results.json）")
    rows = table_rows(text, "**Table 15.**")
    data = [r for r in rows if r[1].strip() in CONF]
    mean = [r for r in rows if "mean" in r[1]]
    want("Table 15 行数 = 15", len(data) == 15, f"实为 {len(data)}")
    want("Table 15 有均值行", len(mean) == 1)
    bad15 = []
    for r in data:
        c = CASES[CONF.index(r[1].strip())]
        for k, m in enumerate(("SCA-median-opt", "SCA-max-opt", "top-K-opt")):
            if abs(num(r[6 + k]) - col(m, c)) > 0.005:
                bad15.append((r[1].strip(), m, num(r[6 + k]), round(col(m, c), 3)))
    want("Table 15 的三个 tuned 列逐行与归档一致（±0.005 四舍五入）", not bad15, str(bad15[:3]))
    if mean:
        for k, m in enumerate(("SCA-median-opt", "SCA-max-opt", "top-K-opt")):
            v = st.mean([col(m, c) for c in CASES])
            want(f"Table 15 均值行 {m}", abs(num(mean[0][6 + k]) - v) <= 0.005,
                 f"稿 {num(mean[0][6 + k])} 归档 {v:.3f}")

    # ---------- best-observed 与派生比值（从手稿抓） ----------
    print("\n[汇总量] best-observed 与比值（手稿抓取）")
    b3 = [min(col(m, c) for m in ("SCA-median-opt", "SCA-max-opt", "top-K-opt")) for c in CASES]
    b7 = [min(b, col("SCA-median", c), col("SCA-max", c), col("top-K", c))
          for b, c in zip(b3, CASES)]
    nf = [col("NF-SSP", c) for c in CASES]
    conv = [col("SCA-median", c) for c in CASES]

    def grab(pat: str, name: str) -> float:
        m = re.search(pat, text, re.S)
        if not m:
            want(f"手稿读不到 {name}", False)
            return float("nan")
        return float(m.group(1))

    c_b3 = grab(r"whose mean is \$([\d.]+)°\$", "Table 15 题注 三列均值")
    c_b7 = grab(r"all seven competing columns it would be \$([\d.]+)°\$", "Table 15 题注 七列值")
    c_conv = grab(r"penalty of the conventional baseline against this bound is \$([\d.]+)\\times\$",
                  "Table 15 题注 conventional penalty")
    c_ratio = grab(r"\(ratio of means: \$([\d.]+)\\times\$\)", "Table 15 题注 ratio of means")
    c_nf = grab(r"and that of NF-SSP is \$([\d.]+)\\times\$", "Table 15 题注 NF penalty")
    c_max = grab(r"penalty averages \$[\d.]+\\times\$ and reaches \$([\d.]+)\\times\$", "§9.3 最大 penalty")
    c_bnd = grab(r"against \$([\d.]+)°\$ for the best-observed reference", "§9.3 best-observed")

    want(f"三列最小值的均值（稿 {c_b3}）", abs(c_b3 - st.mean(b3)) <= 0.005, f"归档 {st.mean(b3):.3f}")
    want(f"七列最小值的均值（稿 {c_b7}）", abs(c_b7 - st.mean(b7)) <= 0.005, f"归档 {st.mean(b7):.3f}")
    want(f"conventional penalty（稿 {c_conv}）", abs(c_conv - st.mean([a / b for a, b in zip(conv, b3)])) <= 0.05)
    want(f"ratio of means（稿 {c_ratio}）", abs(c_ratio - st.mean(conv) / st.mean(b3)) <= 0.05)
    want(f"NF penalty（稿 {c_nf}）", abs(c_nf - st.mean([a / b for a, b in zip(nf, b3)])) <= 0.005)
    want(f"最大 penalty（稿 {c_max}）", abs(c_max - max(a / b for a, b in zip(conv, b3))) <= 0.5)
    want(f"§9.3 best-observed（稿 {c_bnd}）", abs(c_bnd - st.mean(b3)) <= 0.005)

    # ---------- 同一量的第二处出现（本轮**目视**才发现的两处漏网） ----------
    # 教训：用"句子片段"做替换时，片段往往没覆盖句内**所有**受影响的数字；
    # 同一名义量在正文里常出现两次（§9.3 同句后半、§12 结论），必须逐处断言。
    print("\n[同量的第二次出现] §9.3 同句后半 / §12 结论")
    c_ratio2 = grab(r"the ratio of the two means would be \$([\d.]+)\\times\$", "§9.3 的 ratio of the two means")
    want(f"§9.3 ratio of the two means（稿 {c_ratio2}）",
         abs(c_ratio2 - st.mean(conv) / st.mean(b3)) <= 0.05)
    c_fold = grab(r"removes a silent \$([\d.]+)\$-fold penalty", "§12 的 penalty")
    want(f"§12 的 penalty（稿 {c_fold}）",
         abs(c_fold - st.mean([a / b for a, b in zip(conv, b3)])) <= 0.05)

    # ---------- §A.7 的平坦度 ----------
    print("\n[§A.7] 系数范围与平坦度")
    cbest = {c: min(G[f"{c}|max"], key=lambda k: G[f"{c}|max"][k]) for c in CASES}
    star_te = [G[f"{c}|star"]["te"] for c in CASES]
    g_lo = grab(r"coefficient runs over\s+\$([\d.]+)\$–\$[\d.]+\$", "§A.7 系数下限")
    g_hi = grab(r"coefficient runs over\s+\$[\d.]+\$–\$([\d.]+)\$", "§A.7 系数上限")
    g_fix = grab(r"coefficient at \$0\.05\$ costs \$([\d.]+)°\$", "§A.7 固定 0.05 均值")
    g_opt = grab(r"on average against \$([\d.]+)°\$ for the\s+per-configuration optimum", "§A.7 逐档最优")
    want(f"§A.7 max 系数下限（稿 {g_lo}）", abs(g_lo - min(float(x) for x in cbest.values())) < 1e-9)
    want(f"§A.7 max 系数上限（稿 {g_hi}）", abs(g_hi - max(float(x) for x in cbest.values())) < 1e-9)
    want(f"§A.7 固定 0.05 均值（稿 {g_fix}）",
         abs(g_fix - st.mean([G[f"{c}|max"]["0.05"] for c in CASES])) <= 0.005)
    want(f"§A.7 逐档最优均值（稿 {g_opt}）",
         abs(g_opt - st.mean([min(G[f"{c}|max"].values()) for c in CASES])) <= 0.005)
    print(f"       median 系数范围 {min(star_te):g}–{max(star_te):g}（稿 §9.5 与 §11 各引一次）")
    for lit in (f"${min(star_te):g}$ to ${max(star_te):g}$", f"${min(star_te):g}$–${max(star_te):g}$"):
        want(f"手稿含区间 {lit!r}", lit in text)

    # ---------- 结构 ----------
    print("\n[结构] 表/图题注齐全")
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", text)]
    want(f"表题注编号连续（共 {len(caps)} 张，1..{max(caps)}）",
         caps == list(range(1, len(caps) + 1)), f"{len(caps)} 张")
    figs = [int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", text)]
    want(f"图题注编号连续（共 {len(figs)} 张）",
         figs == list(range(1, len(figs) + 1)), f"{len(figs)} 张")

    print("\n" + "=" * 74)
    print("✓ verify_r9 全部通过" if not FAIL else f"仍有 {len(FAIL)} 项不一致：{FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
