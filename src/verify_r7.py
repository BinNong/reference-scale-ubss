#!/usr/bin/env python3
"""复核第七轮修订引入的全部改动。对应 `apply_r7_fixes.py`。

**判据一律是"从手稿里读出的字面值 vs 从归档重算的值"**，不把数字写死在脚本里。
本轮修订是**纯增补 + 一处措辞**：不改任何既有数字，因此这里除了新增的 Tables 40--41，
还专门加了**跨表一致性**与**措辞替换**两类检查——前者证明新表与原表同源，后者证明旧措辞
确实被替换而非仅仅被追加。

检查项：
  A  Table 40（逐配置 SIR，16 行）：逐格 vs `real_results.json`；配置标签与行序必须与 Table 15
     逐行相同（同一批记录、同一名义量）。
  B  Table 41（三个指标的配置均值）：SIR 行必须与 Table 40 的 mean 行逐格相同（同一名义量）；
     angle 行必须与 Table 15 的 mean 行前四格相同；SDR 行 vs 归档；Oracle-A 的 angle 格为 "—"。
  C  正文文字断言：15/15 与 14/15 两个计数、3.36 dB 与 0.31 dB 两个差值、配对检验统计量、
     以及 §9.4 引到的 SDR 区间必须真的能在表里找到。
  D  措辞替换：Fig. 10 题注的旧句必须消失、新句必须出现；§A.20 与 §9.4 的指针句必须在。
  E  图 11 题注的构建指令标记 `(fig_real_summary.pdf)` 必须仍在（删了 build_pdf.py 会 die）。

用法（在 src/ 下）：
    python3 verify_r7.py
    python3 verify_r7.py --manuscript /tmp/mutated.md      # 负向测试用
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "results"
MS = HERE.parent / "paper" / "manuscript.md"
FAIL: list[str] = []

METH = ["NF-SSP", "SCA-median", "SCA-max", "top-K", "Oracle-A"]


def tables(path: pathlib.Path) -> dict[int, list[list[str]]]:
    lines = path.read_text().split("\n")
    out: dict[int, list[list[str]]] = {}
    for i, l in enumerate(lines):
        mo = re.match(r"^\*\*Table (\d+)\.", l)
        if not mo:
            continue
        n = int(mo.group(1))
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        rows = []
        while j < len(lines) and lines[j].startswith("|"):
            cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
            j += 1
        out[n] = rows
    return out


def num(cell: str) -> float | None:
    t = cell.replace("**", "").replace("$", "").replace("{,}", "").replace(",", "")
    t = t.replace("\\%", "%").replace("\\", "")
    if t.strip() in ("", "---", "-", "—"):
        return None
    if t.endswith("%"):
        t = t[:-1]
    mo = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(mo.group(0)) if mo else None


def dec(cell: str) -> int:
    t = cell.replace("**", "").replace("$", "").replace("{,}", "").replace(",", "").replace("\\", "")
    mo = re.search(r"-?\d+\.(\d+)", t)
    return len(mo.group(1)) if mo else 0


def cmp_cell(label: str, cell: str, actual, scale: float = 1.0) -> None:
    claimed = num(cell)
    if claimed is None:
        return
    if actual is None or not np.isfinite(actual):
        FAIL.append(f"{label}: 稿 {cell!r} 但重算无值")
        print(f"  FAIL {label:<56} 稿 {cell!r} 重算无值")
        return
    tol = 0.5 * 10 ** (-dec(cell)) * scale + 1e-9
    got = actual * scale
    if abs(claimed - got) > tol:
        FAIL.append(f"{label}: 稿 {claimed} vs 重算 {got:g}")
        print(f"  FAIL {label:<56} 稿 {claimed} 重算 {got:g} (tol {tol:g})")
    else:
        print(f"  ok   {label:<56} {claimed}")


def want(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        FAIL.append(f"{label} {detail}")
        print(f"  FAIL {label} {detail}")


# ------------------------------------------------------------------ 归档
def archive():
    """{case: {method: {'sir': mean, 'sdr': mean, 'ang': mean}}}，逐 seed 平均。"""
    hdr = json.loads((R / "real_results.json").read_text())
    recs = hdr["records"]
    acc: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in recs:
        if r["method"] not in METH:
            continue
        d = acc.setdefault(r["case"], {}).setdefault(
            r["method"], {"sir": [], "sdr": [], "ang": []})
        d["sir"].append(r["SIR"])
        d["sdr"].append(r["SDR"])
        d["ang"].append(r["A_angle_deg"])
    out = {c: {m: {k: float(np.mean(v)) for k, v in mm.items()}
               for m, mm in cm.items()} for c, cm in acc.items()}
    return hdr, out


# ------------------------------------------------------- A. Table 40 逐配置 SIR
def table40(tb, A) -> None:
    print("\n=== A. Table 40：逐配置 SIR vs real_results.json ===")
    t40, t15 = tb.get(40), tb.get(15)
    if not t40 or not t15:
        FAIL.append("Table 40 或 Table 15 缺失")
        print("  FAIL Table 40 / Table 15 缺失")
        return
    want("Table 40 行数 = 表头+15+均值", len(t40) == 17, f"实为 {len(t40)}")
    want("Table 40 列数 = 配置+5 方法", all(len(r) == 6 for r in t40),
         f"列数 {[len(r) for r in t40]}")

    print("  --- 行序与标签 vs Table 15（同一批记录的顺序必须一致） ---")
    for i in range(1, 16):
        want(f"  行{i:>2} 标签 == Table 15", t40[i][0] == t15[i][0],
             f"{t40[i][0]!r} vs {t15[i][0]!r}")

    print("  --- 逐格 vs 归档 ---")
    for i in range(1, 16):
        case = CASE[t40[i][0]]
        for j, m in enumerate(METH):
            cmp_cell(f"  {t40[i][0]:<18} {m:<10} SIR", t40[i][j + 1], A[case][m]["sir"])

    print("  --- 均值行 ---")
    for j, m in enumerate(METH):
        cmp_cell(f"  mean {m:<14} SIR", t40[16][j + 1],
                 float(np.mean([A[CASE[t40[i][0]]][m]["sir"] for i in range(1, 16)])))


# ------------------------------------------------- B. Table 41 与跨表一致性
def table41(tb, A) -> None:
    print("\n=== B. Table 41：指标均值 + 跨表一致性 ===")
    t41, t40, t15 = tb.get(41), tb.get(40), tb.get(15)
    if not t41:
        FAIL.append("Table 41 缺失")
        print("  FAIL Table 41 缺失")
        return
    want("Table 41 行数 = 表头+3", len(t41) == 4, f"实为 {len(t41)}")
    want("Table 41 列数 = 指标+5 方法", all(len(r) == 6 for r in t41))

    mean = {m: float(np.mean([A[CASE[t40[i][0]]][m]["sir"] for i in range(1, 16)]))
            for m in METH}
    msdr = {m: float(np.mean([A[CASE[t40[i][0]]][m]["sdr"] for i in range(1, 16)]))
            for m in METH}
    mang = {m: float(np.mean([A[CASE[t40[i][0]]][m]["ang"] for i in range(1, 16)]))
            for m in METH}

    rows = {r[0].replace("$", "").replace("\\", ""): r for r in t41[1:]}
    want("Table 41 有 angle / SDR / SIR 三行",
         len(rows) == 3, f"实为 {list(rows)}")
    if len(rows) != 3:
        return
    r_ang = rows.get("angle error (°)")
    r_sdr = rows.get("SDR (dB)")
    r_sir = rows.get("SIR (dB)")
    if not (r_ang and r_sdr and r_sir):
        FAIL.append(f"Table 41 行标签: {list(rows)}")
        print(f"  FAIL 行标签 {list(rows)}")
        return

    print("  --- 与归档 ---")
    for j, m in enumerate(METH):
        cmp_cell(f"  angle {m:<12}", r_ang[j + 1], mang[m])
        cmp_cell(f"  SDR   {m:<12}", r_sdr[j + 1], msdr[m])
        cmp_cell(f"  SIR   {m:<12}", r_sir[j + 1], mean[m])

    print("  --- 跨表一致性（同一名义量必须逐格相同） ---")
    for j, m in enumerate(METH):
        cmp_cell(f"  SIR 行 vs Table 40 mean  {m:<10}", r_sir[j + 1], mean[m])
    for j, m in enumerate(METH[:4]):
        cmp_cell(f"  angle 行 vs Table 15 mean {m:<10}", r_ang[j + 1], mang[m])
        if num(r_ang[j + 1]) is not None and num(t15[16][j + 1]) is not None:
            want(f"  angle 行 == Table 15 mean（字面） {m}",
                 abs(num(r_ang[j + 1]) - num(t15[16][j + 1])) < 5e-3)
    want("Oracle-A 的 angle 格是 '—'（构造上为 0，不作为竞争者测量）",
         num(r_ang[5]) is None, f"实为 {r_ang[5]!r}")


# ------------------------------------------------------------- C/D/E 文字断言
def wording(text: str, tb) -> None:
    print("\n=== C. 正文数字断言（必须能从表里找到） ===")
    t40, t41 = tb.get(40), tb.get(41)
    if not t41:
        FAIL.append("Table 41 缺失：无法核对 §9.4 引用的 SDR 区间")
        print("  FAIL Table 41 缺失，跳过 C 组的表格侧核对")
        return
    flat = text.replace("$", "")          # 手稿把数字包在 $ 里，比对前先剥掉
    want("mean SIR 差 NF−median = 3.36", "3.36 dB" in flat)
    want("mean SIR 差 max−NF = 0.31", "by 0.31 dB" in flat)
    want("NF 领先 median 的配置数写作 all fifteen",
         "proposed gate exceeds it in all fifteen" in flat)
    want("max 领先 NF 的配置数写作 fourteen of the fifteen",
         "fourteen of the fifteen configurations" in flat)
    want("配对统计量 t=6.78 与 p=9e-6 在文中",
         "t=6.78" in flat and "9\\times10^{-6}" in text)
    want("SIR 均值跨度 0.31 与 SDR 均值跨度 0.08 都在文中",
         "span only 0.31 dB" in flat and "only 0.08 dB" in flat)

    print("  --- §9.4 引用的 SDR 区间必须与 Table 41 的 SDR 行相符 ---")
    three = [num(t41[2][j]) for j in (1, 3, 4)]          # NF / max / top-5%（行已含标签列）
    want("§9.4 的 4.85–4.93 区间成立",
         abs(min(three) - 4.85) < 5e-3 and abs(max(three) - 4.93) < 5e-3,
         f"实为 {min(three)}–{max(three)}")
    want("§9.4 的 oracle 4.95 成立", abs(num(t41[2][5]) - 4.95) < 5e-3)
    want("§9.4 的 'to within 0.1 dB' 成立",
         max(num(t41[2][5]) - x for x in three) <= 0.1 + 1e-9,
         f"实为 {max(num(t41[2][5]) - x for x in three):.3f}")

    print("\n=== D/E. 措辞替换与构建指令标记 ===")
    want("Fig. 10 题注旧句 'positive values mean the calibrated gate is better' 已消失",
         "positive values mean the calibrated gate is better" not in text)
    want("Fig. 10 题注新句已在",
         "a positive value indicates that the proposed calibrated gate" in text
         and "attains the lower angle error" in text)
    want("§A.17 小节标题在", "### A.17 Separation-quality metrics on real speech" in text)
    want("§9.4 指向 §A.17 的句子在",
         "Appendix A.17 reports the signal-to-interference ratio" in text)
    want("Fig. 10 题注仍保留构建指令标记 (fig_real_summary.pdf)",
         "(`fig_real_summary.pdf`)" in text)
    want("SAR 的省略已在正文明说（不假装报了三件套）",
         "the archived runs record SDR and SIR only" in text)


# ------------------------------------ G. §9.4 那句 SDR 亏损：独立于本轮的一条旧缺陷
def section94_sdr(text: str, A) -> None:
    """§9.4 末段 "The conventional gate loses $2.2$ dB of SDR on average and up to $4.5$ dB."

    "2.2" 可复算（NF−median 的逐配置配对差均值 = 2.177）；"4.5" **复算不出来**：
    试遍全部方法对 × {逐配置, 逐实例} × {mean, median, min, max} × 4-seed 子集，
    自然口径的上限只有 3.28（NF−median）与 3.34（oracle−median）。既有 verify_r5/r6
    对这句话没有任何断言，所以它一直没被覆盖。本条按同一配对口径断言，令其失败可见。
    """
    print("\n=== G. §9.4 的 SDR 亏损（本轮之前就存在、此前无人断言） ===")
    cases = [CASE[t] for t in CASE if CASE[t] in A]
    d = [A[c]["NF-SSP"]["sdr"] - A[c]["SCA-median"]["sdr"] for c in cases]
    mean, mx = float(np.mean(d)), float(max(d))
    worst = cases[int(np.argmax(d))]
    flat = text.replace("$", "")
    print(f"  复算（NF−median，逐配置）: mean {mean:.3f}  max {mx:.3f} (在 {worst})")
    want("『2.2 dB on average』= NF−median 的配对差均值", abs(mean - 2.2) < 0.06,
         f"复算 {mean:.3f}")
    want("『up to X dB』= NF−median 的配对差最大值(3.28)",
         ("up to 3.3 dB" in flat) or ("up to 3.28 dB" in flat),
         "手稿仍写着其它值；复算上限为 3.28，见本函数 docstring")


# ------------------------------------------------ F. 计数与排序的可复算断言
def counts(text: str, A) -> None:
    print("\n=== F. 两个计数与排序断言（手稿论断 vs 归档重算） ===")
    cases = [CASE[t] for t in CASE if CASE[t] in A]
    win = sum(1 for c in cases if A[c]["NF-SSP"]["sir"] > A[c]["SCA-median"]["sir"])
    lose = sum(1 for c in cases if A[c]["SCA-max"]["sir"] > A[c]["NF-SSP"]["sir"])
    last = sum(1 for c in cases
               if all(A[c]["SCA-median"]["sir"] <= A[c][m]["sir"] for m in METH))
    want("NF-SSP 的 SIR 在全部 15 个配置上高于 conventional", win == 15, f"实为 {win}")
    want("max-referenced 的 SIR 在 14/15 个配置上高于 NF-SSP", lose == 14, f"实为 {lose}")
    want("conventional 在全部 15 个配置上 SIR 垫底", last == 15, f"实为 {last}")
    a = np.array([A[c]["NF-SSP"]["sir"] for c in cases])
    b = np.array([A[c]["SCA-median"]["sir"] for c in cases])
    t = stats.ttest_rel(a, b)
    want("配对 t（n=15）t≈6.78", abs(t.statistic - 6.78) < 0.01, f"实为 {t.statistic:.4f}")
    want("配对 p ≈ 9e-6", abs(t.pvalue - 9e-6) / 9e-6 < 0.1, f"实为 {t.pvalue:.3g}")
    three = ['NF-SSP', 'SCA-max', 'top-K']
    msir = [float(np.mean([A[c][m]["sir"] for c in cases])) for m in three]
    want("三层规则（NF/max/top-K）的 SIR 均值跨度 = 0.31 dB",
         abs((max(msir) - min(msir)) - 0.31) < 5e-3, f"实为 {max(msir) - min(msir):.4f}")
    msdr = [float(np.mean([A[c][m]["sdr"] for c in cases])) for m in three]
    want("三层规则（NF/max/top-K）的 SDR 均值跨度 = 0.08 dB",
         abs((max(msdr) - min(msdr)) - 0.08) < 5e-3, f"实为 {max(msdr) - min(msdr):.4f}")

CASE = {
    "win 256": "R1_res:w256", "win 512": "R1_res:w512", "win 1024": "R1_res:w1024",
    "win 2048": "R1_res:w2048", "$N=3$": "R2_n:N3", "$N=5$": "R2_n:N5", "$N=6$": "R2_n:N6",
    "SNR $0$ dB": "R3_snr:snr00", "SNR $10$ dB": "R3_snr:snr10",
    "SNR $30$ dB": "R3_snr:snr30", "SNR $40$ dB": "R3_snr:snr40",
    "1 dense source": "R4_dense:d1", "2 dense sources": "R4_dense:d2",
    "babble noise": "R5_noise:babble", "noise-free": "R5_noise:clean",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=str(MS))
    a = ap.parse_args()
    ms = pathlib.Path(a.manuscript)
    text = ms.read_text()
    tb = tables(ms)

    # —— R8 表号整体重排后的兼容垫片（旧号 -> 新号），见 verify_r5.py 同名注释 ——
    _R8_TAB = {19: 23, 20: 24, 21: 25, 22: 26, 23: 19, 24: 20, 25: 27, 26: 28,
               27: 29, 28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36,
               35: 37, 36: 38, 37: 39, 38: 40, 39: 21, 40: 41, 41: 42}
    tb.update({o: tb[n] for o, n in _R8_TAB.items() if n in tb})
    hdr, A = archive()

    print(f"手稿 {ms}  |  归档 real_results.json  n_seeds={hdr['n_seeds']}")
    want("归档 n_seeds = 8（表注称 eight seeds）", hdr["n_seeds"] == 8)
    want("归档含 15 个配置 × 8 方法 × 8 seed = 960 条",
         len(hdr["records"]) == 960, f"实为 {len(hdr['records'])}")

    table40(tb, A)
    table41(tb, A)
    wording(text, tb)
    counts(text, A)
    section94_sdr(text, A)

    print()
    if FAIL:
        print(f"✗ 失败 {len(FAIL)} 项：")
        for f in FAIL:
            print(f"    · {f}")
        sys.exit(1)
    print("✓ verify_r7 全部通过")


if __name__ == "__main__":
    main()
