#!/usr/bin/env python3
"""复核第五轮修订引入的全部新数字：扩展 Table 17、Tables 31--39、Fig. 11。

**判据是"从手稿里读出的字面值 vs 从归档重算的值"**，而不是把数字写死在脚本里再比一遍——
写死会让脚本只验证"我的记忆"（本轮正是这样才差点漏掉三处错）。容差由手稿印出的**小数位**
决定，所以四舍五入错一位必然被抓到。

另含两项**跨表一致性**检查，它们不依赖我的核对习惯，只比手稿内部与归档：
  · Table 17 的 median / max 两列必须与 Table 15 的同名列逐行相同（同一批记录、同一名义量）；
  · Table 17 的 median / max 两列必须与 `real_results.json` 的 SCA-median / SCA-max 相同。

用法（在 src/ 下）：
    python3 verify_r5.py
    python3 verify_r5.py --manuscript /tmp/mutated.md      # 负向测试用
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent / "results"
MS = HERE.parent / "paper" / "manuscript.md"
FAIL: list[str] = []


# ------------------------------------------------------------------ 解析手稿
def tables(path: pathlib.Path) -> dict[int, list[list[str]]]:
    """把稿中的 markdown 表格读成 {表号: [[单元格, ...], ...]}（含表头行，去对齐行）。"""
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
    """把单元格里的数取出来；`---`、空、纯文字返回 None。"""
    t = cell.replace("**", "").replace("$", "").replace("{,}", "").replace(",", "")
    t = t.replace("\\%", "%").replace("\\lvert", "").replace("\\rvert", "")
    t = t.replace("\\", "")
    if t.strip() in ("", "---", "-", "—"):
        return None
    if t.endswith("%"):
        t = t[:-1]
    mo = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(mo.group(0)) if mo else None


def dec(cell: str) -> int:
    """印出的小数位数——用来定容差（四舍五入到 d 位时误差 ≤ 0.5·10^-d）。"""
    t = cell.replace("**", "").replace("$", "").replace("{,}", "").replace(",", "")
    mo = re.search(r"-?\d+\.(\d+)", t)
    return len(mo.group(1)) if mo else 0


def cmp(label: str, cell: str, actual) -> None:            # noqa: A001
    claimed = num(cell)
    if claimed is None:
        return
    if actual is None or not np.isfinite(actual):
        print(f"  FAIL {label:<46} 稿 {cell!r} 但归档无值")
        FAIL.append(label)
        return
    tol = 0.5 * 10 ** (-dec(cell)) + 1e-9
    ok = abs(claimed - float(actual)) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<46} 稿 {claimed:>9}  归档 {float(actual):>9.4f}")
    if not ok:
        FAIL.append(label)


def load(name: str):
    return json.load(open(R / f"{name}.json"))


def mA(recs, method, case, key="A") -> float:
    v = [r[key] for r in recs if r["method"] == method and r["case"] == case]
    return float(np.mean(v)) if v else float("nan")


def m(recs, **sel) -> float:
    v = [r["A"] for r in recs if all(r[k] == val for k, val in sel.items())]
    return float(np.mean(v)) if v else float("nan")


# --------------------------------------------------- 1. 扩展 Table 17（8 列）
T17_CASES = ["R1_res:w256", "R1_res:w512", "R1_res:w1024", "R1_res:w2048",
             "R2_n:N3", "R2_n:N5", "R2_n:N6",
             "R3_snr:snr00", "R3_snr:snr10", "R3_snr:snr30", "R3_snr:snr40",
             "R4_dense:d1", "R4_dense:d2", "R5_noise:babble", "R5_noise:clean"]
T17_METH = ["nf", "nf+w", "nf+w4", "base+w", "med0.02", "max", "max+w4"]
T15_CASES = T17_CASES          # Table 15 的行序与 Table 17 相同


def t17(tb) -> None:
    print("\n=== 扩展 Table 17：七列逐格 vs r5_unified_weight.json ===")
    W = load("r5_unified_weight")["records"]
    rows = tb[17][1:]
    if len(rows) != 16:
        print(f"  FAIL 行数 {len(rows)}（应为 15 配置 + mean）")
        FAIL.append("Table 17 行数")
    for i, c in enumerate(T17_CASES):
        for j, meth in enumerate(T17_METH):
            cmp(f"  {c:<18} {meth:<9}", rows[i][1 + j], mA(W, meth, c))
    for j, meth in enumerate(T17_METH):
        cmp(f"  MEAN               {meth:<9}", rows[15][1 + j], m(W, method=meth))

    print("  --- 跨表一致性：Table 17 的 NF / median / max 列 vs Table 15 的同名列 ---")
    t15 = tb[15][1:]
    for i, c in enumerate(T15_CASES):
        cmp(f"  {c:<18} NF      vs Table 15", rows[i][1], num(t15[i][1]))
        cmp(f"  {c:<18} median  vs Table 15", rows[i][5], num(t15[i][2]))
        cmp(f"  {c:<18} max     vs Table 15", rows[i][6], num(t15[i][3]))

    print("  --- 跨表一致性：median / max 列 vs Table 15 的来源 real_results.json ---")
    RR = load("real_results")["records"]
    for i, c in enumerate(T17_CASES):
        cmp(f"  {c:<18} median  vs 归档", rows[i][5],
            mA(RR, "SCA-median", c, key="A_angle_deg"))
        cmp(f"  {c:<18} max     vs 归档", rows[i][6],
            mA(RR, "SCA-max", c, key="A_angle_deg"))


# ------------------------------------------------- 2. Tables 31 / 32（密度）
DENS_CASES = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
IREAL_CASES = ["R1_res:w1024", "R2_n:N6", "R4_dense:d2", "R5_noise:babble"]


def density(tb) -> None:
    print("\n=== Table 31：合成阶梯上的密度前端竞争者 ===")
    D = load("r5_density_baseline")["records"]
    dens = sorted({r["method"] for r in D if r["method"] not in ("classical", "nf")})
    rows = tb[31][1:]
    picks = ["classical", "nf", None, "dbscan[collin,eps=0.02]",
             "optics[collin,eps=0.02]", "dpeak[collin]"]
    for i, meth in enumerate(picks):
        for j, c in enumerate(DENS_CASES):
            actual = (min(mA(D, x, c) for x in dens) if meth is None
                      else mA(D, meth, c))
            cmp(f"  {rows[i][0][:20]:<21} {c:<9}", rows[i][1 + j], actual)
        cmp(f"  {rows[i][0][:20]:<21} MEAN", rows[i][7],
            float(np.mean([num(rows[i][1 + j]) for j in range(6)])))

    print("  --- Table 32：真实语音子集 ---")
    DR = load("r5_density_real")["records"]
    densr = sorted({r["method"] for r in DR if r["method"] not in ("classical", "nf")})
    rows = tb[32][1:]
    picks = ["classical", "nf", None, "dbscan[collin,eps=0.02]", "dpeak[collin]"]
    for i, meth in enumerate(picks):
        for j, c in enumerate(IREAL_CASES):
            actual = (min(mA(DR, x, c) for x in densr) if meth is None
                      else mA(DR, meth, c))
            cmp(f"  {rows[i][0][:20]:<21} {c:<18}", rows[i][1 + j], actual)
        cmp(f"  {rows[i][0][:20]:<21} MEAN", rows[i][5],
            float(np.mean([num(rows[i][1 + j]) for j in range(4)])))


# ------------------------------------------ 3. Tables 33 / 34 / 35（复值）
CMPLX_KEYS = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
M35_ORDER = ["all", "collin", "median0.02", "nf", "nfE", "nfE+sc"]


def complex_study(tb) -> None:
    print("\n=== Table 33：三种混合矩阵下的参考尺度比 ===")
    C = load("r5_complex")["records"]
    rows = tb[33][1:]
    for i, k in enumerate(CMPLX_KEYS):
        for j, kind in enumerate(("real", "cfixed", "cperfreq")):
            cmp(f"  {k:<9} r ({kind})", rows[i][1 + j],
                float(np.mean([r["r_emp"] for r in C
                               if r["cfg_key"] == k and r["kind"] == kind])))
        cmp(f"  {k:<9} closed form", rows[i][4],
            float(np.mean([r["r_closed"] for r in C
                           if r["cfg_key"] == k and r["kind"] == "real"])))
        cmp(f"  {k:<9} σ̂²/σ²", rows[i][5],
            float(np.mean([r["s2_hat_over_s2"] for r in C
                           if r["cfg_key"] == k and r["kind"] == "real"])))

    print("  --- Table 34：实值共线判据在三种矩阵下的质量 ---")
    rows = tb[34][1:]
    for i, kind in enumerate(("real", "cfixed", "cperfreq")):
        sub = [r for r in C if r["kind"] == kind]
        for j, key in enumerate(("collin_admit_frac", "collin_recall_single",
                                 "collin_precision", "collin_fpr_noise")):
            cmp(f"  {kind:<9} {key[7:]:<16}", rows[i][1 + j],
                float(np.mean([r[key] for r in sub])))

    print("  --- Table 35：六种掩码在复值机器上（及实值对照）---")
    rows = tb[35][1:]
    for i, mask in enumerate(M35_ORDER):
        for j, k in enumerate(CMPLX_KEYS):
            cmp(f"  {mask:<9} {k:<9} (cplx)", rows[i][1 + j],
                float(np.nanmean([r[f"A_cx_{mask}"] for r in C
                                  if r["kind"] == "cfixed" and r["cfg_key"] == k])))
        cmp(f"  {mask:<9} mean (cplx)", rows[i][7],
            float(np.nanmean([r[f"A_cx_{mask}"] for r in C if r["kind"] == "cfixed"])))
        cmp(f"  {mask:<9} mean (real)", rows[i][8],
            float(np.nanmean([r[f"A_cx_{mask}"] for r in C if r["kind"] == "real"])))
        cmp(f"  {mask:<9} retention %", rows[i][9],
            100 * float(np.mean([r[f"keep_{mask}"] for r in C if r["kind"] == "cfixed"])))


# ------------------------------------------------------- 4. Table 36（α）
M36_CELLS = [("tf_p20", 0.0), ("tf_p20", 10.0), ("tf_p10", 0.0), ("tf_p40", 0.0)]
M36_ALPHAS = [0.1, 0.001, 1e-5, 1e-4]        # 表头给出的列序


def alpha_cells(tb) -> None:
    print("\n=== Table 36：地板偏高处的 α 敏感性 ===")
    A = load("r5_alpha_cell")["records"]
    rows = tb[36][1:]
    for i, (cs, snr) in enumerate(M36_CELLS):
        sub = [r for r in A if r["case"] == cs and r["snr_db"] == snr]
        cmp(f"  {cs} SNR{snr:.0f} σ̂²/σ²", rows[i][1],
            float(np.mean([r["s2_ratio"] for r in sub])))
        for j, al in enumerate(M36_ALPHAS):
            parts = [x.strip() for x in rows[i][2 + j].split("/")]
            if len(parts) != 3:
                print(f"  FAIL 单元格格式异常：{rows[i][2 + j]!r}")
                FAIL.append(f"Table 36 单元格 {cs} α={al:g}")
                continue
            g = [r for r in sub if r["alpha"] == al]
            cmp(f"  {cs} SNR{snr:.0f} α={al:g} keep%", parts[0],
                100 * float(np.mean([r["keep_frac"] for r in g])))
            cmp(f"  {cs} SNR{snr:.0f} α={al:g} recall", parts[1],
                float(np.mean([r["recall"] for r in g])))
            cmp(f"  {cs} SNR{snr:.0f} α={al:g} error", parts[2],
                float(np.mean([r["A"] for r in g])))


# ------------------------------------------------- 5. Table 37（分位数代价）
def complexity(tb) -> None:
    print("\n=== Table 37：已移交 verify_r6 ===")
    # 第六轮按新协议重测了分位数步（预热 + 单线程 + 多 seed 精度），并改写了 Table 37 的
    # 列结构：原 "selection" 与 "full sort" 两列实测**都在做全排序**（实现为 np.sort，
    # 归档 ratio 随 n 收敛到 1.00），且原计时无预热。故 Table 37 与其归档 r5_complexity.json
    # 由 r6_quantile_cost.json 取代，断言移至 verify_r6.py；本函数不再复核 r5 的计时，
    # 以免两轮争同一张表。
    print("  ok   Table 37 的断言现由 verify_r6.py 承担（本题只做移交）")


# ---------------------------------------------------- 6. Table 38（逐频带底）
M38_ORDER = ([("babble", b) for b in (1, 4, 16, 64)]
             + [("babble0", b) for b in (1, 4, 16, 64)]
             + [("gauss", b) for b in (1, 4, 16, 64)])


def banded_floor(tb) -> None:
    print("\n=== Table 38：逐频带噪声底 ===")
    B = load("r5_band_floor")["records"]
    rows = tb[38][1:]
    if len(rows) != 12:
        print(f"  FAIL 行数 {len(rows)}（应为 12）")
        FAIL.append("Table 38 行数")
    for i, (cs, nb) in enumerate(M38_ORDER):
        sub = [r for r in B if r["case"] == cs and r["bands"] == nb]
        cmp(f"  {cs:<8} bands={nb:<3} ν̂/ν", rows[i][2],
            float(np.mean([r["nu_ratio"] for r in sub])))
        cmp(f"  {cs:<8} bands={nb:<3} retention%", rows[i][3],
            100 * float(np.mean([r["keep"] for r in sub])))
        cmp(f"  {cs:<8} bands={nb:<3} error", rows[i][4],
            float(np.mean([r["A"] for r in sub])))


# ---------------------------------------------------- 7. Table 39（定阶）
M39_ORDER = ["collin", "med0.02", "med5", "nf"]
PEAK_RATIO = 0.2


def source_number(tb) -> None:
    print("\n=== Table 39：定阶（peak_ratio=0.2）===")
    D = load("r5_source_number")["records"]
    rows = tb[39][1:]
    for i, meth in enumerate(M39_ORDER):
        sub = [r for r in D if r["method"] == meth and r["peak_ratio"] == PEAK_RATIO]
        cmp(f"  {meth:<9} exact %", rows[i][1],
            100 * float(np.mean([r["n_est"] == r["n_true"] for r in sub])))
        cmp(f"  {meth:<9} mean|ΔN|", rows[i][2],
            float(np.mean([r["abs_err"] for r in sub])))
        cmp(f"  {meth:<9} under %", rows[i][3],
            100 * float(np.mean([r["n_est"] < r["n_true"] for r in sub])))
        cmp(f"  {meth:<9} over %", rows[i][4],
            100 * float(np.mean([r["n_est"] > r["n_true"] for r in sub])))
        cmp(f"  {meth:<9} error at N̂", rows[i][5],
            float(np.nanmean([r["A"] for r in sub])))
        cmp(f"  {meth:<9} points kept %", rows[i][6],
            100 * float(np.mean([r["keep"] for r in sub])))


# ------------------------------------------------------------ 8. Fig. 11
def fig11(tb) -> None:
    print("\n=== Fig. 11 与正文引用的两个数 ===")
    W = load("r5_unified_weight")["records"]
    ks = sorted({(r["case"], r["seed"]) for r in W})
    a = {(r["case"], r["seed"]): r["A"] for r in W if r["method"] == "max"}
    b = {(r["case"], r["seed"]): r["A"] for r in W if r["method"] == "nf+w4"}
    cases = sorted({k[0] for k in ks})
    win = sum(1 for c in cases
              if np.mean([b[k] for k in ks if k[0] == c])
              < np.mean([a[k] for k in ks if k[0] == c]))
    ok = win == 14
    print(f"  {'OK  ' if ok else 'FAIL'} 加权后在 15 个配置中领先 max 的数目   稿 14  归档 {win}")
    if not ok:
        FAIL.append("Fig. 11 配对计数")
    t15 = tb[15][1:]
    bnd = float(np.mean([min(num(t15[i][5]), num(t15[i][6]), num(t15[i][7]))
                         for i in range(len(T15_CASES))]))
    # 正文引用的上界从手稿抓取（不写死：写死只验证"记忆"）
    mm = re.search(r"against \$([\d.]+)°\$ for the best-observed reference", MS.read_text())
    ref = float(mm.group(1)) if mm else float("nan")
    ok = abs(bnd - ref) <= 0.006
    print(f"  {'OK  ' if ok else 'FAIL'} 未加权族的逐配置上界（正文引用 {ref}）   归档 {bnd:.4f}")
    if not ok:
        FAIL.append("Table 15 逐配置上界")


def main() -> None:
    global MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=str(MS),
                    help="要复核的手稿路径（负向测试时指向改动过的副本）")
    a = ap.parse_args()
    MS = pathlib.Path(a.manuscript)

    print("=" * 80)
    print("第五轮新增数字的复核：手稿字面值 vs 归档（含跨表一致性）")
    print(f"手稿：{MS}")
    print("=" * 80)
    tb = tables(MS)

    # —— R8 表号整体重排后的兼容垫片 ——
    # R8 把旧 23/24/39 提到正文 §10（新 19/20/21），其余附录表整体移号。
    # 本脚本按**旧号**索引，这里用 旧->新 映射把正确的表挂回旧号，全部既有断言无需逐条改动。
    _R8_TAB = {19: 23, 20: 24, 21: 25, 22: 26, 23: 19, 24: 20, 25: 27, 26: 28,
               27: 29, 28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36,
               35: 37, 36: 38, 37: 39, 38: 40, 39: 21, 40: 41, 41: 42}
    tb.update({o: tb[n] for o, n in _R8_TAB.items() if n in tb})
    need = [15, 17, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    missing = [n for n in need if n not in tb]
    if missing:
        print("手稿缺少表格：", missing)
        sys.exit(1)
    t17(tb)
    density(tb)
    complex_study(tb)
    alpha_cells(tb)
    complexity(tb)
    banded_floor(tb)
    source_number(tb)
    fig11(tb)
    print("\n" + "=" * 80)
    if FAIL:
        print(f"未通过 {len(FAIL)} 项：")
        for x in FAIL:
            print("   -", x)
        sys.exit(1)
    print("全部通过。")


if __name__ == "__main__":
    main()
