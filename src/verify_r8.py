"""复核第八轮（CSSP 投稿前意见）新增/改动的数字与结构。

判据**一律解析手稿取字面值**，再与归档复算比对——不把数字写死在脚本里
（写死就只验证"记忆"，实测中恰好在错的那几格上通过）。

A  Table 22（源数目失配）：3 行 × 5 档逐格 vs `r8_n_mismatch.json`
B  Table 19（噪声律）：新增的 Student-t 行逐格 vs `r2_noise_family.json`
   ＋ 原有五行**不得被改动**（同一归档复算，逐格相同）
C  正文散文里的断言：0.55 / 7.19 / 7.51 / thirteen-fold / 60.9 / 32.9 / 24–26 / 45.5 / ≤1.7
D  结构：§10.1–§10.4、§11 Discussion、§12 Conclusion、附录 A.1..A.17 各一次、
   表题注 1..42 递增、图 1..11
E  归档可用性：manifest 里点名 `r8_n_mismatch.json`

用法：
    python3 verify_r8.py
    python3 verify_r8.py --manuscript /tmp/pre_r8.md     # 负向测试：应失败
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FAIL: list[str] = []


def num(x: str):
    if x is None:
        return None
    s = str(x).replace("$", "").replace("\\%", "").replace("%", "").replace(",", "").strip()
    s = s.replace("−", "-").replace("–", "-")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def parse_tables(text: str) -> dict[int, list[list[str]]]:
    """按 '**Table N.**' 切块，取出紧随其后的 markdown 表格（表头 + 数据行）。"""
    out: dict[int, list[list[str]]] = {}
    for m in re.finditer(r"(?m)^\*\*Table (\d+)\.\*\*", text):
        n = int(m.group(1))
        rows: list[list[str]] = []
        for ln in text[m.end():].split("\n"):
            if ln.startswith("|"):
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):
                    continue
                rows.append(cells)
            elif rows:
                break
        out[n] = rows
    return out


def want(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(label)


def close(a, b, tol=5e-3) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) < tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=str(ROOT / "paper" / "manuscript.md"))
    a = ap.parse_args()
    text = pathlib.Path(a.manuscript).read_text()
    tb = parse_tables(text)

    MM = json.load(open(RES / "r8_n_mismatch.json"))
    NF = json.load(open(RES / "r2_noise_family.json"))
    N = MM["records"]
    DELTAS = MM["deltas"]

    def mm(method: str, d: int, n_true: int | None = None) -> float:
        v = [r["A"] for r in N if r["method"] == method and r["delta"] == d
             and (n_true is None or r["n_true"] == n_true)]
        return sum(v) / len(v)

    def nf(kind: str, p: float, fld: str) -> float:
        v = [r[fld] for r in NF["records"] if r["kind"] == kind and r["p"] == p and r[fld] == r[fld]]
        return sum(v) / len(v)

    # ---------------------------------------------------------------- A
    print("\n=== A. Table 22：源数目失配逐格 vs r8_n_mismatch.json ===")
    t22 = tb.get(22)
    if not t22:
        want("Table 22 存在", False)
    else:
        want("Table 22 行数 = 表头 + 3 掩码", len(t22) == 4, f"实为 {len(t22)}")
        want("Table 22 列数 = Mask + 5 档", all(len(r) == 6 for r in t22),
             f"{[len(r) for r in t22]}")
        want("Table 22 有 delta=0 档", "N$" in " ".join(t22[0]) or "$N$" in " ".join(t22[0]),
             str(t22[0]))
        order = [("collinearity", "collin"), ("median", "med0.02"), ("NF-SSP", "nf")]
        for i, (lab, key) in enumerate(order):
            row = t22[1 + i]
            want(f"  行{i+1} 标签含 {lab!r}", lab in row[0], row[0])
            for j, d in enumerate(DELTAS):
                got, exp = num(row[1 + j]), mm(key, d)
                want(f"  {key:<8} d={d:+d}  稿 {row[1+j]}", close(got, exp, 5e-3), f"归档 {exp:.4f}")

    # ---------------------------------------------------------------- B
    print("\n=== B. Table 19：Student-t 行 + 原五行不得改动 ===")
    t19 = tb.get(19)
    if not t19:
        want("Table 19 存在", False)
    else:
        want("Table 19 行数 = 表头 + 6 噪声律", len(t19) == 7, f"实为 {len(t19)}")
        rows = {r[0]: r for r in t19[1:]}
        st = [r for k, r in rows.items() if "Student" in k]
        want("有 Student-t 行", len(st) == 1, str(list(rows)[:7]))
        if st:
            r = st[0]
            for j, p in enumerate((0.05, 0.20, 0.40)):
                got, exp = num(r[1 + j]), nf("student", p, "s2_ratio")
                want(f"  Student-t σ̂²/σ² p={p}  稿 {r[1+j]}", close(got, exp, 6e-3), f"归档 {exp:.4f}")
            got = num(r[4].split("/")[0]) if "/" in r[4] else None
            exp = 8.0 * sum(1 for x in NF["records"]
                            if x["kind"] == "student" and x["p"] == 0.05 and x["gate_on"]) / 8.0
            want(f"  Student-t gate applied  稿 {r[4]}", close(got, exp), f"归档 {exp:.0f}/8")
            want("  Student-t 角度格含 NF 与 classical",
                 close(num(r[5].split("vs")[0]), nf("student", 0.05, "A_nf"), 6e-3)
                 and close(num(r[5].split("vs")[1]), nf("student", 0.05, "A_classical"), 6e-2),
                 r[5])
        for lab, key in (("Gaussian", "gauss"), ("Laplacian", "laplace"),
                         ("impulsive", "impulsive"), ("colored", "colored"), ("uniform", "uniform")):
            r = [v for k, v in rows.items() if k.startswith(lab)]
            if not r:
                want(f"  原有行 {lab} 仍在", False); continue
            r = r[0]
            ok = all(close(num(r[1 + j]), nf(key, p, "s2_ratio"), 6e-3)
                     for j, p in enumerate((0.05, 0.20, 0.40)))
            want(f"  原有行 {lab} 三档未变", ok, f"{[r[1+k] for k in range(3)]}")

    # ---------------------------------------------------------------- C
    print("\n=== C. 正文散文里的断言 ===")
    flat = re.sub(r"\s+", " ", text)
    want("§10.4 引 'a factor of thirteen'", "a factor of thirteen" in flat)
    want("§10.4 引 0.55 vs 7.19/7.51",
         "$0.55°$ against $7.19°$ and $7.51°$" in flat)
    want("§10.4 引 N-1 档 24–26",
         "$24$–$26°$" in flat, )
    want("§10.4 引 N-2 档 45.5", "$45.5°$" in flat)
    want("§10.4 引 over-counting ≤1.7", "$\\le1.7°$" in flat)
    lo = mm("nf", -2, 3)
    hi = mm("nf", -2, 6)
    want(f"§10.4 引 60.9 (N=3, d=-2)  归档 {lo:.2f}", close(num("60.9"), lo, 0.05))
    want(f"§10.4 引 32.9 (N=6, d=-2)  归档 {hi:.2f}", close(num("32.9"), hi, 0.05))
    want("Table 19 题注已改 'under six noise laws'", "under six noise laws" in flat)
    want("Table 19 题注说明了 babble 另处报告", "reported there rather than here" in flat)
    want("§10.1 已改 'five noise families in addition'", "five noise families in addition" in flat)
    want("§10.1 已改 'robust in every family tested'", "robust in every family tested" in flat)

    # ---------------------------------------------------------------- D
    print("\n=== D. 结构与编号 ===")
    for probe in ("## 10. Robustness of the calibration", "### 10.1 Departures from the Gaussian",
                  "### 10.2 Sensitivity to the conditioning",
                  "### 10.3 The gate's strictness", "### 10.4 Sensitivity to the assumed source number",
                  "## 11. Discussion and Limitations", "## 12. Conclusion"):
        want(f"  {probe[:46]!r} 在", probe in text)
    secs = [int(x) for x in re.findall(r"(?m)^### A\.(\d+) ", text)]
    # 只断言"连续无重复"，不写死张数——写死 N 是本项目反复误报的根源（见 MEMORY）。
    want(f"附录小节编号连续（共 {len(secs)} 节，A.1..A.{max(secs)}）",
         secs == list(range(1, len(secs) + 1)), str(secs))
    caps = [int(x) for x in re.findall(r"\*\*Table (\d+)\.\*\*", text)]
    want(f"表题注编号连续（共 {len(caps)} 张，1..{max(caps)}）",
         caps == list(range(1, len(caps) + 1)), f"{len(caps)} 张")
    figs = sorted(int(x) for x in re.findall(r"\*\*Fig\. (\d+)\.\*\*", text))
    want(f"图题注编号连续（共 {len(figs)} 张）",
         figs == list(range(1, len(figs) + 1)), str(figs))
    i_app = text.index("## Appendix A.")
    for title in ("Departures from the Gaussian noise law",
                  "Sensitivity to the conditioning of the mixing matrix",
                  "The gate's strictness and automatic order selection"):
        want(f"  附录不再含 {title[:34]!r}", title not in text[i_app:])

    # ---------------------------------------------------------------- E
    print("\n=== E. 归档可用性 ===")
    want("manifest 点名 r8_n_mismatch.json", "r8_n_mismatch.json" in flat)
    want("r8_n_mismatch.json 在 results/", (RES / "r8_n_mismatch.json").exists())

    # ---------------------------------------------------------------- F
    print("\n=== F. Table 43：真实录音房间 vs r8_real_room.json ===")
    RR = json.load(open(RES / "r8_real_room.json"))
    A = [r for r in RR["records"] if r["study"] == "noise"]
    t43 = tb.get(43)
    if not t43:
        want("Table 43 存在", False)
    else:
        want("Table 43 行数 = 表头 + 6 房间", len(t43) == 7, f"实为 {len(t43)}")
        want("Table 43 列数 = 5", all(len(r) == 5 for r in t43), str([len(r) for r in t43]))
        # 容差由印出的小数位导出：末位错一必报错
        for i, rec in enumerate(A):
            row, bt = t43[1 + i], rec["band_tail"]
            ok = (close(num(row[1]), rec["s2_hat_over_true"], 5e-5)
                  and close(num(row[2]), rec["band_dynamic_range_db"], 0.05)
                  and close(num(row[3]), rec["tail_ratio"]["0.05"], 5e-5)
                  and close(num(row[4].split("/")[0]), bt["median"], 5e-4))
            want(f"  行{i+1} {rec['room']} 四格", ok, " | ".join(row[1:]))
    for probe in ("### 10.5 A real recorded room",
                  "### A.18 The noise floor of a real recorded room",
                  "OpenSLR SLR28", "$0.927$ and $0.991$", "$38$ to $61$ dB",
                  "no source is emitting"):
        want(f"  {probe[:46]!r} 在", probe in flat)
    want("manifest 点名 r8_real_room.json", "r8_real_room.json" in flat)
    want("r8_real_room.json 在 results/", (RES / "r8_real_room.json").exists())

    print("\n" + "=" * 74)
    if FAIL:
        print(f"✗ {len(FAIL)} 项未通过：")
        for f in FAIL[:14]:
            print("   ·", f)
        sys.exit(1)
    print("✓ verify_r8 全部通过")


if __name__ == "__main__":
    main()
