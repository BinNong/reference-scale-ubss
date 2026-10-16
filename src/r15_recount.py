#!/usr/bin/env python3
"""R15 全量复算：把 paper/manuscript.md 里每一张表逐格对回 results/ 下的记录。

动机：现有校验链只查「引用是否悬空」「编号是否连续」「mean 行是否等于列均值」，
不查「这张表里的数是否等于记录算出来的数」。R14 系列把表号顺延过三次，
于是一个语义正确的旧引用会指向内容不同的新表——本脚本负责数字层，`check_refs.py`
负责引用层，两者合起来才覆盖审稿人会查的那一类问题。

设计约定：每个抽取器返回 {(稿例行标签, 稿件列标签): 期望值}，键直接取稿件里的字面
文本，两侧用同一个 norm() 归一化后**精确相等**才配对。刻意不用模糊匹配——
上一个版本用 `a in b` 做行匹配，把 "$N=5$" 配到了 "win 512"（"n5" 是 "win512" 的
子串），160 格误报。宁可少配，不可错配。

用法（从仓库根运行）：

    python3 src/r15_recount.py                          # 人手可读摘要
    python3 src/r15_recount.py --json results/r15_recount.json
    python3 src/r15_recount.py --table 33               # 只看一张表
    python3 src/r15_recount.py --verbose                # 列出未配上的格

判据：mismatch 为空，或每一项都是显示层舍入并在 note 里写明理由。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
MANUSCRIPT = os.path.join(ROOT, "paper", "manuscript.md")

# ---------------------------------------------------------------- 归一化

_SUBS = [(r"\\pi_\{?0\}?", "pi0"), (r"\\pi_\{?1\}?", "pi1"), (r"\\pi_\{?2p?\}?", "pi2"),
         (r"\\pi_0", "pi0"), (r"\\sigma\^?2", "s2"), (r"\\hat\{\\sigma\}\^2", "s2hat"),
         (r"\\hat\sigma\^2", "s2hat"), (r"\\hat\nu", "nuhat"), (r"\\nu", "nu"),
         (r"\\rho", "rho"), (r"\\eta", "eta"), (r"\\alpha", "alpha"),
         (r"\\kappa", "kappa"), (r"\\tau", "tau"), (r"\\theta", "theta"),
         (r"\\Delta", "delta"), (r"\\lambda", "lambda"), (r"\\cos", "cos"),
         (r"\\le", "<="), (r"\\ge", ">="), (r"\\times", "x"), (r"\\cdot", ""),
         (r"\\,", ""), (r"\\%", ""), (r"\\!", ""), (r"\\text\{([^}]*)\}", r"\1"),
         (r"\\mathbf\{([^}]*)\}", r"\1"), (r"\\mathrm\{([^}]*)\}", r"\1"),
         (r"\\hat\{([^}]*)\}", r"\1"), (r"\\_", ""), (r"\\", "")]


def norm(s) -> str:
    """稿件字面文本 -> 比较键。两侧都过它，所以保持同一套约定即可。"""
    if s is None:
        return ""
    t = str(s)
    for pat, rep in _SUBS:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = t.replace("$", "").replace("{", "").replace("}", "")
    # 保留 < > = + - ：它们是有意义的区别（"$\hat N<N$" vs "$\hat N>N$"，"$N-1$" vs "$N+1$"）。
    # 早先把它们一并剥掉，导致这两对键撞车，Table 43/44 各误报 7 格。
    t = re.sub(r"[^0-9a-z.<>+=\-]+", "", t.lower())
    # 折叠 ".0" -> "."：稿件写 "p=.02"、抽取器写 "p=0.02"，两者须同键。
    return t.replace("0.", ".")


def num(s):
    if s is None:
        return None
    t = str(s)
    for pat, rep in _SUBS:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = t.replace("$", "").replace("\\,", "").replace(",", "")
    t = re.sub(r"^\*\*(.*)\*\*$", r"\1", t.strip())
    t = re.sub(r"[^0-9eE.+\-]", "", t)
    if t in ("", "-", ".", "+", "-."):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def nums(s):
    """把 '0.99 / 0.82' 或 '4.14% / 0.057 / 11.77' 解析成多个数。"""
    if s is None:
        return []
    t = str(s)
    for pat, rep in _SUBS:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = t.replace("$", "")
    parts = re.split(r"/|;", t)
    out = []
    for p in parts:
        p = p.replace(",", "").strip()
        p = re.sub(r"[^0-9eE.+\-]", "", p)
        if p in ("", "-", ".", "+"):
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------- markdown 解析

CAP_RE = re.compile(r"^\*\*Table\s+(\d+)\.\*\*\s*(.*)$")
SEP_RE = re.compile(r"^[\s|:\-]+$")


def parse_tables(md: str) -> dict[int, dict]:
    """{表号: {'caption', 'blocks': [{'header':[...], 'rows':[[...]]}]}}。

    一个题注下可以有多个 markdown 表（Table 6 的角度误差块 + SDR 块），
    以空行分隔；每个块有自己的表头行。
    """
    lines = md.split("\n")
    tables: dict[int, dict] = {}
    i = 0
    while i < len(lines):
        m = CAP_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        tnum, caption = int(m.group(1)), m.group(2)
        blocks: list[dict] = []
        cur: list[list[str]] = []
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                if cur:
                    blocks.append(cur)
                    cur = []
                j += 1
                continue
            if not s.startswith("|"):
                break
            cur.append([c.strip() for c in s.strip("|").split("|")])
            j += 1
        if cur:
            blocks.append(cur)
        out = []
        for b in blocks:
            body = [r for r in b if not all(SEP_RE.match(c or "---") for c in r)]
            if not body:
                continue
            out.append({"header": body[0], "rows": body[1:]})
        tables[tnum] = {"caption": caption, "blocks": out}
        i = j
    return tables


def row_labels(tbl: dict, label_cols: int) -> list[list[str]]:
    """每块每行的标签列（按列做 carry-forward，处理合并单元格）。"""
    res = []
    for blk in tbl["blocks"]:
        carried = [""] * label_cols
        for row in blk["rows"]:
            key = []
            for ci in range(label_cols):
                c = row[ci] if ci < len(row) else ""
                if c.strip() == "" and carried[ci]:
                    key.append(carried[ci])
                else:
                    key.append(c)
                    carried[ci] = c
            res.append(key)
    return res


def flat_cells(tbl: dict, label_cols: int):
    """[(rowkey_tuple, col_label, cell_text)]，跨块展平。"""
    out = []
    carried = {}
    for bi, blk in enumerate(tbl["blocks"]):
        carried[bi] = [""] * label_cols
        for row in blk["rows"]:
            key = []
            for ci in range(label_cols):
                c = row[ci] if ci < len(row) else ""
                if c.strip() == "" and carried[bi][ci]:
                    key.append(carried[bi][ci])
                else:
                    key.append(c)
                    carried[bi][ci] = c
            multi = len(tbl["blocks"]) > 1
            seen: dict[str, int] = {}
            for ci in range(label_cols, min(len(row), len(blk["header"]))):
                chead = blk["header"][ci]
                seen[chead] = seen.get(chead, 0) + 1
                if seen[chead] > 1:
                    chead = f"{chead} #{seen[chead]}"
                if multi and ci >= 1:
                    chead = f"{blk['header'][0]}|{chead}"
                out.append((tuple(key), chead, row[ci]))
    return out


# ---------------------------------------------------------------- 记录读取

_cache: dict[str, object] = {}


def rec(name: str):
    if name not in _cache:
        with open(os.path.join(RESULTS, name + ".json")) as fh:
            _cache[name] = json.load(fh)
    return _cache[name]


def R(name: str) -> list[dict]:
    d = rec(name)
    return d["records"] if isinstance(d, dict) and "records" in d else d


def mean(records, key, pred=None):
    vals = [r[key] for r in records if (pred is None or pred(r)) and r.get(key) is not None]
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return st.mean(vals) if vals else float("nan")


def pmean(records, key, pred=None):
    v = mean(records, key, pred)
    return v


def pcol(p: float) -> str:
    """稿件里 p 列的字面写法：0.02 -> "p=.02"。"""
    return f"p={p:.2f}".replace("0.", ".")


LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
P_OF = dict(zip(LADDER, [0.02, 0.05, 0.10, 0.20, 0.40, 1.0]))
COLS6 = ["p=.02", "p=.05", "p=.10", "p=.20", "p=.40", "dense"]
STRUCT = {"linear chirps": "td_chirp", "harmonic stacks": "td_sinusoid",
          "AM–FM tones": "td_amfm", "impulse trains": "td_impulse"}
REAL_ORDER = ["R1_res:w256", "R1_res:w512", "R1_res:w1024", "R1_res:w2048",
              "R2_n:N3", "R2_n:N5", "R2_n:N6", "R3_snr:snr00", "R3_snr:snr10",
              "R3_snr:snr30", "R3_snr:snr40", "R4_dense:d1", "R4_dense:d2",
              "R5_noise:babble", "R5_noise:clean"]
REAL_LABEL = {"R1_res:w256": "win 256", "R1_res:w512": "win 512", "R1_res:w1024": "win 1024",
              "R1_res:w2048": "win 2048", "R2_n:N3": "$N=3$", "R2_n:N5": "$N=5$",
              "R2_n:N6": "$N=6$", "R3_snr:snr00": "SNR $0$ dB", "R3_snr:snr10": "SNR $10$ dB",
              "R3_snr:snr30": "SNR $30$ dB", "R3_snr:snr40": "SNR $40$ dB",
              "R4_dense:d1": "1 dense source", "R4_dense:d2": "2 dense sources",
              "R5_noise:babble": "babble noise", "R5_noise:clean": "noise-free"}


def _rl(case):
    return (REAL_LABEL[case],)


# ================================================================ 抽取器
# 键 = (稿件的行标签元组, 稿件的列标签)。值 = 记录算出的数。

def t1():
    d = rec("prop6_convention")
    out = {}
    for r in d["rows"]:
        p = r.get("p")
        lab = ("dense",) if (p is None or p >= 1.0) else (f"{p:.2f}",)
        out[(lab, "$\\pi_0$")] = r["pi0"]
        out[(lab, "$r$ measured")] = r["r_emp"]
        cl = r.get("closed_fixed")
        if cl is not None and not math.isnan(cl):
            out[(lab, "$r$ semi-analytic")] = cl
            out[(lab, "rel. error")] = (cl - r["r_emp"]) / r["r_emp"] * 100
        out[(lab, "noise mass $\\ge1/2$")] = float(r["pi0"] >= 0.5)
    return out


def t2():
    d = rec("r4_table2")
    out = {}
    for r in d["rows"]:
        p = P_OF.get(r["case"])
        lab = ("dense",) if p is None else (f"{p:.2f}",)
        out[(lab, "$\\sigma^2$ true")] = r["s2_true"]
        out[(lab, "$\\hat\\sigma^2$")] = r["s2_hat"]
        out[(lab, "ratio")] = r["ratio"]
        out[(lab, "spread")] = r["spread"]
    dd = d.get("dense_5") or {}
    if dd:
        for col, f in (("$\\sigma^2$ true", "s2_true"), ("$\\hat\\sigma^2$", "s2_hat"),
                       ("ratio", "ratio"), ("spread", "spread")):
            out[(("dense",), col)] = dd.get(f, float("nan"))
    return out


def t3():
    """Table 3：τ 逐位复算；经验通过率核对"落在 ±2 个二项标准误内"这一判据。"""
    d = rec("r3_tau_mc")
    out = {}
    for r in d["rows"]:
        ex = int(round(math.log10(r["alpha"])))          # 1e-2 -> -2
        lab = (str(r["m_obs"]), f"$10^{{{ex}}}$")
        out[(lab, "$\\tau$")] = r["tau"]
        out[(lab, "empirical admission rate")] = r["empirical"]
        out[(lab, "counting-error tol")] = 2 * r["binomial_se"]
    return out


def t6():
    rs = [r for r in R("nfr_results") if r["experiment"] == "E1_sparsity"]
    meth = {"**NF-SSP (proposed)**": "NF-SSP", "SCA-default": "SCA-default",
            "SCA-te5 (fixed)": "SCA-fixed-te5", "SCA-tuned (oracle)": "SCA-L1-tuned",
            "Oracle-A": "oracle_a_l1"}
    out = {}
    for lab, m in meth.items():
        for k, col in zip(LADDER, COLS6):
            pred = lambda r, kk=k, mm=m: r["cfg_key"] == kk and r["method"] == mm
            out[(lab, "Angle error (°)|" + col)] = mean(rs, "A_angle_deg", pred)
            out[(lab, "SDR (dB)|" + col)] = mean(rs, "SDR", pred)
    return out


def t7():
    rs = [r for r in R("nfr_results") if r["experiment"] == "E1_sparsity"]
    meth = {"NF-SSP": "NF-SSP", "SCA-default": "SCA-default", "SCA-tuned": "SCA-L1-tuned",
            "potential function": "pf_auto_l1", "smoothed $\\ell_0$": "sl0_l1"}
    return {((), lab): mean(rs, "time_s", lambda r, m=m: r["method"] == m)
            for lab, m in meth.items()}


def _lab8(r):
    return {0.05: ("$p=0.05$",), 0.20: ("$p=0.20$",)}.get(r["p"], ("dense",))


def t8():
    rs = [r for r in R("nfr_results") if r["experiment"] == "E2_snr"]
    out = {}
    for reg in (0.05, 0.20):
        key = "tf_p05" if reg == 0.05 else "tf_p20"
        for snr in (0, 10, 20, 30, 40):
            lab = (f"${pcol(reg)}$", str(snr))
            for m, col in (("NF-SSP", "**NF-SSP**"), ("SCA-default", "SCA-default"),
                           ("SCA-L1-tuned", "SCA-tuned (oracle)")):
                out[(lab, col)] = mean(rs, "A_angle_deg",
                                         lambda r, k=key, s=snr, mm=m:
                                         r["cfg_key"] == k and r["snr_db"] == s and r["method"] == mm)
    dense = [r for r in rs if r["cfg_key"] == "tf_gauss"]
    bysnr = defaultdict(list)
    for r in dense:
        bysnr[r["method"]].append(r["A_angle_deg"])
    out[("dense", "**NF-SSP**")] = min(bysnr["NF-SSP"])
    out[("dense", "SCA-default")] = max(bysnr["SCA-default"]) - min(bysnr["SCA-default"])
    out[("dense", "SCA-tuned (oracle)")] = max(bysnr["SCA-L1-tuned"]) - min(bysnr["SCA-L1-tuned"])
    return out


def _lab9(r):
    return (str(r["m_obs"]), pcol(P_OF[r["cfg_key"]]))


def t9():
    rs = [r for r in R("nfr_results") if r["experiment"] == "E3_M"]
    out = {}
    for r in rs:
        lab = _lab9(r)
        for m, col in (("NF-SSP", "**NF-SSP**"), ("SCA-default", "SCA-default"),
                       ("SCA-fixed-te5", "SCA-te5")):
            out[(lab, col)] = mean(rs, "A_angle_deg",
                                     lambda x, l=lab, mm=m: _lab9(x) == l and x["method"] == mm)
    return out


def _lab10(r):
    return (str(r["n_true"]), pcol(P_OF[r["cfg_key"]]))


def t10():
    rs = [r for r in R("nfr_results") if r["experiment"] == "E4_N"]
    out = {}
    for r in rs:
        lab = _lab10(r)
        for m, col in (("NF-SSP", "**NF-SSP**"), ("SCA-default", "SCA-default"),
                       ("SCA-fixed-te5", "SCA-te5")):
            out[(lab, col)] = mean(rs, "A_angle_deg",
                                     lambda x, l=lab, mm=m: _lab10(x) == l and x["method"] == mm)
    return out


def t11():
    rs = [r for r in R("nfr_results") if r["experiment"] == "E6_plugin"]

    def split(m):
        for base in ("ssp_kmeans_l1", "ssp_fcm_sp"):
            if m.startswith(base):
                return base, m[len(base):].lstrip("-")
        return m, ""

    out = {}
    for base, tag in (("ssp_kmeans_l1", "spherical $k$-means"), ("ssp_fcm_sp", "fuzzy $c$-means")):
        for suf, lab in (("classical", f"{tag}, conventional"), ("NF", f"{tag}, **NF gate**")):
            for k, col in zip(LADDER, COLS6):
                out[(lab, col)] = mean(rs, "A_angle_deg",
                                       lambda r, b=base, s=suf, kk=k:
                                       r["cfg_key"] == kk and split(r["method"]) == (b, s))
        for k, col in zip(LADDER, COLS6):
            a = mean(rs, "A_angle_deg", lambda r, b=base, kk=k:
                     r["cfg_key"] == kk and split(r["method"]) == (b, "classical"))
            b = mean(rs, "A_angle_deg", lambda r, b=base, kk=k:
                     r["cfg_key"] == kk and split(r["method"]) == (b, "NF"))
            out[(tag + ", relative improvement", col)] = (a - b) / a * 100
    return out


def t12():
    rs = [r for r in R("nfr_results") if r["experiment"] == "E8_structured"]
    out = {}
    for lab, key in STRUCT.items():
        for m, col in (("NF-SSP", "**NF-SSP**"), ("SCA-default", "SCA-default"),
                       ("SCA-fixed-te5", "SCA-te5"), ("SCA-L1-tuned", "SCA-tuned")):
            out[(lab, col)] = mean(rs, "A_angle_deg",
                                   lambda r, k=key, mm=m: r["cfg_key"] == k and r["method"] == mm)
    return out


def t13():
    rr = R("real_results")
    d = rec("real_grid")
    out = {}
    for case in REAL_ORDER:
        lab = _rl(case)
        nf = [r for r in rr if r["case"] == case and r["method"] == "NF-SSP"]
        out[(lab, "$\\pi_0$")] = mean(nf, "pi0")
        out[(lab, "$\\pi_1$")] = mean(nf, "pi1")
        out[(lab, "WDO-valid")] = mean(nf, "single_dom_frac")
        out[(lab, "$r=\\mathrm{median}(e)/\\nu$")] = mean(nf, "r_true")
        out[(lab, "optimal $t_e$")] = d[f"{case}|star"]["te"]
        out[(lab, "optimal $c$")] = d[f"{case}|star"]["c"]
    return out


def _real(meth_map):
    rr = R("real_results")
    out = {}
    for lab, m in meth_map.items():
        for case in REAL_ORDER:
            out[(_rl(case), lab)] = mean(rr, "A_angle_deg",
                                           lambda r, mm=m, c=case: r["case"] == c and r["method"] == mm)
    return out


def t14():
    out = _real({"NF-SSP": "NF-SSP", "median $t_e{=}0.02$": "SCA-median",
                 "max $c{=}0.05$": "SCA-max", "top-5%": "top-K",
                 "median-opt": "SCA-median-opt", "max-opt": "SCA-max-opt",
                 "top-K-opt": "top-K-opt"})
    for lab, m in (("NF-SSP", "NF-SSP"), ("median $t_e{=}0.02$", "SCA-median"),
                   ("max $c{=}0.05$", "SCA-max"), ("top-5%", "top-K"),
                   ("median-opt", "SCA-median-opt"), ("max-opt", "SCA-max-opt"),
                   ("top-K-opt", "top-K-opt")):
        out[("**mean**", lab)] = st.mean(
            [v for (rk, ck), v in out.items() if ck == lab and rk != ("**mean**",)])
    return out


def t14_sdr():
    rr = R("real_results")
    out = {}
    for case in REAL_ORDER:
        out[(_rl(case), "Oracle-A (SDR)")] = mean(
            rr, "SDR", lambda r, c=case: r["case"] == c and r["method"] == "oracle_a_l1")
    out[("**mean**", "Oracle-A (SDR)")] = st.mean(
        [v for (rk, ck), v in out.items() if ck == "Oracle-A (SDR)"])
    return out


def _oct15(b: int) -> str:
    """Table 15 的八度列名：首列 "1 (lowest)"，末列 "8 (highest)"。"""
    return "1 (lowest)" if b == 1 else ("8 (highest)" if b == 8 else str(b))


def t15():
    d = rec("real_mechanism")
    out = {}
    for r in d["real|real w1024 S20"]:
        b = int(r["bin"]) + 1
        out[("**real speech** direction error ($°$)", f"{b} (lowest)" if b == 1 else
             (f"{b} (highest)" if b == 8 else str(b)))] = r["ang_med"]
        out[("**real speech** single-source", f"{b} (lowest)" if b == 1 else
             (f"{b} (highest)" if b == 8 else str(b)))] = r["single_frac"]
    for r in d["synth|synth tf_p02"]:
        b = int(r["bin"]) + 1
        out[("synthetic $p=0.05$ direction error", _oct15(b))] = r["ang_med"]
        out[("synthetic $p=0.05$ single-source", _oct15(b))] = r["single_frac"]
    for r in d["synth|synth tf_p40"]:
        b = int(r["bin"]) + 1
        out[("synthetic $p=0.40$ direction error", _oct15(b))] = r["ang_med"]
        out[("synthetic $p=0.40$ single-source", _oct15(b))] = r["single_frac"]
    return out


def t16():
    uw = R("r5_unified_weight")
    meth = {"NF gate (unweighted)": "nf", "NF + energy": "nf+w",
            "**NF + energy$\\cdot\\vert\\cos\\vert^4$**": "nf+w4", "base + energy": "base+w",
            "median $t_e{=}0.02$": "med0.02", "max $c{=}0.05$": "max", "max + same weight": "max+w4"}
    return _real_w(uw, meth, "A")


def _real_w(uw, meth_map, field):
    out = {}
    for lab, m in meth_map.items():
        for case in REAL_ORDER:
            out[(_rl(case), lab)] = mean(uw, field, lambda r, mm=m, c=case:
                                           r["case"] == c and r["method"] == mm)
        out[("**mean**", lab)] = st.mean(
            [v for (rk, ck), v in out.items() if ck == lab and rk != ("**mean**",)])
    return out


def t17():
    wv = R("weighted_variant")
    meth = {"NF gate": "nf", "NF + energy": "nf+w", "NF + energy$\\cdot\\rho^{8}$": "nf+p8",
            "base + energy": "base+w", "hard-threshold oracle": "oracle"}
    out = {}
    for lab, m in meth.items():
        for k in LADDER:
            lab_r = "dense" if k == "tf_gauss" else f"$p={P_OF[k]:.2f}$"
            out[(lab_r, lab)] = mean(wv, "A", lambda r, mm=m, kk=k: r["case"] == kk and r["method"] == mm)
        out[("**mean**", lab)] = st.mean(
            [v for (rk, ck), v in out.items() if ck == lab and rk != ("**mean**",)])
    return out


def t18():
    rs = R("r2_ssp_quality")
    keys = {"none (collinearity only)": "single-source criterion only",
            "classical $t_e=0.02$": "classical $t_e{=}0.02$",
            "median $t_e=5$": "median $t_e{=}5$",
            "max-referenced $c=0.05$": "max-referenced $c{=}0.05$",
            "**NF $\\alpha=10^{-4}$**": "NF gate $\\alpha{=}10^{-4}$"}
    out = {}
    for L, k in keys.items():
        pred = lambda r, kk=k: r["method"] == kk
        out[(L, "admitted")] = mean(rs, "keep_frac", pred) * 100
        out[(L, "noise-only rejected")] = mean(rs, "noise_reject", pred) * 100
        out[(L, "multi-source rejected")] = mean(rs, "multi_reject", pred) * 100
        out[(L, "$F_1$")] = mean(rs, "f1", pred)
        for k2 in LADDER[:-1]:
            col = f"$p{{=}}{P_OF[k2]:.2f}$ prec. / rec."
            p2 = lambda r, kk=k, c=k2: r["method"] == kk and r["case"] == c
            out[(L, col)] = mean(rs, "precision", p2)
            out[(L, f"$p{{=}}{P_OF[k2]:.2f}$")] = mean(rs, "recall", p2)
    return out


def t19():
    rs = R("r2_ssp_quality_real")
    keys = {"none (collinearity only)": "single-source criterion only",
            "classical $t_e=0.02$": "classical $t_e{=}0.02$",
            "median $t_e=5$": "median $t_e{=}5$",
            "max-referenced $c=0.05$": "max-referenced $c{=}0.05$",
            "**NF $\\alpha=10^{-4}$**": "NF gate $\\alpha{=}10^{-4}$"}
    out = {}
    for L, k in keys.items():
        pred = lambda r, kk=k: r["method"] == kk
        out[(L, "admitted")] = mean(rs, "keep_frac", pred) * 100
        out[(L, "precision")] = mean(rs, "precision", pred)
        out[(L, "recall")] = mean(rs, "recall", pred)
        out[(L, "$F_1$")] = mean(rs, "f1", pred)
        out[(L, "noise-only rejected")] = mean(rs, "noise_reject", pred) * 100
        out[(L, "direction error of admitted SSPs")] = mean(rs, "dir_err_med_deg", pred)
    return out


def t21():
    d = rec("r2_qmax_sweep")
    out = {}
    for r in d["summary"]:
        lab = ("dense",) if r["case"] == "tf_gauss" else (f"$p={P_OF[r['case']]:.2f}$",)
        out[(lab, "$\\pi_0$")] = r["pi0"]
        for q in d["qmaxes"]:
            out[(lab, f"{q}")] = r["ratios"][str(q)]
        for c in ("err_rel_010_005", "err_lo_010_005", "err_hi_010_005"):
            if c in r:
                out[(lab, "error over $[0.01,0.05]$")] = r[c]
        out[(lab, "$q_{20}$")] = r.get("q_bias20", float("nan"))
        out[(lab, "$q_{\\rm adapt}$")] = r.get("q_adapt", float("nan"))
    return out


def t23():
    d = rec("r2_stats")["nfr_ladder"]["rows"]
    out = {}
    for r in d:
        head = r["label"].split(":")[0].strip()          # "ladder tf_p02" / "ladder pooled"
        tail = head.split()[-1]
        lab = ("ladder, SCA-default $-$ NF, pooled" if tail == "pooled"
               else "ladder, SCA-default $-$ NF, $" + pcol(P_OF[tail]) + "$")
        out[(lab, "$n$")] = r["n"]
        out[(lab, "$\\Delta$")] = r["mean"]
        out[(lab, "$p$")] = r["p"]
        out[(lab, "$d_z$")] = r["dz"]
        out[(lab, "Holm / BH#0")] = r["p_holm"]
        out[(lab, "Holm / BH#1")] = r["p_bh"]
    return out


def t23b():
    d = rec("r2_stats")["real_speech"]["rows"]
    out = {}
    RL = {"SCA-median": "real, SCA-median $-$ NF", "SCA-max": "real, SCA-max $-$ NF",
          "top-K": "real, top-$5\\%$ $-$ NF", "NF+e": "real, NF $+$ energy $-$ NF",
          "NF+e·cos4": "real, NF $+$ energy$\\cdot\\vert\\cos\\vert^4$ $-$ NF",
          "NF+e·ρ2": "real, NF $+$ energy$\\cdot\\rho^2$ $-$ NF",
          "NF+e·ρ4": "real, NF $+$ energy$\\cdot\\rho^4$ $-$ NF",
          "NF+e·ρ8": "real, NF $+$ energy$\\cdot\\rho^8$ $-$ NF",
          "base+e": "real, base $+$ energy $-$ NF",
          "base+e·ρ4": "real, base $+$ energy$\\cdot\\rho^4$ $-$ NF",
          "Oracle-A": "real, Oracle-A $-$ NF"}
    for r in d:
        tail = r["label"].split(": ", 1)[1].replace(" − NF-SSP", "")
        lab = RL.get(tail)
        if lab is None:
            continue
        out[(lab, "$n$")] = r["n"]
        out[(lab, "$\\Delta$")] = r["mean"]
        out[(lab, "$p$")] = r["p"]
        out[(lab, "$d_z$")] = r["dz"]
        out[(lab, "Holm / BH#0")] = r["p_holm"]
        out[(lab, "Holm / BH#1")] = r["p_bh"]
    return out


def t24():
    d = rec("r4_guard_errors")
    rs = d["records"]
    STRUCT_CASES = {"td_chirp", "td_sinusoid", "td_amfm", "td_impulse"}
    fam = {"sparse ladder, $p=0.02$–$0.40$":
               lambda r: r["case"] in ("tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40"),
           "dense ($p=1$)": lambda r: r["case"] == "tf_gauss",
           "structured sources (chirp, sinusoid, AM–FM, transient)":
               lambda r: r["case"] in STRUCT_CASES,
           "Laplace and block-sparse": lambda r: r["case"] in ("tf_lap", "tf_b06")}
    out = {}
    for lab, pred in fam.items():
        sub = [r for r in rs if pred(r)]
        ne = sum(1 for r in sub if r["error"])
        out[(lab, "decisions")] = len(sub)
        out[(lab, "material errors")] = ne
        out[(lab, "rate")] = ne / len(sub) * 100
    ne = sum(1 for r in rs if r["error"])
    out[("**all**", "decisions")] = len(rs)
    out[("**all**", "material errors")] = ne
    out[("**all**", "rate")] = ne / len(rs) * 100
    return out


def t25():
    d = rec("r4_density_frontend")["by_eps"]
    out = {}
    for eps, blk in d.items():
        lab = (f"${float(eps):.2f}$",)
        per = blk["per"]
        out[(lab, "DBSCAN retained")] = st.mean(
            [v["db_kept_frac"] for v in per.values()]) * 100
        a = st.mean([v["A_classical"] for v in per.values()])
        b = st.mean([v["A_nf"] for v in per.values()])
        out[(lab, "classical gate")] = a
        out[(lab, "NF gate")] = b
        out[(lab, "change")] = (a - b) / a * 100
        out[(lab, "regimes where NF is better#0")] = float(
            sum(1 for v in per.values() if v["A_nf"] < v["A_classical"]))
        out[(lab, "regimes where NF is better#1")] = float(len(per))
    return out


def t26():
    rs = R("r4_robust_floor")
    labmap = {"Gaussian, $p=0.05$": ("gauss", "tf_p05"), "Gaussian, $p=0.40$": ("gauss", "tf_p40"),
              "Laplace, $p=0.40$": ("laplace", "tf_p40"), "impulsive, $p=0.40$": ("impulsive", "tf_p40"),
              "colored, $p=0.40$": ("colored", "tf_p40"), "uniform, $p=0.40$": ("uniform", "tf_p40")}
    bab = [r for r in rs if r.get("dom") == "real" and r.get("kind") == "babble"]
    out = {}
    for L, (k, c) in labmap.items():
        pred = lambda r, kk=k, cc=c: r["kind"] == kk and r.get("p") == P_OF[cc]
        out[(L, "stat.")] = mean(rs, "stat_noise", pred)
        out[(L, "$\\hat\\sigma^2/\\sigma^2$ lower tail")] = mean(rs, "ratio_paper", pred)
        out[(L, "MAD")] = mean(rs, "ratio_mad", pred)
        out[(L, "$q_{25}$")] = mean(rs, "ratio_q25", pred)
        out[(L, "error: lower tail")] = mean(rs, "A_paper", pred)
        out[(L, "MAD #2")] = mean(rs, "A_mad", pred)
        out[(L, "$q_{25}$ #2")] = mean(rs, "A_q25", pred)
    # 真实语音三行来自同一记录的 dom=='real' 行（kind: gauss / babble / None=noise-free）
    rl = [r for r in rs if r.get("dom") == "real"]
    for L, k in (("real speech, Gaussian", "gauss"), ("real speech, babble", "babble"),
                 ("real speech, noise-free", None)):
        sub = [r for r in rl if str(r.get("kind")) == str(k)]
        out[(L, "stat.")] = mean(sub, "stat_noise")
        out[(L, "$\\hat\\sigma^2/\\sigma^2$ lower tail")] = mean(sub, "ratio_paper")
        out[(L, "MAD")] = mean(sub, "ratio_mad")
        out[(L, "$q_{25}$")] = mean(sub, "ratio_q25")
        out[(L, "error: lower tail")] = mean(sub, "A_paper")
        out[(L, "MAD #2")] = mean(sub, "A_mad")
        out[(L, "$q_{25}$ #2")] = mean(sub, "A_q25")
    return out


def t28():
    rs = R("r5_density_baseline")
    meth = {"conventional median $t_e{=}0.02$": "SCA-median", "**NF-SSP, one setting**": "NF-SSP",
            "best of the family (oracle radius)": "best", "DBSCAN, fixed $\\texttt{eps}=0.02$": "DBSCAN",
            "OPTICS, fixed $\\texttt{eps}=0.02$": "OPTICS", "density peaks (no radius)": "density-peak"}
    out = {}
    for lab, m in meth.items():
        for k, col in zip(LADDER, COLS6 + ["mean"][:0]):
            out[(lab, col)] = mean(rs, "A", lambda r, mm=m, kk=k: r["case"] == kk and r["method"] == mm)
        out[(lab, "mean")] = mean(rs, "A", lambda r, mm=m: r["method"] == mm)
    return out


def t29():
    rs = R("r5_density_real")
    meth = {"conventional median $t_e{=}0.02$": "SCA-median", "**NF-SSP, one setting**": "NF-SSP",
            "best of the family (oracle radius)": "best", "DBSCAN, fixed $\\texttt{eps}=0.02$": "DBSCAN",
            "density peaks (no radius)": "density-peak"}
    out = {}
    for lab, m in meth.items():
        for case, col in (("R1_res:w1024", "win 1024"), ("R2_n:N6", "$N=6$"),
                          ("R4_dense:d2", "2 dense"), ("R5_noise:babble", "babble")):
            out[(lab, col)] = mean(rs, "A", lambda r, mm=m, c=case: r["case"] == c and r["method"] == mm)
        out[(lab, "mean")] = mean(rs, "A", lambda r, mm=m: r["method"] == mm)
    return out


def t31():
    rs = R("r5_complex")
    out = {}
    for kind, lab in (("real", "real (this paper)"), ("cfixed", "complex, fixed"),
                      ("cperfreq", "complex, per frequency")):
        pred = lambda r, k=kind: r["kind"] == k
        out[(lab, "admission")] = mean(rs, "collin_admit_frac", pred)
        out[(lab, "single-source recall")] = mean(rs, "collin_recall_single", pred)
        out[(lab, "precision")] = mean(rs, "collin_precision", pred)
        out[(lab, "noise admitted")] = mean(rs, "collin_fpr_noise", pred)
    return out


def t32():
    rs = R("r5_complex")
    meth = {"no criteria (all points)": ("keep_all", "A_cx_all"),
            "collinearity only": ("keep_collin", "A_cx_collin"),
            "conventional median $t_e{=}0.02$": ("keep_median0.02", "A_cx_median0.02"),
            "NF-SSP transplanted verbatim": ("keep_nf", "A_cx_nf"),
            "calibrated energy criterion only": ("keep_nfE", "A_cx_nfE"),
            "the same, with the retention self-check": ("keep_nfE+sc", "A_cx_nfE+sc")}
    out = {}
    for lab, (kk, pre) in meth.items():
        for k2, col in zip(LADDER, COLS6):
            out[(lab, col)] = mean(rs, pre, lambda r, c=k2: r["cfg_key"] == c and r["kind"] == "cfixed")
        out[(lab, "mean (cplx)")] = mean(rs, pre, lambda r: r["kind"] == "cfixed")
        out[(lab, "mean (real)")] = mean(rs, pre, lambda r: r["kind"] == "real")
        out[(lab, "retention")] = mean(rs, kk, lambda r: r["kind"] == "cfixed") * 100
    return out


def t34():
    d = rec("r6_quantile_cost")
    out = {}
    for r in d["rows"]:
        lab = (f"${r['n']:,}$",)
        out[(lab, "levels")] = r.get("levels", r.get("n_levels"))
        for col, tk in (("sort + gather", "t_sort_s"), ("partition + gather", "t_partition_s"),
                        ("histogram", "t_hist_s")):
            if tk in r:
                out[(lab, col)] = r[tk] * 1000
        for col, ek in (("exact error", "s2_rel_err_exact_mean"),
                        ("histogram error", "s2_rel_err_histogram_mean")):
            if ek in r:
                out[(lab, col)] = r[ek] * 100
    return out


def t35():
    rs = R("r5_band_floor")
    CASE35 = {"babble": "babble, SNR $20$ dB", "babble0": "babble, SNR $0$ dB",
              "gauss": "Gaussian, SNR $20$ dB", "clean": "noise-free"}
    out = {}
    for ck, c35 in CASE35.items():
        for b in (1, 4, 16, 64):
            pred = lambda r, c=ck, bb=b: r["case"] == c and r["bands"] == bb
            lab = (c35, str(b))
            out[(lab, "$\\hat\\nu/\\nu$")] = mean(rs, "nu_ratio", pred)
            out[(lab, "retention")] = mean(rs, "keep", pred) * 100
            out[(lab, "angle error")] = mean(rs, "A", pred)
    return out


def t36():
    rr = R("real_results")
    meth = {"NF-SSP": "NF-SSP", "median $t_e{=}0.02$": "SCA-median", "max $c{=}0.05$": "SCA-max",
            "top-5%": "top-K", "Oracle-A": "oracle_a_l1"}
    out = {}
    for lab, m in meth.items():
        for case in REAL_ORDER:
            out[(_rl(case), lab)] = mean(rr, "SIR",
                                           lambda r, mm=m, c=case: r["case"] == c and r["method"] == mm)
        out[("**mean**", lab)] = st.mean(
            [v for (rk, ck), v in out.items() if ck == lab and rk != ("**mean**",)])
    return out


def t39():
    d = rec("r10_gram_spectrum")
    out = {}
    for r in d["summary"]:
        if "median_shift_pct" not in r:
            continue
        J, M = r["J"], r["M"]
        if J == 2:
            lab = "$J=2$, $M\\in\\{2,3\\}$"
        elif J == 3 and M == 3:
            lab = "$J=3$, $M=3$"
        elif J == 3:
            lab = "$J=3$, $M=2$"
        else:
            lab = "$J=4$"
        lo, hi = r.get("cond_lo", 0.0), r.get("cond_hi", 0.0)
        if r.get("rank_deficient") or (lo is None and hi is None):
            col = "$\\lambda_{\\min}=0$"
        elif hi <= 1.5:
            col = "$\\kappa<1.5$"
        elif hi <= 3:
            col = "$1.5\\le\\kappa<3$"
        elif hi <= 10:
            col = "$3\\le\\kappa<10$"
        elif hi <= 100:
            col = "$10\\le\\kappa<100$"
        else:
            col = "$\\kappa\\ge100$"
        if J == 2:
            out[(lab, col)] = _pooled_shift(2, [2, 3], lo, hi)
        elif J == 4:
            out[(lab, col)] = _pooled_shift(4, [2, 3], 0.0, 1e18)
        else:
            out[(lab, col)] = r["median_shift_pct"]
    return out


def _pooled_shift(J: int, Ms, lo, hi) -> float:
    sub = rec("r10_gram_spectrum")["subset_records"]
    v = [r["shift_pct"] for r in sub if r["J"] == J and r["M"] in Ms and lo <= r["cond"] < hi]
    return st.median(v) if v else float("nan")


def t40():
    out = {}
    for lab, rname, gname in (("LibriSpeech dev-clean", "real_results", "real_grid"),
                              ("SLR70 Nigerian English", "r10_real_slr70", "r10_real_slr70_grid")):
        rs = R(rname)
        cols = ["SCA-median-opt", "SCA-max-opt", "top-K-opt"]
        per, nf, cv = [], [], []
        for case in REAL_ORDER:
            per.append(min(mean(rs, "A_angle_deg", lambda r, c=case, m=m:
                                r["case"] == c and r["method"] == m) for m in cols))
            nf.append(mean(rs, "A_angle_deg", lambda r, c=case: r["case"] == c and r["method"] == "NF-SSP"))
            cv.append(mean(rs, "A_angle_deg", lambda r, c=case: r["case"] == c and r["method"] == "SCA-median"))
        out[(lab, "NF-SSP")] = st.mean(nf)
        out[(lab, "conventional")] = st.mean(cv)
        out[(lab, "best-observed")] = st.mean(per)
        out[(lab, "NF penalty")] = st.mean([nf[i] / per[i] for i in range(len(per))])
        out[(lab, "conventional penalty")] = st.mean([cv[i] / per[i] for i in range(len(per))])
        out[(lab, "NF better in")] = sum(1 for i in range(len(per)) if nf[i] < cv[i])
    return out


def t41():
    rs = R("r2_noise_family")
    labmap = {"Gaussian (model law)": "gauss", "Laplacian (heavy tail)": "laplace",
              "impulsive": "impulsive", "colored Gaussian": "colored",
              "uniform (light tail)": "uniform", "Student-$t$ ($\\nu=3$)": "t3"}
    out = {}
    for L, k in labmap.items():
        out[(L, "$\\hat\\sigma^2/\\sigma^2$, $p=0.05$")] = mean(
            rs, "s2_ratio", lambda r, kk=k: r["kind"] == kk and r["p"] == 0.05)
        for p, col in ((0.20, "$p=0.20$"), (0.40, "$p=0.40$")):
            out[(L, col)] = mean(rs, "s2_ratio", lambda r, kk=k, pp=p: r["kind"] == kk and r["p"] == pp)
        sub = [r for r in rs if r["kind"] == k and r["p"] == 0.05]
        out[(L, "gate applied")] = float(sum(1 for r in sub if r["gate_on"]))
        out[(L, "angle error NF#0")] = mean(sub, "A_nf")
        out[(L, "angle error classical#1")] = mean(sub, "A_classical")
    return out



def t42():
    rs = R("r2_angle_sweep")
    out = {}
    for a in sorted({r["angle_deg"] for r in rs}):
        lab = (f"${int(a)}\\degree$",)
        for p, col in ((0.05, "$p=0.05$"), (0.10, "$p=0.10$"), (0.20, "$p=0.20$"), (0.40, "$p=0.40$")):
            pred = lambda r, aa=a, pp=p: r["angle_deg"] == aa and r["p"] == pp
            out[(lab, col + " NF")] = mean(rs, "A_nf", pred)
            out[(lab, col + " classical")] = mean(rs, "A_classical", pred)
            out[(lab, col + " te5")] = mean(rs, "A_te5", pred)
    return out


def t43():
    rs = R("r5_source_number")
    labmap = {"collinearity + balance only": "collin",
              "conventional median $t_e{=}0.02$": "med0.02",
              "conventional median $t_e{=}5$": "med5", "**NF-SSP**": "nf"}
    out = {}
    for L, k in labmap.items():
        sub = [r for r in rs if r["method"] == k and r["peak_ratio"] == 0.2]
        out[(L, "exact")] = sum(1 for r in sub if r["n_est"] == r["n_true"]) / len(sub) * 100
        out[(L, "mean $\\lvert\\Delta N\\rvert$")] = st.mean([r["abs_err"] for r in sub])
        out[(L, "$\\hat N<N$")] = sum(1 for r in sub if r["n_est"] < r["n_true"]) / len(sub) * 100
        out[(L, "$\\hat N>N$")] = sum(1 for r in sub if r["n_est"] > r["n_true"]) / len(sub) * 100
        out[(L, "angle error at $\\hat N$")] = st.mean([r["A"] for r in sub])
        out[(L, "points kept")] = st.mean([r["keep"] for r in sub]) * 100
    return out


def t44():
    rs = R("r8_n_mismatch")
    labmap = {"collinearity + balance only": "collin",
              "conventional median $t_e{=}0.02$": "med0.02", "**NF-SSP**": "nf"}
    out = {}
    for L, k in labmap.items():
        for d, col in ((-2, "$\\hat N=N-2$"), (-1, "$N-1$"), (0, "$N$"), (1, "$N+1$"), (2, "$N+2$")):
            out[(L, col)] = st.mean([r["A"] for r in rs if r["method"] == k and r["delta"] == d])
    return out


def t45():
    rs = R("nfr_results")
    out = {}
    for a, col in ((1e-2, "$10^{-2}$"), (1e-3, "$10^{-3}$"), (1e-4, "$10^{-4}$"),
                   (1e-5, "$10^{-5}$"), (1e-6, "$10^{-6}$")):
        sub = [r for r in rs if r["experiment"] == f"E7_alpha{a:.0e}"]
        if not sub:
            continue
        out[("mean angle error ($°$)", col)] = st.mean([r["A_angle_deg"] for r in sub])
        per = defaultdict(list)
        for r in sub:
            per[r["cfg_key"]].append(r["A_angle_deg"])
        out[("worst regime ($°$)", col)] = max(st.mean(v) for v in per.values())
        out[("(worst single run, $°$)", col)] = max(r["A_angle_deg"] for r in sub)
    return out


# ---------------------------------------------------------------- 登记表

# 表号 -> (抽取器, 标签列数)
REGISTRY = {
    1: (t1, 1), 2: (t2, 1), 3: (t3, 2), 6: (t6, 1), 7: (t7, 0), 8: (t8, 2), 9: (t9, 2), 10: (t10, 2),
    11: (t11, 1), 12: (t12, 1), 13: (t13, 1), 14: (lambda: {**t14(), **t14_sdr()}, 1), 15: (t15, 1), 16: (t16, 1),
    17: (t17, 1), 18: (t18, 1), 19: (t19, 1), 21: (t21, 1), 23: (lambda: {**t23(), **t23b()}, 1), 24: (t24, 1),
    25: (t25, 1), 26: (t26, 1), 28: (t28, 1), 29: (t29, 1), 31: (t31, 1), 32: (t32, 1),
    34: (t34, 1), 35: (t35, 2), 36: (t36, 1), 39: (t39, 1), 40: (t40, 1), 41: (t41, 1),
    42: (t42, 1), 43: (t43, 1), 44: (t44, 1), 45: (t45, 1),
}

# 一格多值的表：抽取器给 (行, 列#i) 形式的键，与稿件该格第 i 个数对账。
# Table 18 每格是 "precision / recall"；Table 42 每格是 "NF / classical / te5"；
# Table 33 每格是 "retention / recall / angle error"；Table 40 的 "NF better in" 是 "15/15"。
SPLIT = {18: 2, 42: 3, 33: 3, 40: 2, 41: 2, 23: 2, 25: 2}


def _rowkey11(rows: list[list[str]]) -> list[tuple[str, ...]]:
    """Table 11 的 "relative improvement" 行第一列只写这四个字，靠前一组补上前缀。"""
    out, group = [], ""
    for row in rows:
        c0 = row[0] if row else ""
        if "relative improvement" in c0:
            out.append((group + ", relative improvement",))
        else:
            group = c0
            out.append((c0,))
    return out

ROWKEY = {11: _rowkey11}

# 记录里确实没有对应量的格子：逐条写明原因，避免被误读成"漏查"。
KNOWN_GAPS = [
    ("Table 1", "noise mass $\\ge1/2$", "布尔列，稿件印 yes / no，记录只给 π0；已在抽取器里按 π0>=0.5 复核"),
    ("Table 12", "overlap", "E8_structured 的记录不写 overlap（data.empirical_overlap 每题现算但不入档），"
                            "四值 2.40/2.34/2.33/2.05 无法从记录复算"),
]

# 记录 -> 稿件的表号。REGISTRY 里的由抽取器自动派生；SKIP 的几张手写
# （它们的抽取器没登记，或有专门的 verifier）。README 的两张映射表据此自检。
RECORD_TABLES_EXTRA = {
    "guard_validation": [4],
    "real_mechanism": [15],
    "r2_eta_sweep": [20],
    "r2_runtime": [22],
    "r5_alpha_cell": [33],
    "r5_complex": [30, 31, 32],
    "r8_real_room": [38],
    "real_results": [13, 14, 36, 37],
    "r2_stats": [23],
    "real_grid": [13],
    "weighted_variant": [17],
    "r10_real_slr70": [40],
}


def record_tables() -> dict[str, list[int]]:
    """记录 -> 稿件表号。抽取器自动派生 + RECORD_TABLES_EXTRA 补齐。"""
    import inspect
    out: dict[str, set[int]] = defaultdict(set)
    for tn, (ex, _lc) in REGISTRY.items():
        src = inspect.getsource(ex)
        for name in set(re.findall(r'(?:R|rec)\("([a-z0-9_]+)"\)', src)):
            out[name].add(tn)
    for name, tabs in RECORD_TABLES_EXTRA.items():
        out[name].update(tabs)
    return {k: sorted(v) for k, v in sorted(out.items())}


SKIP_WITH_REASON = {
    4: "自检诊断表：多为区间与决策字符串（keep/preferred/decision），非纯数值",
    5: "超参数清单，无对应记录",
    20: "r2_eta_sweep 的 64 格已在 R14c 轮全量对过",
    22: "r2_runtime 已对过（gate subtotal 0.51ms/0.66%、4.09ms/0.17%，份额逐行复核）",
    27: "参考尺度分类表，无数值",
    30: "r5_complex 的六制度 × 三列 + σ̂²/σ² 已全量对过",
    33: "r5_alpha_cell 由 verify_r5.py 的 alpha_cells() 逐格对（四格 × 五 α × 三个量）；R15 给该表补了 α=10^-2 列并改为升序，verify_r5 的 M36_ALPHAS 已同步",
    37: "Table 14 / 36 的均值行，check_tables.py 已覆盖",
    38: "r8_real_room 已全量对过（六房间 × 四列）",
}

TOL_REL = 0.006
TOL_ABS = 0.006



def close(v: float, ev: float, text: str) -> bool:
    """按稿件该格印出的小数位定容差：印 1 位小数允许半个最低位，另叠相对容差。"""
    m = re.search(r"\.(\d+)", str(text).replace("$", ""))
    dp = len(m.group(1)) if m else 0
    dec_tol = 0.5 * (10 ** -dp)
    return abs(v - ev) <= max(dec_tol, TOL_ABS, TOL_REL * abs(ev))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=MANUSCRIPT)
    ap.add_argument("--json", default=None)
    ap.add_argument("--table", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.manuscript) as fh:
        tables = parse_tables(fh.read())

    report = {}
    tot_cells = tot_bad = tot_unmatched = 0
    for tnum in sorted(tables):
        if args.table is not None and tnum != args.table:
            continue
        tbl = tables[tnum]
        entry = {"caption": tbl["caption"][:80], "n_rows": sum(len(b["rows"]) for b in tbl["blocks"]),
                 "n_cols": [len(b["header"]) for b in tbl["blocks"]]}
        if tnum in SKIP_WITH_REASON:
            entry["status"], entry["note"] = "skipped", SKIP_WITH_REASON[tnum]
            report[tnum] = entry
            continue
        if tnum not in REGISTRY:
            entry["status"], entry["note"] = "no-extractor", "未登记抽取器"
            report[tnum] = entry
            continue
        extractor, label_cols = REGISTRY[tnum]
        rowkey_fn = ROWKEY.get(tnum)
        try:
            exp = extractor()
        except Exception as exc:  # noqa: BLE001
            entry["status"], entry["note"] = "error", f"{type(exc).__name__}: {exc}"
            report[tnum] = entry
            continue

        def _split(key):
            k, ck = key
            return ((k,) if isinstance(k, str) else tuple(k), ck)

        if rowkey_fn is not None:
            tbl = dict(tbl)
            tbl["blocks"] = [{"header": b["header"],
                              "rows": rowkey_fn(b["rows"])} for b in tbl["blocks"]]
        expn = {}
        for k, v in exp.items():
            rk, ck = _split(k)
            expn[(tuple(norm(c) for c in rk), norm(ck))] = v
        bad, unmatched, matched = [], [], 0
        for (rkey, chead, ctext) in flat_cells(tbl, label_cols):
            rn = tuple(norm(c) for c in rkey)
            cn = norm(chead)
            vals = nums(ctext)
            # 单值格
            cands = [(k, v) for k, v in expn.items() if k[0] == rn and k[1] == cn]
            if tnum in SPLIT and len(vals) >= 1:
                ns = SPLIT[tnum]
                for ix in range(ns):
                    sub = [(k, v) for k, v in expn.items()
                           if k[0] == rn and k[1] == f"{cn}#{ix}"]
                    if len(vals) <= ix or not sub:
                        continue
                    ev = sub[0][1]
                    if ev is None or (isinstance(ev, float) and math.isnan(ev)):
                        continue
                    matched += 1
                    if not close(vals[ix], ev, vals[ix]):
                        bad.append({"row": "/".join(rkey), "col": f"{chead}#{ix}",
                                    "manuscript": vals[ix], "record": round(ev, 6)})
                continue
            if not cands:
                if vals:
                    unmatched.append({"row": "/".join(rkey), "col": chead, "text": ctext})
                continue
            ev = cands[0][1]
            if ev is None or (isinstance(ev, float) and math.isnan(ev)):
                continue
            if len(vals) > 1:
                # 未声明的多值格：跳过（不猜）
                unmatched.append({"row": "/".join(rkey), "col": chead, "text": ctext,
                                  "why": "multi-valued, no parser declared"})
                continue
            if not vals:
                continue
            matched += 1
            tol_key = (rn, "counting-errortol")
            if tol_key in expn:
                ev2 = expn[tol_key]
                if abs(vals[0] - ev) <= ev2:
                    continue
            if not close(vals[0], ev, ctext):
                bad.append({"row": "/".join(rkey), "col": chead, "manuscript": vals[0],
                            "record": round(ev, 6)})
        entry["status"] = "mismatch" if bad else "ok"
        entry["cells_matched"] = matched
        entry["mismatch"] = bad
        entry["unmatched"] = unmatched
        report[tnum] = entry
        tot_cells += matched
        tot_bad += len(bad)
        tot_unmatched += len(unmatched)

    rmap = record_tables()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"tables": report, "cells_matched": tot_cells,
                       "cells_mismatched": tot_bad, "cells_unmatched": tot_unmatched,
                       "record_tables": rmap},
                      fh, indent=1, ensure_ascii=False)
    if args.verbose:
        print("\n记录 -> 表号：")
        for k, v in rmap.items():
            print(f"  {k:26s} Tables {v}")

    done = [t for t in report if report[t]["status"] in ("ok", "mismatch")]
    print(f"对稿 {len(done)} 张表 | 命中 {tot_cells} 格 | 不一致 {tot_bad} 格 | 未配上 {tot_unmatched} 格")
    for t, col, why in KNOWN_GAPS:
        print(f"  · {t} / {col}：{why}")
    for tnum in sorted(report):
        e = report[tnum]
        if e["status"] == "mismatch":
            print(f"\n✗ Table {tnum}: {e['caption']}")
            for b in e["mismatch"]:
                print(f"    {b['row']} / {b['col']}: 稿={b['manuscript']} 记录={b['record']}")
        elif e["status"] != "ok":
            print(f"  – Table {tnum}: {e['status']} — {e['note']}")
        elif args.verbose:
            print(f"  ✓ Table {tnum}: {e['cells_matched']} 格全中")
            for u in e["unmatched"]:
                print(f"      未配上: {u['row']} / {u['col']} = {u['text']} {u.get('why','')}")
    return 1 if tot_bad else 0


if __name__ == "__main__":
    sys.exit(main())
