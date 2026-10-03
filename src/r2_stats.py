"""R2-6  统计规范补强（审稿意见第十五节）。

对已有的归档结果补三件论文目前缺的东西：
  1. 配对差的 95% 置信区间；
  2. 效应量（配对 Cohen's d_z）；
  3. 多重比较校正（Holm 与 Benjamini–Hochberg），并说明原稿未校正的理由。

用法（服务器或本地均可，只要 results/*.json 在）：
    python3 r2_stats.py --out ../results/r2_stats.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import stats


def paired(diff: np.ndarray) -> dict:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n < 2:
        return dict(n=int(n), mean=float("nan"), sd=float("nan"),
                    ci_lo=float("nan"), ci_hi=float("nan"),
                    t=float("nan"), p=float("nan"), dz=float("nan"))
    m, sd = float(diff.mean()), float(diff.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))
    t, p = stats.ttest_rel(np.zeros(n), diff)
    return dict(n=int(n), mean=m, sd=sd, se=float(se),
                ci_lo=m - tcrit * se, ci_hi=m + tcrit * se,
                t=float(t), p=float(p),
                dz=float(m / sd) if sd > 0 else float("nan"))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return [float(x) for x in adj]


def bh(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order):
        val = pvals[i] * m / (m - rank)
        running = min(running, val)
        adj[i] = min(running, 1.0)
    return [float(x) for x in adj]


def collect_nfr(path: str) -> list[tuple[str, np.ndarray]]:
    """§8.1 稀疏阶梯：NF-SSP vs SCA-default，逐档配对（按种子）。"""
    d = json.load(open(path))
    R = d["records"]
    cfgs = sorted({x["cfg_key"] for x in R if x["experiment"] == "E1_sparsity"})
    out = []
    for c in cfgs:
        a = {x["seed"]: x["A_angle_deg"] for x in R if x["experiment"] == "E1_sparsity"
             and x["cfg_key"] == c and x["method"] == "NF-SSP"}
        b = {x["seed"]: x["A_angle_deg"] for x in R if x["experiment"] == "E1_sparsity"
             and x["cfg_key"] == c and x["method"] == "SCA-default"}
        ks = sorted(set(a) & set(b))
        out.append((f"ladder {c}: SCA-default − NF-SSP",
                    np.array([b[k] - a[k] for k in ks])))
    # 池化
    a = {(x["cfg_key"], x["seed"]): x["A_angle_deg"] for x in R
         if x["experiment"] == "E1_sparsity" and x["method"] == "NF-SSP"}
    b = {(x["cfg_key"], x["seed"]): x["A_angle_deg"] for x in R
         if x["experiment"] == "E1_sparsity" and x["method"] == "SCA-default"}
    ks = sorted(set(a) & set(b))
    out.append(("ladder pooled: SCA-default − NF-SSP",
                np.array([b[k] - a[k] for k in ks])))
    return out


def collect_real(path: str) -> list[tuple[str, np.ndarray]]:
    """§9.3：真实语音上 NF-SSP vs 各基线，池化 15 配置 × 种子。"""
    d = json.load(open(path))
    R = d["records"]
    meths = sorted({x["method"] for x in R})
    base = "NF-SSP"
    A = {(x["case"], x["seed"]): x["A_angle_deg"] for x in R if x["method"] == base}
    out = []
    for m in meths:
        if m == base:
            continue
        B = {(x["case"], x["seed"]): x["A_angle_deg"] for x in R if x["method"] == m}
        ks = sorted(set(A) & set(B))
        if len(ks) < 5:
            continue
        out.append((f"real: {m} − NF-SSP", np.array([B[k] - A[k] for k in ks])))
    return out


def family(title: str, rows: list[tuple[str, np.ndarray]]) -> dict:
    stats_rows = []
    for lab, diff in rows:
        s = paired(diff)
        s["label"] = lab
        stats_rows.append(s)
    ps = [r["p"] for r in stats_rows]
    ho, be = holm(ps), bh(ps)
    for r, h, b in zip(stats_rows, ho, be):
        r["p_holm"], r["p_bh"] = h, b
    print(f"\n=== {title} ===")
    print(f"{'comparison':<44}{'n':>4}{'Δmean':>10}{'95% CI':>20}{'p':>11}"
          f"{'d_z':>8}{'p_holm':>10}{'p_BH':>10}")
    print("-" * 117)
    for r in stats_rows:
        ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
        print(f"{r['label']:<44}{r['n']:>4}{r['mean']:>+10.3f}{ci:>20}"
              f"{r['p']:>11.2e}{r['dz']:>+8.2f}{r['p_holm']:>10.2e}{r['p_bh']:>10.2e}")
    nsig_raw = sum(1 for r in stats_rows if r["p"] < 0.05)
    nsig_ho = sum(1 for r in stats_rows if r["p_holm"] < 0.05)
    nsig_bh = sum(1 for r in stats_rows if r["p_bh"] < 0.05)
    print(f"  → 显著数：原始 {nsig_raw}/{len(stats_rows)}，"
          f"Holm {nsig_ho}，BH {nsig_bh}")
    return dict(title=title, rows=stats_rows,
                n_significant=dict(raw=nsig_raw, holm=nsig_ho, bh=nsig_bh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../results")
    ap.add_argument("--out", default="../results/r2_stats.json")
    a = ap.parse_args()

    out = {}
    nfr = os.path.join(a.root, "nfr_results.json")
    real = os.path.join(a.root, "real_results.json")
    if os.path.exists(nfr):
        out["nfr_ladder"] = family("合成稀疏阶梯（§8.1）", collect_nfr(nfr))
    if os.path.exists(real):
        out["real_speech"] = family("真实语音（§9.3，池化 15 配置 × 8 seeds）",
                                    collect_real(real))

    purity = os.path.join(a.root, "purity_stage2.json")
    if os.path.exists(purity):
        P = json.load(open(purity))["records"]
        P = [r for r in P if r["dom"] == "real"]
        meths = sorted({r["method"] for r in P})
        rows = []
        for cand in meths:
            if cand == "NF":
                continue
            A = {(r["case"], r["seed"]): r["A"] for r in P if r["method"] == "NF"}
            B = {(r["case"], r["seed"]): r["A"] for r in P if r["method"] == cand}
            ks = sorted(set(A) & set(B))
            rows.append((f"real: {cand} − NF", np.array([B[k] - A[k] for k in ks])))
        out["real_weighting"] = family("真实语音加权族（§9.5）", rows)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
