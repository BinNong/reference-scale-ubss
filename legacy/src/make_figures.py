"""从实验结果 JSON 生成论文级图表（矢量 PDF + 高分辨率 PNG）。

用法（项目 src 目录下）:
    ../.venv/bin/python make_figures.py --in ../results/raw/main_results.json --out ../results/figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import BASELINE_LABELS

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

# 颜色/标记：本方法用醒目红色（中国习惯：红=好/强调），基线用冷色系
STYLE = {
    "SA-DUN":            dict(color="#C0392B", marker="o", lw=2.0, ms=5, zorder=10),
    "oracle_a_l1":       dict(color="#2C3E50", marker="s", lw=1.2, ls="--"),
    "ssp_kmeans_l1":     dict(color="#2980B9", marker="^", lw=1.2),
    "ssp_fcm_sp":        dict(color="#27AE60", marker="v", lw=1.2),
    "pf_auto_l1":        dict(color="#8E44AD", marker="D", lw=1.2),
    "sl0_l1":            dict(color="#E67E22", marker="P", lw=1.2),
    "duet":              dict(color="#7F8C8D", marker="x", lw=1.2, ls=":"),
}
LABEL = dict(BASELINE_LABELS)
LABEL["SA-DUN"] = "SA-DUN (proposed)"
LABEL["oracle_a_l1"] = "Oracle-A (true A)"
LABEL["pf_auto_l1"] = "Potential-function (auto-K)"

CFG_LABEL = {
    "tf_p02": "p=0.02", "tf_p05": "p=0.05", "tf_p10": "p=0.10",
    "tf_p20": "p=0.20", "tf_p40": "p=0.40", "tf_gauss": "dense",
}
CFG_ORDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]


def load(path):
    with open(path) as f:
        return json.load(f)["results"]


def agg(records, by, metric):
    """按 by 字段聚合 metric，返回 {key: (mean, std, n)}。"""
    g = defaultdict(list)
    for r in records:
        g[r[by]].append(r[metric])
    out = {}
    for k, v in g.items():
        v = np.asarray(v, dtype=float)
        out[k] = (float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0, v.size)
    return out


def methods_in(records):
    seen, order = set(), []
    for r in records:
        if r["method"] not in seen:
            seen.add(r["method"])
            order.append(r["method"])
    order.sort(key=lambda m: (m != "SA-DUN", m != "oracle_a_l1", m))
    return order


def fig_sparsity(records, out_dir):
    """图：混合矩阵夹角误差 & SDR 随稀疏度的变化。"""
    recs = [r for r in records if r["method"] == "SA-DUN" or True]
    ms = methods_in(recs)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1))

    specs = [
        ("A_angle_deg", "Mixing-matrix angle error (deg)", "lower is better", 0, 45),
        ("SDR", "SDR (dB)", "higher is better", None, None),
        ("SIR", "SIR (dB)", "higher is better", None, None),
    ]
    xs = np.arange(len(CFG_ORDER))
    for ax, (metric, ylab, note, lo, hi) in zip(axes, specs):
        for m in ms:
            sub = [r for r in recs if r["method"] == m]
            a = agg(sub, "cfg_key", metric)
            mu = [a.get(k, (np.nan, 0, 0))[0] for k in CFG_ORDER]
            sd = [a.get(k, (np.nan, 0, 0))[1] for k in CFG_ORDER]
            st = STYLE.get(m, dict(color="gray", marker="."))
            ax.errorbar(xs, mu, yerr=sd, capsize=2, label=LABEL.get(m, m), **st)
        ax.set_xticks(xs)
        ax.set_xticklabels([CFG_LABEL[k] for k in CFG_ORDER])
        ax.set_xlabel("TF activation probability p (denser to the right)")
        ax.set_ylabel(ylab)
        ax.set_title(note)
        if lo is not None:
            ax.set_ylim(lo, hi)
    axes[0].legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_sparsity.{ext}"))
    plt.close(fig)


def fig_snr(records, out_dir):
    """图：SDR 随 SNR 的变化（按稀疏度分面）。"""
    cfgs = [c for c in CFG_ORDER if any(r["cfg_key"] == c for r in records)]
    ms = methods_in(records)
    n = len(cfgs)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.0), squeeze=False)
    for ax, cf in zip(axes[0], cfgs):
        sub = [r for r in records if r["cfg_key"] == cf]
        for m in ms:
            s2 = [r for r in sub if r["method"] == m]
            a = agg(s2, "snr_db", "SDR")
            ks = sorted(a)
            if not ks:
                continue
            st = STYLE.get(m, dict(color="gray"))
            ax.errorbar(ks, [a[k][0] for k in ks], yerr=[a[k][1] for k in ks],
                        capsize=2, label=LABEL.get(m, m), **st)
        ax.set_title(CFG_LABEL.get(cf, cf))
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("SDR (dB)")
    axes[0][0].legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_snr.{ext}"))
    plt.close(fig)


def fig_nsources(records, out_dir):
    """图：SDR 与 A 误差随源数目的变化。"""
    ms = methods_in(records)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, metric, ylab in [(axes[0], "SDR", "SDR (dB)"),
                             (axes[1], "A_angle_deg", "Mixing-matrix angle error (deg)")]:
        for m in ms:
            sub = [r for r in records if r["method"] == m]
            a = agg(sub, "n_true", metric)
            ks = sorted(a)
            if not ks:
                continue
            st = STYLE.get(m, dict(color="gray"))
            ax.errorbar(ks, [a[k][0] for k in ks], yerr=[a[k][1] for k in ks],
                        capsize=2, label=LABEL.get(m, m), **st)
        ax.set_xlabel("True number of sources N")
        ax.set_ylabel(ylab)
        ax.set_xticks(sorted({r["n_true"] for r in records}))
    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_nsources.{ext}"))
    plt.close(fig)


def fig_ncount(records, out_dir):
    """图：源数目估计准确率。"""
    ms = methods_in(records)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    # 左：完全正确率 vs 稀疏度
    for m in ms:
        sub = [r for r in records if r["method"] == m]
        a = agg(sub, "cfg_key", "n_exact")
        ks = [k for k in CFG_ORDER if k in a]
        st = STYLE.get(m, dict(color="gray"))
        axes[0].plot(range(len(ks)), [a[k][0] for k in ks], label=LABEL.get(m, m), **st)
        axes[0].set_xticks(range(len(ks)))
        axes[0].set_xticklabels([CFG_LABEL[k] for k in ks])
    axes[0].set_ylabel("Exact-match rate of source count")
    axes[0].set_xlabel("Sparsity regime")
    axes[0].set_ylim(-0.03, 1.03)
    # 右：平均绝对误差 vs 真实源数
    for m in ms:
        sub = [r for r in records if r["method"] == m]
        a = agg(sub, "n_true", "n_abs_err")
        ks = sorted(a)
        st = STYLE.get(m, dict(color="gray"))
        axes[1].plot(ks, [a[k][0] for k in ks], label=LABEL.get(m, m), **st)
    axes[1].set_ylabel("Mean absolute error of source count")
    axes[1].set_xlabel("True number of sources N")
    axes[0].legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_ncount.{ext}"))
    plt.close(fig)


def fig_time(records, out_dir):
    """图：运行时间对比（对数坐标）。"""
    ms = methods_in(records)
    names, vals, errs = [], [], []
    for m in ms:
        v = [r["time_s"] for r in records if r["method"] == m]
        if not v:
            continue
        names.append(LABEL.get(m, m))
        vals.append(float(np.mean(v)))
        errs.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    colors = [STYLE.get(m, {}).get("color", "gray") for m in ms]
    ax.barh([names[i] for i in order], [vals[i] for i in order],
            xerr=[errs[i] for i in order], capsize=3,
            color=[colors[i] for i in order], alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per problem (s, log scale)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_time.{ext}"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/raw/main_results.json")
    ap.add_argument("--out", default="../results/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res = load(args.inp)

    if "E1_sparsity" in res:
        fig_sparsity(res["E1_sparsity"], args.out)
        print("已生成 fig_sparsity")
    if "E2_snr" in res:
        fig_snr(res["E2_snr"], args.out)
        print("已生成 fig_snr")
    if "E3_nsources" in res:
        fig_nsources(res["E3_nsources"], args.out)
        print("已生成 fig_nsources")
    if "E4_ncount" in res:
        fig_ncount(res["E4_ncount"], args.out)
        print("已生成 fig_ncount")
    if "E6_time" in res:
        fig_time(res["E6_time"], args.out)
        print("已生成 fig_time")
    print(f"图表输出目录: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
