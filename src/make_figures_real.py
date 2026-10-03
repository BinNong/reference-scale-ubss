"""真实数据实验的论文级图表（英文标签、矢量输出）。

图 1（fig_real_scale）：参考尺度诊断
  (a) A 误差 vs 中位数倍数 t_e（对数横轴），逐档曲线 + 本文方法的等效 t_e 位置
  (b) A 误差 vs 最大值比例系数 c
  (c) 各参考尺度下"保留点数"随档位的变化 —— 说明为什么同一系数无法通用
图 2（fig_real_main）：主结果
  (a) 各方法在全部真实配置上的 A 误差（含逐档 oracle）
  (b) 各方法在全部真实配置上的 SDR
  (c) 机理：方向角误差随能量的变化（真实语音 vs 合成）

用法: ../.venv/bin/python make_figures_real.py --results ../results --out ../results/figures_real
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "figure.dpi": 150,
})

C_NF = "#c0392b"      # 本文方法
C_MED = "#2c7fb8"
C_MAX = "#31a354"
C_TOPK = "#756bb1"
C_GREY = "#7f7f7f"

CASE_LABEL = {
    "R1_res:w256": "win 256", "R1_res:w512": "win 512",
    "R1_res:w1024": "win 1024", "R1_res:w2048": "win 2048",
    "R2_n:N3": "N=3", "R2_n:N5": "N=5", "R2_n:N6": "N=6",
    "R3_snr:snr00": "SNR 0 dB", "R3_snr:snr10": "SNR 10 dB",
    "R3_snr:snr30": "SNR 30 dB", "R3_snr:snr40": "SNR 40 dB",
    "R4_dense:d1": "1 dense src", "R4_dense:d2": "2 dense src",
    "R5_noise:babble": "babble noise", "R5_noise:clean": "noise-free",
}


def _cmap(n):
    return plt.cm.viridis(np.linspace(0.05, 0.9, n))


def fig_scale(grid, res, out):
    cases = [c for c in CASE_LABEL if f"{c}|median" in grid]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    cols = _cmap(len(cases))

    # (a) 中位数参考：误差 vs t_e
    ax = axes[0]
    for c, col in zip(cases, cols):
        d = grid[f"{c}|median"]
        xs = np.array([float(k) for k in d], dtype=float)
        ys = np.array([d[k] for k in d], dtype=float)
        o = np.argsort(xs)
        ax.plot(xs[o], np.maximum(ys[o], 1e-2), "-o", color=col, ms=2.4, lw=1.1,
                label=CASE_LABEL[c])
    nf = [r["t_median"] for r in res["records"]
          if r["method"] == "NF-SSP" and r.get("t_median")]
    if nf:
        ax.axvspan(np.quantile(nf, 0.1), np.quantile(nf, 0.9), color=C_NF,
                   alpha=0.16, lw=0)
        ax.axvline(float(np.median(nf)), color=C_NF, ls="--", lw=1.2)
    ax.axvline(0.02, color=C_GREY, ls=":", lw=1.3)
    ax.set_xscale("log")
    ax.set_xlabel(r"median-referenced multiple $t_e$")
    ax.set_ylabel("mixing-matrix angle error (deg)")
    ax.set_title("(a) median-referenced gate")
    ax.set_ylim(0, 24)
    ax.text(0.024, 21, r"conventional" "\n" r"$t_e=0.02$", fontsize=6, color=C_GREY)
    ax.legend(ncol=2, loc="lower left", framealpha=0.85, handlelength=1.2)

    # (b) 最大值参考：误差 vs c
    ax = axes[1]
    for c, col in zip(cases, cols):
        d = grid.get(f"{c}|max")
        if not d:
            continue
        xs = np.array([float(k) for k in d], dtype=float)
        ys = np.array([d[k] for k in d], dtype=float)
        o = np.argsort(xs)
        ax.plot(xs[o], np.maximum(ys[o], 1e-2), "-o", color=col, ms=2.4, lw=1.1)
    ax.axvline(0.05, color=C_MAX, ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"max-referenced coefficient $c$")
    ax.set_title("(b) max-referenced gate")
    ax.set_ylim(0, 24)

    # (c) 保留比例随档位的变化
    ax = axes[2]
    keep_nf = [np.mean([r["keep_frac"] for r in res["records"]
                        if r["method"] == "NF-SSP" and r.get("case") == c and r.get("keep_frac")])
               for c in cases]
    ax.barh(np.arange(len(cases)), keep_nf, color=C_NF, alpha=0.8, height=0.6)
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels([CASE_LABEL[c] for c in cases], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("surviving fraction after the gate")
    ax.set_title("(c) NF-SSP gate retention")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"  {out}.pdf / .png")


def fig_main(res, mech, out):
    recs = res["records"]
    cases = sorted({r["case"] for r in recs},
                   key=lambda c: list(CASE_LABEL).index(c) if c in CASE_LABEL else 99)
    METHODS = [("NF-SSP", C_NF), ("SCA-median", C_MED), ("SCA-max", C_MAX),
               ("top-K", C_TOPK), ("SCA-median-opt", C_GREY),
               ("SCA-max-opt", "#b2df8a"), ("top-K-opt", "#bcbddc")]
    NICE = {"NF-SSP": "NF-SSP (proposed)", "SCA-median": r"median $t_e{=}0.02$",
            "SCA-max": r"max $c{=}0.05$", "top-K": "top-5%", "FCM-median": "FCM + median gate",
            "SCA-median-opt": "median-opt (oracle)", "SCA-max-opt": "max-opt (oracle)",
            "top-K-opt": "top-K-opt (oracle)"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    x = np.arange(len(cases))
    nb = len(METHODS)
    w = 0.8 / nb

    for ax, metric, ttl, ylim in [(axes[0], "A_angle_deg",
                                   "(a) mixing-matrix angle error",
                                   (0, 16)),
                                  (axes[1], "SDR", "(b) SDR", (-6, 11))]:
        for i, (m, col) in enumerate(METHODS):
            vals = [np.mean([r[metric] for r in recs
                             if r["case"] == c and r["method"] == m] or [np.nan])
                    for c in cases]
            ax.bar(x + (i - (nb - 1) / 2) * w, vals, w, color=col,
                   label=NICE[m], edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([CASE_LABEL[c] for c in cases], rotation=52,
                           ha="right", fontsize=6.2)
        ax.set_title(ttl)
        ax.set_ylim(*ylim)
        if metric == "A_angle_deg":
            ax.set_ylabel("angle error (deg)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=6.0, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), frameon=False, handlelength=1.1,
               columnspacing=1.0)

    # (c) 机理：方向角误差 vs 能量分箱
    ax = axes[2]
    styles = {"real|real w1024 S20": ("-o", C_NF, r"real speech, SNR $20$ dB"),
              "real|real w1024 N3": ("-^", "#8e44ad", r"real speech, $N{=}3$"),
              "synth|synth tf_p02": ("--o", C_MED, r"synthetic $p{=}0.02$"),
              "synth|synth tf_p40": ("--^", "#8c564b", r"synthetic $p{=}0.40$")}
    for k, (mk, col, lab) in styles.items():
        if k not in mech:
            continue
        b = mech[k]
        xs = [max(v["e_over_max"], 1e-9) for v in b]
        ys = [v["ang_med"] for v in b]
        ax.plot(xs, ys, mk, color=col, ms=3, lw=1.1, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"point energy / $\max_l e_l$")
    ax.set_ylabel("median direction error (deg)")
    ax.set_title("(c) quality of single-source points")
    ax.legend(fontsize=5.6, framealpha=0.92, loc="lower left", labelspacing=0.28,
              handlelength=1.2, borderpad=0.3)
    ax.set_ylim(0, None)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    print(f"  {out}.pdf / .png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/figures_real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.results, "real_grid.json")) as f:
        grid = json.load(f)
    with open(os.path.join(a.results, "real_results.json")) as f:
        res = json.load(f)
    mech_path = os.path.join(a.results, "real_mechanism.json")
    mech = json.load(open(mech_path)) if os.path.exists(mech_path) else {}

    fig_scale(grid, res, f"{a.out}/fig_real_scale")
    fig_main(res, mech, f"{a.out}/fig_real_main")


if __name__ == "__main__":
    main()
