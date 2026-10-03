"""生成新版论文图表（基于 10-seed 实测数据，内嵌数据保证可追溯）。

用法（服务器 src 目录下）:
    ../.venv/bin/python make_figures_v2.py --out ../results/figures_v2
"""
from __future__ import annotations
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "figure.dpi": 120, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})

STYLE = {
    "MDDE": dict(color="#C0392B", marker="o", lw=2.0, ms=6, zorder=10),
    "SCA-L1": dict(color="#2980B9", marker="^", lw=1.2),
    "SCA-SP": dict(color="#27AE60", marker="v", lw=1.2),
    "PF": dict(color="#8E44AD", marker="D", lw=1.2),
    "SL0": dict(color="#E67E22", marker="P", lw=1.2),
    "DUET": dict(color="#7F8C8D", marker="x", lw=1.2, ls=":"),
}
CFG = ["p=0.02", "p=0.05", "p=0.10", "p=0.20", "p=0.40", "dense"]

# ---- 10-seed 实测数据 ----
A_SP = {
    "MDDE": [1.75, 1.04, 1.02, 1.73, 3.79, 10.56],
    "SCA-L1": [15.36, 12.99, 10.98, 4.68, 1.46, 10.08],
    "SCA-SP": [15.13, 10.74, 5.68, 3.56, 1.30, 9.73],
    "PF": [5.07, 8.52, 10.12, 10.26, 16.96, 36.65],
    "SL0": [8.67, 6.25, 1.46, 4.51, 7.52, 15.58],
    "DUET": [27.05, 25.57, 24.53, 25.89, 29.41, 31.73],
}
SDR_SP = {
    "MDDE": [14.21, 10.82, 7.80, 4.39, 0.57, -3.55],
    "SCA-L1": [3.17, 3.52, 2.91, 3.12, 0.92, -3.58],
    "SCA-SP": [7.10, 6.88, 5.12, 2.42, -0.83, -5.36],
    "PF": [10.57, 5.45, 2.48, 0.01, -5.72, -15.79],
    "SL0": [9.09, 7.24, 7.87, 3.35, -0.34, -4.09],
    "DUET": [-1.17, -0.70, -0.29, -2.96, -4.01, -5.16],
}
SNR_VS = {
    "MDDE": [8.18, 2.91, 0.87, 0.63, 2.06],
    "SCA-L1": [11.69, 8.62, 7.91, 6.08, 6.94],
    "SL0": [10.64, 4.80, 4.13, 4.01, 2.95],
}
N_VS = {
    "MDDE": [0.91, 0.91, 1.07, 4.61],
    "SCA-L1": [9.66, 3.60, 2.91, 4.77],
    "SL0": [9.14, 1.54, 3.56, 5.09],
}


def fig_sparsity(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    xs = np.arange(len(CFG))
    for ax, data, ylab in [(axes[0], A_SP, "Mixing-matrix angle error (deg)"),
                           (axes[1], SDR_SP, "SDR (dB)")]:
        for m, vals in data.items():
            st = STYLE.get(m, dict(color="gray"))
            ax.plot(xs, vals, label=m, **st)
        ax.set_xticks(xs); ax.set_xticklabels(CFG)
        ax.set_xlabel("TF activation probability p (denser to the right)")
        ax.set_ylabel(ylab)
    axes[0].set_title("(a) Angle error (lower is better)")
    axes[1].set_title("(b) SDR (higher is better)")
    axes[0].legend(loc="upper left", framealpha=0.9, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_sparsity.{ext}"))
    plt.close(fig)


def fig_snr(out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    snrs = [0, 10, 20, 30, 40]
    for m, vals in SNR_VS.items():
        st = STYLE.get(m, dict(color="gray"))
        ax.plot(snrs, vals, label=m, **st)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Mixing-matrix angle error (deg)")
    ax.set_title("Angle error vs SNR (p=0.05)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_snr.{ext}"))
    plt.close(fig)


def fig_nsources(out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ns = [3, 4, 5, 6]
    for m, vals in N_VS.items():
        st = STYLE.get(m, dict(color="gray"))
        ax.plot(ns, vals, label=m, **st)
    ax.set_xticks(ns)
    ax.set_xlabel("Number of sources N"); ax.set_ylabel("Mixing-matrix angle error (deg)")
    ax.set_title("Angle error vs source count (p=0.10)")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_nsources.{ext}"))
    plt.close(fig)


def fig_ablation(out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    labels = ["Full", "w/o single-\nsource gate", "w/o energy\ngate", "w/o balance\ngate"]
    vals = [3.31, 17.83, 8.75, 3.79]
    colors = ["#C0392B", "#7F8C8D", "#7F8C8D", "#7F8C8D"]
    ax.bar(range(len(vals)), vals, color=colors, alpha=0.85, edgecolor="black", lw=0.4)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Mean angle error (deg)")
    ax.set_title("Ablation of the three gates (10 seeds)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_ablation.{ext}"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/figures_v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_sparsity(args.out); print("fig_sparsity 完成")
    fig_snr(args.out); print("fig_snr 完成")
    fig_nsources(args.out); print("fig_nsources 完成")
    fig_ablation(args.out); print("fig_ablation 完成")
    print("输出目录:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
