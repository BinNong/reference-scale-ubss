"""生成论文方法架构图（矢量 PDF + PNG）。

用法（服务器 src 目录下）:
    ../.venv/bin/python make_architecture_fig.py --out ../results/figures_v2
"""
from __future__ import annotations
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "savefig.bbox": "tight", "figure.dpi": 130})

EDGE = "#2C3E50"


def box(ax, x, y, w, h, fc, ec=EDGE, lw=1.0, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def txt(ax, x, y, s, fs=9, w="normal", ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, weight=w, zorder=3)


def arrow(ax, x, y1, y2, color=EDGE):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.1, color=color, zorder=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/figures_v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 16); ax.axis("off")

    C_OBS, C_GATE, C_GATEBOX, C_CLUS, C_OUT = "#EAF2F8", "#FFFFFF", "#FDEBD0", "#E8F8F5", "#D6EAF8"

    # (1) observations
    box(ax, 2.6, 14.9, 4.8, 0.85, C_OBS)
    txt(ax, 5, 15.32, r"TF observations  $\mathbf{x}(f,t)=\mathbf{A}\mathbf{s}(f,t)+\mathbf{n}$", 9)
    arrow(ax, 5, 14.9, 14.4)

    # (2) real reformulation
    box(ax, 1.7, 13.4, 6.6, 0.85, C_OBS)
    txt(ax, 5, 13.82, r"Exact real reformulation  $\mathbf{W}=[\,\mathbf{X}_r\ |\ \mathbf{X}_i\,]$", 9)
    arrow(ax, 5, 13.4, 12.9)

    # (3) multi-gate box
    box(ax, 0.7, 8.2, 8.6, 4.5, C_GATEBOX)
    txt(ax, 5, 12.35, "Multi-gate directional weighting  (per TF point $l$)", 9.5, "bold")

    gw, gh = 2.5, 2.0
    for i, (title, form) in enumerate([
        ("Single-source gate", r"$g^{\mathrm{ssp}}_l$" "\n" r"$\sigma\left(\frac{|\cos(\mathbf{X}_r,\mathbf{X}_i)|-c_0}{c_1}\right)$"),
        ("Energy gate",        r"$g^{\mathrm{en}}_l$"  "\n" r"$\sigma\left(\frac{e_l/\bar e-\tau_e}{\sigma_e}\right)$"),
        ("Balance gate",       r"$g^{\mathrm{bal}}_l$" "\n" r"$\frac{\min(\|\mathbf{X}_r\|,\|\mathbf{X}_i\|)}{\min(\cdot)+\gamma m}$"),
    ]):
        x = 1.0 + i * 2.75
        box(ax, x, 9.6, gw, gh, C_GATE, r=0.05)
        txt(ax, x + gw / 2, 11.15, title, 8, "bold")
        txt(ax, x + gw / 2, 10.45, form, 8)

    box(ax, 2.4, 8.45, 5.2, 0.85, "#FCF3CF")
    txt(ax, 5, 8.87, r"$w_l=g^{\mathrm{ssp}}_l\,g^{\mathrm{en}}_l\,g^{\mathrm{bal}}_l$", 10)
    arrow(ax, 5, 8.2, 7.7)

    # (4) greedy init
    box(ax, 1.5, 6.75, 7.0, 0.9, C_CLUS)
    txt(ax, 5, 7.20, r"Greedy density-peak initialisation  $\mathbf{A}^{(0)}$", 9)
    arrow(ax, 5, 6.75, 6.25)

    # (5) k-means
    box(ax, 1.5, 5.3, 7.0, 0.9, C_CLUS)
    txt(ax, 5, 5.75, r"Weighted spherical $k$-means refinement", 9)
    arrow(ax, 5, 5.3, 4.8)

    # (6) A_hat
    box(ax, 2.6, 3.85, 4.8, 0.9, C_OUT)
    txt(ax, 5, 4.30, r"Mixing matrix  $\hat{\mathbf{A}}$", 10)
    arrow(ax, 5, 3.85, 3.35)

    # (7) l1 recovery
    box(ax, 1.5, 2.4, 7.0, 0.9, C_CLUS)
    txt(ax, 5, 2.85, r"Debiased $\ell_1$ recovery  (identical to baselines)", 8.5)
    arrow(ax, 5, 2.4, 1.9)

    # (8) S_hat
    box(ax, 2.6, 0.95, 4.8, 0.9, C_OUT)
    txt(ax, 5, 1.40, r"Sources  $\hat{\mathbf{S}}$", 10)

    # side note
    ax.text(5, 0.35, "Fully differentiable — no training required", ha="center", va="center",
            fontsize=8, style="italic", color="#555")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"fig_architecture.{ext}"))
    plt.close(fig)
    print("fig_architecture 完成 ->", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
