"""生成理论验证图（论文版式，英文标签，矢量 + 位图输出）。

三个子图：
    (a) 纯噪声点通过硬阈值的概率：蒙特卡洛   vs   Corollary 1 闭式
    (b) 通过点中的噪声占比：硬阈值 vs 能量门限（理论曲线 + 实测点）
    (c) Remark 2 中拟合常数 c 随稀疏度的变化（检验其是否 O(1)）

依赖 theory_validation.json（由 theory_validate_data.py 生成）。

用法（项目 src 目录）:
    ../.venv/bin/python make_theory_fig.py --in ../results/theory_validation.json \
        --out ../results/figures_v2
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
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "legend.fontsize": 5.8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.45,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "figure.dpi": 150,
})

C_BLUE = "#1f5fa8"
C_RED = "#c0392b"
C_GREEN = "#1d9e75"
C_GRAY = "#7a7a7a"


def panel_a(ax) -> None:
    rng = np.random.default_rng(0)
    n = 400_000
    Xr = rng.standard_normal((2, n))
    Xi = rng.standard_normal((2, n))
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    ac = np.abs((Xr * Xi).sum(0) / (nr * ni))
    xs = np.sort(ac)
    surv = 1.0 - np.arange(1, n + 1) / n
    ax.plot(xs, surv, color=C_BLUE, lw=1.3,
            label="noise-only, Monte Carlo ($M=2$)")

    c = np.linspace(0.0, 0.9999, 600)
    ax.plot(c, 1.0 - (2.0 / np.pi) * np.arcsin(c), "k--", lw=1.0,
            label=r"$1-\frac{2}{\pi}\arcsin c_0$  (Cor. 1, $M=2$)")
    ax.plot(c, 1.0 - c, color=C_GRAY, ls=":", lw=1.0,
            label=r"$1-c_0$  (Cor. 1, $M=3$)")

    ax.axvline(0.98, color=C_RED, lw=0.9)
    ax.plot([0.98], [0.1275], "o", color=C_RED, ms=3.4, zorder=5)
    ax.annotate("12.75%", xy=(0.98, 0.1275), xytext=(0.60, 0.34),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_RED),
                color=C_RED, fontsize=6.5)
    ax.set_xlabel(r"threshold $c_0$")
    ax.set_ylabel(r"$P(|\cos|>c_0)$")
    ax.set_title("(a) noise survives the hard test")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.96, handlelength=1.6)


def panel_b(ax, data) -> None:
    lad = data["ladder"]
    ps = np.array([r["p"] for r in lad], dtype=float)
    kth = np.array([r["kappa_th"] for r in lad], dtype=float)
    kme = np.array([r["kappa_meas"] for r in lad], dtype=float)
    gen = np.array([r["gen_n"] for r in lad], dtype=float)

    share = lambda k: 100.0 * k / (1.0 + k)
    ax.plot(ps, share(kth), "k--", lw=1.1, label="hard threshold (Prop. 4)")
    ax.plot(ps, share(kth * gen), color=C_GREEN, lw=1.4, marker="s", ms=3.0,
            label=r"with energy gate ($g_{\mathrm{en}}$ from Cor. 2)")
    ax.plot(ps, share(kme), "o", color=C_RED, ms=4.0, mfc="none", mew=1.2,
            label="hard threshold, measured")
    ax.set_xlabel(r"TF activation probability $p$")
    ax.set_ylabel("noise share of admitted points (%)")
    ax.set_title("(b) contamination and its suppression")
    ax.set_xlim(-0.01, 0.43)
    ax.set_ylim(0, 70)
    ax.legend(loc="upper right", framealpha=0.96, handlelength=1.6)


def panel_c(ax, data) -> None:
    dk = data["davis_kahan"]
    ps = np.array([r["p"] for r in dk], dtype=float)
    cn = np.array([r["c_no_gate"] for r in dk], dtype=float)

    ax.axvspan(0.15, 0.43, color=C_RED, alpha=0.07, zorder=0)
    ax.plot(ps, cn, "o-", color=C_BLUE, ms=3.6, lw=1.2,
            label=r"fitted $c$ in $\sin\angle \lesssim c\sqrt{K_0}/K_s$")
    ax.axhline(1.0, color=C_GRAY, ls="--", lw=0.9, label=r"$c=1$")
    ax.text(0.295, 6.1, "multi-source\ndominated", fontsize=5.8,
            ha="center", va="center", color=C_RED)
    ax.text(0.135, 1.90, r"sparse regimes: $c=0.58$–$1.09$", fontsize=5.9,
            color=C_BLUE, ha="left", va="center")
    ax.set_xlabel(r"TF activation probability $p$")
    ax.set_ylabel(r"fitted constant $c$")
    ax.set_title("(c) the bound holds where it should")
    ax.set_xlim(0.0, 0.44)
    ax.set_ylim(0, 8.2)
    ax.legend(loc="upper left", framealpha=0.96, handlelength=1.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/theory_validation.json")
    ap.add_argument("--out", default="../results/figures_v2")
    a = ap.parse_args()

    with open(a.inp) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    panel_a(axes[0])
    panel_b(axes[1], data)
    panel_c(axes[2], data)
    fig.tight_layout(pad=0.4)

    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_theory.{ext}"), dpi=300,
                    bbox_inches="tight")
    print(f"已写出 {a.out}/fig_theory.pdf 与 .png")


if __name__ == "__main__":
    main()
