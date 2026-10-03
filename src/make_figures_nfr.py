"""NF-SSP 论文图表（英文标签、矢量 PDF + PNG）。

输出
----
fig_scale.pdf        参考尺度 r(p)：实测 vs 闭式 + 对数轴（Fig. 1）
fig_architecture.pdf 方法框图（Fig. 2）
fig_sparsity.pdf     A 误差 / SDR vs 稀疏度（Fig. 3）
fig_snr.pdf          A 误差 vs SNR（Fig. 4）
fig_nsources.pdf     A 误差 vs 源数目（Fig. 5）
fig_guard.pdf        保留诊断与决策（Fig. 6）
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import data as D
from nfr import mixture_median_ratio, point_energies

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.linewidth": 0.6,
    "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False,
})

C_BLUE, C_RED, C_GREEN, C_ORANGE, C_GREY, C_PURPLE = (
    "#3266ad", "#c0392b", "#1a8a5a", "#d68910", "#6b7280", "#7d3c98")

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
XL = ["0.02", "0.05", "0.10", "0.20", "0.40", "dense"]
P_OF = {"tf_p02": 0.02, "tf_p05": 0.05, "tf_p10": 0.10, "tf_p20": 0.20, "tf_p40": 0.40}
M_LABEL = {
    "NF-SSP": ("NF-SSP (proposed)", C_RED, "-", "o"),
    "SCA-default": ("SCA-default (0.02$\\times$median)", C_GREY, "--", "s"),
    "SCA-fixed-te5": ("SCA fixed $t_e$=5", C_ORANGE, "-.", "^"),
    "SCA-L1-tuned": ("SCA-tuned (per-regime oracle)", C_BLUE, ":", "D"),
    "oracle_a_l1": ("Oracle-A", C_GREEN, "-", "*"),
    "ssp_kmeans_l1": ("SCA-$\\ell_1$", C_BLUE, ":", "D"),
    "pf_auto_l1": ("Potential function", C_PURPLE, "-.", "v"),
    "sl0_l1": ("Smoothed $\\ell_0$", C_ORANGE, "--", "P"),
    "duet": ("DUET", C_GREY, ":", "x"),
}


def load(path):
    with open(path) as f:
        return json.load(f)


def recs_of(d, exp, method=None, cfg=None, **filt):
    out = []
    for r in d["records"]:
        if r["experiment"] != exp:
            continue
        if method is not None and r["method"] != method:
            continue
        if cfg is not None and r["cfg_key"] != cfg:
            continue
        ok = True
        for k, v in filt.items():
            rv = r.get(k)
            if isinstance(v, float):
                if rv is None or abs(rv - v) > 1e-9:
                    ok = False; break
            elif rv != v:
                ok = False; break
        if ok:
            out.append(r)
    return out


def mean(v, metric="A_angle_deg"):
    a = [r[metric] for r in v if r[metric] is not None and np.isfinite(r[metric])]
    return float(np.mean(a)) if a else float("nan")


# ==========================================================================
# Fig. 1  参考尺度
# ==========================================================================

def fig_scale(out):
    keys = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
    meas, cf_, pi0 = [], [], []
    for k in keys:
        r_m, r_c = [], []
        for s in range(5):
            p = D.make_problem(2, 4, 33, 64, k, 20.0, seed=1000 + s)
            s2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
            r_m.append(float(np.median(point_energies(p["X_tf"]))) / (2 * s2))
            pv = P_OF.get(k, p["active_ratio"])
            r_c.append(mixture_median_ratio(pv, 4, 2, 20.0)["ratio"])
        meas.append(float(np.mean(r_m)))
        cf_.append(float(np.mean(r_c)))
        pv = P_OF.get(k, 1.0)
        pi0.append((1 - pv) ** 4)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(len(keys))
    ax = axes[0]
    ax.plot(x, meas, "-o", color=C_RED, ms=4, lw=1.4, label="measured")
    cfin = [v for v in cf_ if np.isfinite(v)]
    ax.plot(x[:len(cfin)], cfin, "--", color=C_BLUE, lw=1.2,
            label="semi-analytic form (Prop. 6)")
    ax.axvline(2.5, color=C_GREY, lw=0.8, ls=":")
    ax.text(2.55, 40, "knee $p^\\star=0.159$\n($\\pi_0=1/2$)", fontsize=6.4,
            color=C_GREY, va="center")
    ax.set_xlabel("activation probability $p$")
    ax.set_ylabel("$r(p)=\\mathrm{median}(e)/\\nu$")
    ax.set_xticks(x); ax.set_xticklabels(XL)
    ax.set_title("(a) reference-scale ratio")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    ax.semilogy(x, meas, "-o", color=C_RED, ms=4, lw=1.4)
    for xi, v in zip(x, meas):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.2, color=C_RED)
    ax.axhline(1.0, color=C_GREY, lw=0.8, ls="--")
    ax.text(0.05, 1.15, "$r=1$ (median = noise floor)", fontsize=6.2, color=C_GREY)
    ax.set_xlabel("activation probability $p$")
    ax.set_ylabel("$r(p)$ (log scale)")
    ax.set_xticks(x); ax.set_xticklabels(XL)
    ax.set_title("(b) drift across the sparsity range")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_scale.pdf"))
    fig.savefig(os.path.join(out, "fig_scale.png"))
    plt.close(fig)
    print("  fig_scale", [round(v, 2) for v in meas], [round(v, 2) for v in cf_])


# ==========================================================================
# Fig. 2  方法框图
# ==========================================================================

def _box(ax, xy, w, h, text, fc, ec, fs=6.6):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color="#444441"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=color,
                                 shrinkA=0, shrinkB=0))


def _check_no_overlap(boxes, arrows, name, pad=0.012, tol=0.006):
    """几何自检：箭头不得穿过方框内部。

    「箭头穿框」是方框图最典型的**静默**缺陷——渲染不报错、页数正常，只有肉眼能发现。
    Fig. 2 实际发生过：一条竖箭头的起止 y 恰好取成了红框的上下边，整条穿框而过。
    所有箭头都以方框边缘为端点，故只采样线段**内部**（t∈[0.125,0.875]）并把每个方框
    按 FancyBboxPatch 的 pad 外扩、再留 tol 容差，端点落在边上不会被误报。
    """
    hits = []
    for seg in arrows:
        x0, y0, x1, y1 = seg
        for i in range(5, 36):
            t = i / 40
            px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            box = next(((x, y, w, h) for (x, y, w, h) in boxes
                        if x + pad + tol < px < x + w - pad - tol
                        and y + pad + tol < py < y + h + pad - tol), None)
            if box is not None:
                hits.append((seg, box))
                break
    if hits:
        for s, b in hits:
            print(f"  ✗ {name}：箭头 {s} 穿过方框 {b}")
        raise SystemExit(f"{name}：{len(hits)} 条箭头穿框，请修正坐标")


def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(7.0, 2.45))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis("off")
    yc = 1.62
    boxes = [
        ((0.05, yc - 0.28), 1.15, 0.56,
         "observation\n$\\mathbf{X}_{\\rm tf}$", "#eef2f7", C_BLUE),
        ((1.42, yc + 0.06), 1.5, 0.5,
         "per-point energy\n$e=\\|\\mathbf{x}(f,t)\\|^2$", "#eef2f7", C_BLUE),
        ((1.42, yc - 0.66), 1.5, 0.52,
         "collinearity + balance\n$|\\cos|>c_0$, both parts", "#f7f2ee", C_ORANGE),
        ((3.14, yc + 0.06), 1.72, 0.5,
         "blind noise floor\n$\\hat s^2=2e_{(q)}/Q_q(\\chi^2_{2M})$", "#fdeeec", C_RED),
        ((3.14, yc - 0.66), 1.72, 0.52,
         "calibrated multiple\n$\\tau=Q_{1-\\alpha}(\\chi^2_{2M})/2M$", "#fdeeec", C_RED),
        ((5.08, yc - 0.28), 1.35, 0.56,
         "retention\nself-check", "#f3eefa", C_PURPLE),
        ((6.65, yc + 0.06), 1.5, 0.5,
         "energy gate\n$e>\\tau\\hat\\nu$", "#eaf6f0", C_GREEN),
        ((6.65, yc - 0.66), 1.5, 0.52,
         "fallback:\nclassical gate", "#f4f4f2", C_GREY),
        ((8.45, yc - 0.28), 1.5, 0.56,
         "spherical $k$-means\n$+$ debiased $\\ell_1$", "#eef2f7", C_BLUE),
    ]
    for xy, w, h, txt, fc, ec in boxes:
        _box(ax, xy, w, h, txt, fc, ec)

    # 原有一条竖箭头 (4.00, yc-0.14) -> (4.00, yc-0.66) 已删除：
    # 其起止 y 恰为 "calibrated multiple" 红框的上下边（0.96 / 1.48），整条穿过框体，
    # 把框内标题划断、箭头尖戳出下边框；且 τ=Q_{1-α}(χ²_{2M})/2M 只依赖 M 与 α，
    # 与噪声底 ν̂ 无关，该箭头本身也不表达任何正确的依赖关系。
    arrows = [
        (1.20, yc, 1.42, yc + 0.31),
        (1.20, yc, 1.42, yc - 0.40),
        (2.92, yc + 0.31, 3.14, yc + 0.31),
        (4.86, yc + 0.31, 5.08, yc + 0.10),
        (4.86, yc - 0.40, 5.08, yc - 0.10),
        (6.43, yc + 0.10, 6.65, yc + 0.31),
        (6.43, yc - 0.10, 6.65, yc - 0.40),
        (8.15, yc + 0.31, 8.45, yc + 0.05),
        (8.15, yc - 0.40, 8.45, yc - 0.05),
    ]
    _check_no_overlap([(xy[0], xy[1], w, h) for xy, w, h, *_ in boxes], arrows, "Fig. 2")
    for a in arrows:
        _arrow(ax, *a)

    ax.text(4.00, yc - 1.02, "premise fails: disable gate",
            fontsize=6.0, color=C_GREY, ha="center")
    # 措辞须与 Table 5 一致：六个量是"固定设置"，不是一个都不需要设定。
    ax.text(0.05, 0.16, "no training; six fixed settings; no per-condition tuning; runtime $\\approx$ conventional pipeline",
            fontsize=6.2, color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_architecture.pdf"))
    fig.savefig(os.path.join(out, "fig_architecture.png"))
    plt.close(fig)
    print("  fig_architecture")


# ==========================================================================
# Fig. 3  稀疏度阶梯
# ==========================================================================

def fig_sparsity(d, out):
    x = np.arange(len(LADDER))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, metric, ylab, title in [
            (axes[0], "A_angle_deg", "Mixing-matrix angle error (deg)", "(a) mixing-matrix accuracy"),
            (axes[1], "SDR", "SDR (dB)", "(b) end-to-end source quality")]:
        for m in ["NF-SSP", "SCA-default", "SCA-fixed-te5", "SCA-L1-tuned", "oracle_a_l1"]:
            v = [mean(recs_of(d, "E1_sparsity", m, k), metric) for k in LADDER]
            lab, col, ls, mk = M_LABEL[m]
            if metric == "SDR" and m == "oracle_a_l1":
                lab = "Oracle-A"
            ax.plot(x, v, ls, color=col, marker=mk, ms=3.6, lw=1.2, label=lab)
        ax.set_xticks(x); ax.set_xticklabels(XL)
        ax.set_xlabel("TF activation probability $p$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if metric == "A_angle_deg":
            ax.set_yscale("log"); ax.set_ylim(0.2, 40)
            ax.legend(loc="upper left", frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_sparsity.pdf"))
    fig.savefig(os.path.join(out, "fig_sparsity.png"))
    plt.close(fig)
    print("  fig_sparsity")


# ==========================================================================
# Fig. 4  SNR
# ==========================================================================

def fig_snr(d, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    snrs = [0.0, 10.0, 20.0, 30.0, 40.0]
    for key, ls in [("tf_p05", "-"), ("tf_p20", "--")]:
        for m in ["NF-SSP", "SCA-default"]:
            v = [mean(recs_of(d, "E2_snr", m, key, snr_db=s), "A_angle_deg") for s in snrs]
            lab, col, _, mk = M_LABEL[m]
            tag = "NF-SSP" if m == "NF-SSP" else "SCA-default"
            ax.plot(snrs, v, ls, color=col, marker=mk, ms=3.4, lw=1.2,
                    label=f"{tag}, $p$={'0.05' if key=='tf_p05' else '0.20'}")
        v = [mean(recs_of(d, "E2_snr", "SCA-L1-tuned", key, snr_db=s), "A_angle_deg")
             for s in snrs]
        ax.plot(snrs, v, ":", color=C_BLUE, marker="D", ms=3.0, lw=1.0,
                label=f"oracle, $p$={'0.05' if key=='tf_p05' else '0.20'}")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mixing-matrix angle error (deg)")
    ax.legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_snr.pdf"))
    fig.savefig(os.path.join(out, "fig_snr.png"))
    plt.close(fig)
    print("  fig_snr")


# ==========================================================================
# Fig. 5  源数目
# ==========================================================================

def fig_nsources(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    Ns = [3, 4, 5, 6]
    for ax, key, ttl in [(axes[0], "tf_p05", "(a) $p=0.05$"),
                         (axes[1], "tf_p20", "(b) $p=0.20$")]:
        for m in ["NF-SSP", "SCA-default", "SCA-fixed-te5"]:
            v = [mean(recs_of(d, "E4_N", m, key, n_true=N), "A_angle_deg") for N in Ns]
            lab, col, ls, mk = M_LABEL[m]
            ax.plot(Ns, v, ls, color=col, marker=mk, ms=3.6, lw=1.2, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("true number of sources $N$")
        ax.set_title(ttl)
        ax.set_xticks(Ns)
    axes[0].set_ylabel("Mixing-matrix angle error (deg)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_nsources.pdf"))
    fig.savefig(os.path.join(out, "fig_nsources.png"))
    plt.close(fig)
    print("  fig_nsources")


# ==========================================================================
# Fig. 6  保留诊断
# ==========================================================================

def fig_guard(out):
    # 条件名、保留比例、门是否装配、门开/门关的 A 误差
    rows = [
        ("$p$=0.02", 0.350, True, 0.33, 16.53),
        ("$p$=0.05", 0.571, True, 0.34, 14.10),
        ("$p$=0.10", 0.719, True, 0.30, 12.30),
        ("$p$=0.20", 0.834, True, 0.59, 4.94),
        ("$p$=0.40", 0.865, True, 1.28, 5.17),
        ("$p$=0.05,\n0 dB", 0.140, True, 4.98, 13.44),
        ("$p$=0.20,\n0 dB", 0.034, False, 14.32, 15.25),
        ("chirp", 0.061, True, 14.58, 18.28),
        ("impulse", 0.150, True, 6.59, 10.91),
        ("dense,\n20 dB", 0.022, False, 16.27, 10.08),
        ("dense,\n0 dB", 0.001, False, 26.98, 13.41),
    ]
    lab = [r[0] for r in rows]
    keep = np.array([r[1] for r in rows]) * 100
    on = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(rows))
    cols = [C_GREEN if o else C_RED for o in on]
    ax.bar(x, keep, color=cols, width=0.62)
    ax.axhline(3.5, color=C_PURPLE, lw=1.1, ls="--")
    ax.text(len(rows) - 0.4, 4.6, "decision boundary 3.5%", fontsize=6.4,
            color=C_PURPLE, ha="right")
    ax.set_yscale("log"); ax.set_ylim(0.5, 200)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.2)
    ax.set_ylabel("candidates surviving the gate (%)")
    ax.set_title("retention diagnostic: green = gate active, red = gate disabled")
    for xi, (v, o) in enumerate(zip(keep, on)):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=5.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_guard.pdf"))
    fig.savefig(os.path.join(out, "fig_guard.png"))
    plt.close(fig)
    print("  fig_guard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/nfr_results.json")
    ap.add_argument("--out", default="../results/figures_nfr")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("生成图表 ...")
    fig_scale(a.out)
    fig_architecture(a.out)
    fig_guard(a.out)
    if os.path.exists(a.inp):
        d = load(a.inp)
        fig_sparsity(d, a.out)
        fig_snr(d, a.out)
        fig_nsources(d, a.out)
    else:
        print(f"  (跳过性能图：{a.inp} 不存在)")
    print("完成 →", a.out)


if __name__ == "__main__":
    main()
