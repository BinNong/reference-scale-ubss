"""Fig. 9：1/sqrt(FT) 缩放预言的检验（双对数）。

命题 5 预言角度误差随 TF 点数按 FT^(-1/2) 下降。本图用
`results/scale_invariance.json` 的 18 组 (M,N,p) × 3 个 FT 点检验它：

(a) 双对数图。实线 = 各格的实测 NF-SSP 误差；虚线 = 斜率 -1/2 的参考线。
    颜色区分稀疏度 p。可见：误差确实随 FT 下降，但总体比 -1/2 平缓，
    只有地板项可忽略的格子才贴近 -1/2。
(b) 拟合两项模型 err^2 = a/FT + b 后，"FT 无关地板 b 在 FT 最大处的占比"
    与实测对数斜率的关系。地板占比越高，斜率越接近 0（r = 0.93）。

输出: results/figures_nfr/fig_scaling.pdf / .png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.8,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.linewidth": 0.6,
    "font.family": "sans-serif", "mathtext.fontset": "dejavusans",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE, C_RED, C_ORANGE, C_GREY = "#3266ad", "#c0392b", "#d68910", "#6b7280"
P_STYLE = {0.05: (C_BLUE, "o"), 0.20: (C_ORANGE, "s"), 0.40: (C_RED, "^")}


def fit_line(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    s2 = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * s2 - sx * sx
    a = (n * sxy - sx * sy) / den
    return a, (sy - a * sx) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="../results/scale_invariance.json")
    ap.add_argument("--outdir", default="../results/figures_nfr")
    a = ap.parse_args()

    d = json.load(open(a.json))
    g: dict = defaultdict(dict)
    for r in d["rows"]:
        g[(r["M"], r["N"], r["p"])][r["F"] * r["T"]] = r["nf"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.75))

    # ---- (a) 双对数 ----
    allx, ally = [], []
    for (M, N, p), v in sorted(g.items()):
        fts = sorted(v)
        col, mk = P_STYLE[p]
        ax1.plot(fts, [v[f] for f in fts], "-", color=col, lw=0.9,
                 marker=mk, ms=3.0, mfc="white", mew=0.8, alpha=0.85)
        allx += fts; ally += [v[f] for f in fts]

    ax1.set_xscale("log"); ax1.set_yscale("log")
    x0 = min(allx)
    y0 = st.mean([y for x, y in zip(allx, ally) if x == x0])
    xr = [x0, max(allx)]
    ax1.plot(xr, [y0 * (x / x0) ** -0.5 for x in xr], "--",
             color=C_GREY, lw=1.1, zorder=1)
    ax1.annotate("slope $-1/2$", xy=(xr[1], y0 * (xr[1] / x0) ** -0.5),
                 xytext=(xr[1] * 0.30, y0 * 0.30 ** -0.5 * 0.62),
                 fontsize=7, color=C_GREY)
    ax1.set_xticks([2112, 8320, 33024])
    ax1.set_xticklabels(["2112", "8320", "33024"])
    ax1.set_xlabel("number of TF points $FT$")
    ax1.set_ylabel("angle error ($^\\circ$)")
    ax1.set_title("(a) scaling of the gate across $FT$")
    for p, (col, mk) in P_STYLE.items():
        ax1.plot([], [], "-", color=col, marker=mk, ms=3.0, mfc="white",
                 mew=0.8, label=f"$p={p:.2f}$")
    ax1.legend(frameon=False, handlelength=1.5, loc="lower left")

    # ---- (b) 地板占比 vs 斜率 ----
    fl, sl = [], []
    for (M, N, p), v in sorted(g.items()):
        fts = sorted(v)
        ai, bi = fit_line([1.0 / f for f in fts], [v[f] ** 2 for f in fts])
        ai, bi = max(ai, 0.0), max(bi, 0.0)
        sl.append(fit_line([math.log(f) for f in fts],
                           [math.log(v[f]) for f in fts])[0])
        fl.append(100.0 * bi / (ai / fts[-1] + bi))

    n = len(fl); mf, ms = st.mean(fl), st.mean(sl)
    cov = sum((x - mf) * (y - ms) for x, y in zip(fl, sl)) / n
    sdf = math.sqrt(sum((x - mf) ** 2 for x in fl) / n)
    sds = math.sqrt(sum((y - ms) ** 2 for y in sl) / n)
    r = cov / (sdf * sds)

    for (M, N, p) in sorted(g):
        col, mk = P_STYLE[p]
        i = sorted(g).index((M, N, p))
        ax2.plot(fl[i], sl[i], mk, color=col, ms=4.2, mfc="white", mew=0.9)

    k, b = fit_line(fl, sl)
    xs = [0, 100]
    ax2.plot(xs, [k * x + b for x in xs], "-", color=C_GREY, lw=0.9)
    ax2.axhline(-0.5, ls="--", color=C_GREY, lw=1.0)
    ax2.annotate("predicted $-1/2$", xy=(62, -0.5), xytext=(46, -0.55),
                 fontsize=7, color=C_GREY)
    ax2.set_xlabel("floor share of the squared error (%)")
    ax2.set_ylabel("fitted log-log slope")
    ax2.set_title(f"(b) slower scaling where a floor dominates ($r={r:.2f}$)")
    ax2.set_xlim(-4, 104); ax2.set_ylim(-0.62, 0.03)

    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.outdir, f"fig_scaling.{ext}"))
    print(f"已写出 {a.outdir}/fig_scaling.pdf|.png")
    print(f"  地板占比 vs 斜率 相关系数 r = {r:.3f}")
    print(f"  斜率: 中位 {st.median(sl):.3f} 均值 {ms:.3f} 范围 [{min(sl):.3f}, {max(sl):.3f}]")
    print(f"  地板占比: 中位 {st.median(fl):.1f}% 范围 [{min(fl):.1f}%, {max(fl):.1f}%]")


if __name__ == "__main__":
    main()
