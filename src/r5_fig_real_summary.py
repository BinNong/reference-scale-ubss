"""R5-8  W5：真实语音核心结果的紧凑提炼图（Fig. 11）。

审稿人指出 Table 15/17 信息密度过大、Fig. 7/8 的逐配置对比偏拥挤，建议对真实语音部分的
核心结果做一次更直观的可视化提炼。本脚本出两面板：

  (a) 逐配置的**配对对数比**  ln(A_对照 / A_NFSSP)
      条 > 0 表示本文门限更好。两支：对照 = 惯例中位数门限 / 最大值门限。
      排序后一眼可见"对前者 14/15 更好、对后者 12/15 更差"，且**幅度不对称**。
  (b) 三个方法的均值（对数横轴）+ 关键比值标注，替代通读 Table 15 的均值行。

**黑白可读性（第七轮）**：审稿人要求图例/线型在黑白打印下仍可区分。原图两个面板都**只靠颜色**
编码——(a) 的两支在转灰度后明度接近，(b) 的 derived(0.70°) 与 per-config bound(1.12°) 也接近。
现给每个条目加**填充纹样**，按"角色"跨面板统一：

    惯例中位数 solid · max-referenced '////' · derived (NF-SSP) 'xxxx' · per-config bound 反斜线纹

纹样即 legend 手柄上的纹样，故黑白环境下无需依赖颜色。**改纹样不改任何数字**；重跑后应只有
fig_real_summary.{pdf,png} 变化（本脚本 main 也只写这两个文件）。

用法：
    python3 r5_fig_real_summary.py --results ../results --out ../results/figures_real
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

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "figure.dpi": 150,
})

SHORT = {
    "R1_res:w256": "win 256", "R1_res:w512": "win 512", "R1_res:w1024": "win 1024",
    "R1_res:w2048": "win 2048", "R2_n:N3": "$N=3$", "R2_n:N5": "$N=5$",
    "R2_n:N6": "$N=6$", "R3_snr:snr00": "SNR 0 dB", "R3_snr:snr10": "SNR 10 dB",
    "R3_snr:snr30": "SNR 30 dB", "R3_snr:snr40": "SNR 40 dB",
    "R4_dense:d1": "1 dense", "R4_dense:d2": "2 dense",
    "R5_noise:babble": "babble", "R5_noise:clean": "noise-free",
}


HATCH = {                       # 跨面板统一的"角色→纹样"映射（黑白可读，见文件头说明）
    "SCA-median": "",           # 惯例中位数：实心
    "SCA-max": "////",          # max-referenced
    "NF-SSP": "xxxx",           # derived（本文门限）
    "bound": "\\\\",            # per-configuration bound
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--out", default="../results/figures_real")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    R = json.load(open(os.path.join(a.results, "real_results.json")))["records"]
    acc = defaultdict(lambda: defaultdict(list))
    for r in R:
        acc[r["case"]][r["method"]].append(r["A_angle_deg"])
    cases = sorted(acc, key=lambda c: list(SHORT).index(c) if c in SHORT else 99)
    mean = {c: {m: float(np.mean(v)) for m, v in acc[c].items()} for c in cases}

    nf = np.array([mean[c]["NF-SSP"] for c in cases])
    cm = np.array([mean[c]["SCA-median"] for c in cases])
    cx = np.array([mean[c]["SCA-max"] for c in cases])
    lab = [SHORT.get(c, c) for c in cases]
    r1, r2 = np.log(cm / nf), np.log(cx / nf)
    order = np.argsort(r1)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(6.45, 3.30),
                             gridspec_kw=dict(width_ratios=[1.75, 1.0], wspace=0.30))

    # ---------- (a) 配对对数比 ----------
    ax = axes[0]
    y = np.arange(len(cases))
    ax.barh(y + 0.19, r1[order], height=0.34, color="#1f5fa9",
            hatch=HATCH["SCA-median"], edgecolor="#123c69", linewidth=0.5,
            label="vs conventional median ($t_e=0.02$)")
    ax.barh(y - 0.19, r2[order], height=0.34, color="#c0504d",
            hatch=HATCH["SCA-max"], edgecolor="#7a2f2d", linewidth=0.5,
            label="vs max-referenced ($c=0.05$)")
    ax.set_yticks(y)
    ax.set_yticklabels([lab[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$\ln\,(A_{\rm competitor}/A_{\rm NF\text{-}SSP})$"
                  "   (right: derived threshold better)")
    ax.set_title("(a) Paired comparison, 15 real-speech configurations",
                 loc="left", y=1.16)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
              frameon=False, handlelength=1.4, columnspacing=1.2, borderpad=0.0)
    n_win1 = int(np.sum(r1 > 0))
    n_win2 = int(np.sum(r2 > 0))
    ax.set_xlim(-1.45, 3.75)
    ax.text(0.985, 0.035, f"better in {n_win1}/15\nworse in {15-n_win2}/15",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#333333", linespacing=1.4)

    # ---------- (b) 均值对比 ----------
    ax = axes[1]
    names = ["conventional\nmedian", "max-referenced\n(one coefficient)", "derived\n(NF-SSP)",
             "per-config\nbound"]
    vals = [cm.mean(), cx.mean(), nf.mean(),
            np.mean([min(mean[c][m] for m in mean[c] if m.endswith("-opt")) for c in cases])]
    cols = ["#c0504d", "#d99694", "#1f5fa9", "#7f7f7f"]
    edg = ["#7a2f2d", "#7a4a48", "#123c69", "#3f3f3f"]
    hat = [HATCH["SCA-median"], HATCH["SCA-max"], HATCH["NF-SSP"], HATCH["bound"]]
    ax.barh(np.arange(4), vals, color=cols, height=0.62,
            edgecolor=edg, hatch=hat, linewidth=0.5)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("mean angle error (deg)")
    for i, v in enumerate(vals):
        ax.text(v * 1.12, i, f"{v:.2f}°", va="center", fontsize=7.5)
    ax.set_xlim(0.45, 60)
    ax.set_title("(b) Means on the same runs", loc="left", y=1.16)
    fig.subplots_adjust(left=0.150, right=0.985, top=0.84, bottom=0.28, wspace=0.42)
    fig.text(0.60, 0.045,
             f"derived vs conventional: {cm.mean()/nf.mean():.1f}$\\times$ lower\n"
             f"derived vs max-ref: {nf.mean()/cx.mean():.2f}$\\times$ higher, "
             f"and better in only {int(np.sum(cx > nf))}/15",
             ha="center", va="bottom", fontsize=7, linespacing=1.5, color="#333333")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_real_summary.{ext}"))
    plt.close(fig)
    print(f"  fig_real_summary  ->  {a.out}")
    print(f"  NF 优于惯例中位数 {n_win1}/15；优于 max-referenced {n_win2}/15")
    print(f"  均值：惯例 {cm.mean():.3f}  NF {nf.mean():.3f}  max {cx.mean():.3f}")


if __name__ == "__main__":
    main()
