"""R2-4  p × SNR 二维网格（审稿意见「必做实验 4」）。

目前稀疏度与 SNR 是分开扫描的，无法看出二者的交互。
本脚本在 p ∈ {0.01,0.02,0.05,0.10,0.20,0.40} × SNR ∈ {0,5,10,20,30} dB
上跑满，输出三种能量门限的角度误差，并画热图。

用法：
    python3 r2_heatmap.py --seeds 5 --workers 10 --out ../results/r2_heatmap_p_snr.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import data as D
import metrics as MT
from nfr import nfr_mask, point_energies
from baselines import kmeans_sphere, ssp_directions

M_OBS, N_SRC, F, T = 2, 4, 33, 64
P_GRID = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
SNR_GRID = [0.0, 5.0, 10.0, 20.0, 30.0]


def base_mask(X: np.ndarray) -> np.ndarray:
    Xr, Xi = X.real, X.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    eps = 1e-15
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
    d = np.maximum(nr + ni, eps)
    return (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)


def job(j: tuple[float, float, int]) -> dict:
    pratio, snr, s = j
    rng = np.random.default_rng(9000 + s)
    A = D.gen_mixing_matrix(M_OBS, N_SRC, rng, 12.0)
    S = D.gen_tf_sources(N_SRC, F, T, "tf_sparse", pratio, rng)
    Xc = np.einsum("mn,nft->mft", A, S)
    power = float(np.mean(np.abs(Xc) ** 2)) / (10.0 ** (snr / 10.0))
    noise = (rng.standard_normal(Xc.shape) + 1j * rng.standard_normal(Xc.shape))
    noise *= np.sqrt(power / 2.0)
    X = Xc + noise

    E = point_energies(X)
    e2 = np.sum(np.abs(X) ** 2, axis=0)
    b = base_mask(X)
    med = float(np.median(E))
    m_nf, dg = nfr_mask(X, M_OBS, alpha=1e-4, use_self_check=True)
    m_cl = b & (e2 > 0.02 * med)
    m_t5 = b & (e2 > 5.0 * med)

    def err(mk):
        U = ssp_directions(X, mk)
        if U.shape[1] < N_SRC:
            return float("nan")
        C = kmeans_sphere(U, N_SRC, np.random.default_rng(s))
        return float(MT.mixing_matrix_angle_error_deg(A, C))

    return dict(p=pratio, snr_db=snr, seed=s, A_nf=err(m_nf),
                A_classical=err(m_cl), A_te5=err(m_t5),
                gate_on=bool(dg["gate_on"]),
                r_measured=float(med / (M_OBS * float(np.mean(np.abs(noise) ** 2)))),
                keep_frac=float(dg["keep_frac"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="../results/r2_heatmap_p_snr.json")
    ap.add_argument("--figdir", default="../results/figures_nfr")
    a = ap.parse_args()

    jobs = [(p, s, k) for p in P_GRID for s in SNR_GRID for k in range(a.seeds)]
    t0 = time.perf_counter()
    if a.workers <= 1:
        recs = [job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            recs = list(ex.map(job, jobs))
    print(f"\n=== p × SNR 网格（{len(P_GRID)}×{len(SNR_GRID)}×{a.seeds}，"
          f"{time.perf_counter()-t0:.0f}s）===")

    METH = [("A_nf", "NF gate"), ("A_classical", "classical $t_e{=}0.02$"),
            ("A_te5", "median $t_e{=}5$")]
    grids = {}
    for key, name in METH:
        print(f"\n--- {name} 角度误差（°）---")
        print(f"{'p \\ SNR':>9}" + "".join(f"{s:>9.0f}" for s in SNR_GRID))
        g = np.zeros((len(P_GRID), len(SNR_GRID)))
        for i, p in enumerate(P_GRID):
            for jj, s in enumerate(SNR_GRID):
                v = [r[key] for r in recs if r["p"] == p and r["snr_db"] == s]
                g[i, jj] = np.nanmean(v)
            print(f"{p:>9.2f}" + "".join(f"{g[i, jj]:>9.2f}" for jj in range(len(SNR_GRID))))
        grids[key] = g.tolist()

    print("\n--- 门限开启率 ---")
    print(f"{'p \\ SNR':>9}" + "".join(f"{s:>9.0f}" for s in SNR_GRID))
    gr = np.zeros((len(P_GRID), len(SNR_GRID)))
    for i, p in enumerate(P_GRID):
        for jj, s in enumerate(SNR_GRID):
            v = [r["gate_on"] for r in recs if r["p"] == p and r["snr_db"] == s]
            gr[i, jj] = float(np.mean(v))
        print(f"{p:>9.2f}" + "".join(f"{gr[i, jj]:>9.2f}" for jj in range(len(SNR_GRID))))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": vars(a), "p_grid": P_GRID, "snr_grid": SNR_GRID,
                   "grids": grids, "gate_rate": gr.tolist(), "records": recs},
                  open(a.out, "w"))
        print(f"\n已写 {a.out}")

    # ---- 图 ----
    if a.figdir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # 按**最终显示尺寸**出图（正文宽 16cm = 6.3in），避免缩放后标注不可读
        fig, axes = plt.subplots(1, 3, figsize=(6.45, 2.35))
        allv = np.array([grids[k] for k, _ in METH])
        vmin, vmax = max(np.nanmin(allv), 1e-2), np.nanmax(allv)
        for ax, (key, name) in zip(axes, METH):
            g = np.array(grids[key])
            im = ax.imshow(g, aspect="auto", origin="lower", cmap="viridis",
                           norm=LogNorm(vmin=vmin, vmax=vmax))
            ax.set_xticks(range(len(SNR_GRID)))
            ax.set_xticklabels([f"{s:.0f}" for s in SNR_GRID], fontsize=5)
            ax.set_yticks(range(len(P_GRID)))
            ax.set_yticklabels([f"{p:.2f}" for p in P_GRID], fontsize=5)
            ax.set_xlabel("SNR (dB)", fontsize=5.5)
            if ax is axes[0]:
                ax.set_ylabel("activation probability $p$", fontsize=5.5)
            ax.set_title(name, fontsize=5.5)
            for i in range(len(P_GRID)):
                for jj in range(len(SNR_GRID)):
                    v = g[i, jj]
                    ax.text(jj, i, f"{v:.1f}", ha="center", va="center", fontsize=4.0,
                            color="w" if v < np.sqrt(vmin * vmax) else "k")
        cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.015)
        cb.set_label("angle error (deg)", fontsize=5.5)
        cb.ax.tick_params(labelsize=5)
        fig.subplots_adjust(left=0.090, right=0.885, top=0.89, bottom=0.22, wspace=0.24)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.figdir, f"fig_heatmap.{ext}"))
        plt.close(fig)
        print(f"已画 {a.figdir}/fig_heatmap.pdf")


if __name__ == "__main__":
    main()
