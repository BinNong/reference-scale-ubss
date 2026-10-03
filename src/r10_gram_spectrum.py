r"""第十轮（对外）B1：固定最小夹角下，参考尺度诊断对活跃 Gram 谱结构的敏感性。

回应 CSSP 第二轮意见 MC2：它认可 §4.2 已把 $\\|A_J s_J\\|^2$ 写成
"A_J^H A_J 特征值加权的指数和"，但追问

    "现在 Table 20 证明的是 minimum column separation 的影响比较小，
     但 minimum angle ≠ complete conditioning characterization"

即：**最小列夹角**不是 Gram 谱结构的完整刻画，要求给出 $r(p, \\lambda(A_J^{H}A_J))$。

本脚本在**固定最小列夹角 12°**（§7 的默认）下，直接扫描活跃 Gram 的条件数/谱展宽，
回答两件事：

  (1) 均值-特征值替换（$\\Gamma(J,E_J/J)$ 代替加权指数和）的偏差，
      是否随 Gram 条件数单调变化？
  (2) 该替换对可观测量 $r(p)=\\mathrm{median}(e)/\\nu$ 的影响有多大？

数学（单位列 $\Rightarrow$ $\mathrm{tr}(G)=J$ $\Rightarrow$ $\bar\\lambda=1$）：

    精确      $\\|A_J s_J\\|^2 \\stackrel{d}{=} (E_J/J)\\sum_j \\lambda_j e_j$,  $e_j\\sim\\mathrm{Exp}(1)$
    均值替换  $(E_J/J)\\sum_j \\bar\\lambda\\, e_j = (E_J/J)\\,\\Gamma(J,1)$

两者同均值 $E_J$；偏差完全来自 $\\{\\lambda_j\\}$ 的**展宽**。故本脚本把
$\kappa=\\lambda_{\\max}/\\lambda_{\\min}$ 作为横轴，直接把偏差对 $\\kappa$ 回归。

协议（可复现性）：
  · 混合矩阵由 `data.gen_mixing_matrix` 生成，最小列夹角固定 12°（拒绝采样），
    故"最小夹角"这一变量在本脚本里是**常数**，变化的只有列子集的 Gram。
  · 子集为**随机 J 列子集**（与 §4.2 的口径一致），不是"夹角最小的那一对"。
  · $M\\in\\{2,3\\}$：$M=2$ 时 $J>M$ 的 Gram 秩亏（$\\lambda_{\\min}=0$，$\\kappa=\\infty$），
    单独成一类；$M=3$ 时 $J\\le3$ 全秩，谱展宽连续变化。
  · 单进程、固定种子；无聚类，故不依赖 sklearn，本机可跑。
  · 蒙特卡洛规模写在输出 JSON 的 `meta` 里。

用法：
    /usr/local/bin/python3 src/r10_gram_spectrum.py --out ../results/r10_gram_spectrum.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import data as D                                        # noqa: E402

M_DEFAULT = 2
N_DEFAULT = 4
SNR_DB = 20.0
MIN_ANGLE = 12.0
P_GRID = [0.02, 0.05, 0.10, 0.20, 0.40]


def eigen_pool(M: int, N: int, n_mat: int, seed: int) -> dict[int, np.ndarray]:
    """对每个 J 收集随机 J 列子集的 Gram 特征值（已降序）。"""
    rng = np.random.default_rng(seed)
    pool: dict[int, list[np.ndarray]] = {J: [] for J in range(1, N + 1)}
    combos = {J: np.array(list(__import__("itertools").combinations(range(N), J)))
              for J in range(1, N + 1)}
    for _ in range(n_mat):
        A = D.gen_mixing_matrix(M, N, rng, MIN_ANGLE)
        for J in range(1, N + 1):
            for cols in combos[J]:
                G = A[:, cols].T @ A[:, cols]
                lam = np.linalg.eigvalsh(G)[::-1]
                pool[J].append(np.clip(lam, 0.0, None))
    return {J: np.stack(v) for J, v in pool.items() if v}


def subset_stats(pool: dict[int, np.ndarray], n_mc: int, rng: np.random.Generator) -> list[dict]:
    """逐子集：Gram 条件数 + 均值替换导致的信号能量中位数偏移（%）。"""
    out = []
    for J, LAM in sorted(pool.items()):
        if J < 2:
            continue
        e = rng.standard_exponential((n_mc, LAM.shape[1]))
        approx_med = float(np.median(e.sum(axis=1)))
        for k in range(LAM.shape[0]):
            lam = LAM[k]
            lmax, lmin = float(lam[0]), float(lam[-1])
            exact = e @ lam
            exact_med = float(np.median(exact))
            out.append(dict(
                J=J,
                lam_min=lmin,
                lam_max=lmax,
                cond=(lmax / lmin if lmin > 1e-9 else float("inf")),
                # 正值 = 均值替换把中位能量**高估**了
                shift_pct=100.0 * (approx_med - exact_med) / exact_med,
            ))
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def r_of_p(pool: dict[int, np.ndarray], p: float, M: int, N: int, snr_db: float,
           n_mc: int, rng: np.random.Generator) -> tuple[float, float]:
    """混合律的中位数 / ν，两种信号律各算一次。

    精确：每个样本独立抽一个 J 列子集，用该子集的 {λ_j}。
    均值替换：权重一律取 λ̄ = 1。
    """
    lam_binom = np.array([math.comb(N, J) * p ** J * (1 - p) ** (N - J)
                          for J in range(N + 1)], dtype=float)
    lam_binom /= lam_binom.sum()
    Js = rng.choice(N + 1, size=n_mc, p=lam_binom)
    noise = rng.gamma(shape=M, scale=1.0, size=n_mc)      # (σ²/2)χ²_{2M}, σ²=1
    E1 = M * 10.0 ** (snr_db / 10.0) / (N * p)            # §4.2 的 SNR 约定
    devs = []
    for mode in ("exact", "approx"):
        sig = np.zeros(n_mc)
        for J in range(1, N + 1):
            sel = np.flatnonzero(Js == J)
            if sel.size == 0:
                continue
            if mode == "exact":
                idx = rng.integers(0, pool[J].shape[0], size=sel.size)
                lam = pool[J][idx]                        # (n_sel, J) 逐样本取不同子集
                e = rng.standard_exponential(sel.size * J).reshape(sel.size, J)
                sig[sel] = (E1 * J / J) * np.einsum("nk,nk->n", e, lam)
            else:
                e = rng.standard_exponential(sel.size * J).reshape(sel.size, J)
                sig[sel] = (E1 * J / J) * e.sum(axis=1)   # λ̄ = 1
        m = float(np.median(noise + sig))
        devs.append(m / (M * 1.0))                        # /ν, ν = Mσ²
    return devs[0], devs[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mat", type=int, default=200, help="混合矩阵个数")
    ap.add_argument("--n-mc", type=int, default=400_000, help="逐子集中位数 MC 规模")
    ap.add_argument("--n-mc-r", type=int, default=2_000_000, help="r(p) 的 MC 规模")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    meta = dict(protocol=dict(min_angle_deg=MIN_ANGLE, snr_db=SNR_DB,
                              p_grid=P_GRID, n_matrices=a.n_mat,
                              n_mc_subset=a.n_mc, n_mc_r=a.n_mc_r,
                              subset_rule="random J-column subsets (as in Section 4.2)",
                              threads=1, seed=a.seed))
    records, rp = [], []
    for M in (2, 3):
        pool = eigen_pool(M, N_DEFAULT, a.n_mat, a.seed + M)
        recs = subset_stats(pool, a.n_mc, rng)
        for r in recs:
            r["M"] = M
        records += recs
        for p in P_GRID:
            r_ex, r_ap = r_of_p(pool, p, M, N_DEFAULT, SNR_DB, a.n_mc_r, rng)
            rp.append(dict(M=M, p=p, r_exact=r_ex, r_approx=r_ap,
                           dev_pct=100.0 * (r_ap - r_ex) / r_ex))
        print(f"  M={M}: {len(recs)} 个子集，r(p) 已算 {len(P_GRID)} 档")

    # ---- 汇总：按条件数分箱 ----
    print("\n=== 替换偏差 vs Gram 条件数（固定最小夹角 12°）===")
    print(f"{'M':>2} {'J':>2} {'区间':>16} {'n':>6} {'中位偏移 %':>12} {'均值偏移 %':>12}")
    bins = [(1.0, 1.5), (1.5, 3.0), (3.0, 10.0), (10.0, 100.0), (100.0, float("inf"))]
    summary = []
    for M in (2, 3):
        for J in (2, 3, 4):
            sub = [r for r in records if r["M"] == M and r["J"] == J]
            if not sub:
                continue
            for lo, hi in bins:
                g = [r["shift_pct"] for r in sub if lo <= r["cond"] < hi]
                if not g:
                    continue
                lab = f"[{lo:g},{('inf' if hi == float('inf') else format(hi, 'g'))})"
                med, mn = float(np.median(g)), float(np.mean(g))
                summary.append(dict(M=M, J=J, cond_lo=lo, cond_hi=hi, n=len(g),
                                    median_shift_pct=med, mean_shift_pct=mn))
                print(f"{M:>2} {J:>2} {lab:>16} {len(g):>6} {med:>12.2f} {mn:>12.2f}")
            # 秩亏类别（λ_min = 0）
            inf = [r["shift_pct"] for r in sub if not np.isfinite(r["cond"])]
            if inf:
                summary.append(dict(M=M, J=J, cond_lo=None, cond_hi=None, n=len(inf),
                                    rank_deficient=True,
                                    median_shift_pct=float(np.median(inf)),
                                    mean_shift_pct=float(np.mean(inf))))
                print(f"{M:>2} {J:>2} {'秩亏 (λmin=0)':>16} {len(inf):>6} "
                      f"{np.median(inf):>12.2f} {np.mean(inf):>12.2f}")
            # 单调性：κ 与偏移的 Spearman（只取有限 κ）
            fin = [r for r in sub if np.isfinite(r["cond"])]
            if len(fin) > 10:
                rho = spearman(np.array([r["cond"] for r in fin]),
                               np.array([r["shift_pct"] for r in fin]))
                sh = [r["shift_pct"] for r in fin]
                print(f"{M:>2} {J:>2} {'Spearman(κ, 偏移)':>16} {len(fin):>6} "
                      f"{rho:>12.3f}   （偏移范围 {min(sh):.2f} .. {max(sh):.2f} %）")
                summary.append(dict(M=M, J=J, spearman_kappa_shift=rho,
                                    shift_min_pct=min(sh), shift_max_pct=max(sh)))

    print("\n=== r(p) 的两口径对照 ===")
    print(f"{'M':>2} {'p':>6} {'r 精确':>10} {'r 均值替换':>12} {'偏差 %':>9}")
    for d in rp:
        print(f"{d['M']:>2} {d['p']:>6.2f} {d['r_exact']:>10.4f} "
              f"{d['r_approx']:>12.4f} {d['dev_pct']:>9.3f}")

    pathlib.Path(a.out).write_text(json.dumps(
        dict(meta=meta, subset_records=records, summary=summary, r_p=rp), indent=1))
    print(f"\n已写出 {a.out}（{len(records)} 条子集记录）")


if __name__ == "__main__":
    main()
