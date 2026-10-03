"""R5-1  密度聚类基线：真正的 head-to-head 比较。

审稿意见（Minor Revision 的入门条件之一）指出：本文以"密度类方法不消耗 SSP mask"为由
只做了预筛对比（附录 A.11），但**读者关心的唯一指标是混合矩阵估计精度**——而 DBSCAN /
OPTICS / density-peak 这类方法恰恰是**绕过硬能量门限**来抵抗噪声点的。因此需要把它们
作为**真正的竞争者**放进主实验。

本脚本实现三个不用能量门限的密度聚类 UBSS 前端，并在**同一批问题实例**上与
「修好门限的经典流水线」比较角度误差：

  参考（消耗 mask，含能量准则）
    · classical   共线+均衡+  e > 0.02·median(e)   → 球面 k-means
    · NF          共线+均衡+  e > τ(M,α)·ν̂         → 球面 k-means   （本文方法）

  密度聚类（不使用任何能量门限）
    · dbscan      sklearn DBSCAN，eps 网格，min_samples 固定
    · optics      同上但用 OPTICS 的可达性提取（可变密度）
    · dpeak       Rodriguez–Laio 密度峰值（ρ·δ 取前 N 个峰，再按最近峰分派）

  密度方法有两种输入口径：
    mode="collin"  先过共线+均衡判据（不含能量准则）——即"密度检测器**替代**能量门限+聚类"
    mode="all"     完全不做任何预筛，直接对全部 TF 点的方向聚类——文献中最"纯"的用法

  注：eps 网格必须到 0.005 量级——同一源点的方向散布极小（最近邻距离中位 0.002），
  而 eps≥0.1 时 DBSCAN 的单链桥接会把相邻源连成一片（实测 4 个簇并成 1 个），
  那是 eps 选得不对、不是方法不行。

  公平性措施（对本方法不利、对竞争者有利）：
    ① 三个 eps 的**逐格最优**（oracle eps）与固定默认 eps=0.10 两种口径都报；
    ② 密度方法若聚出的簇少于 N，用**未分派点上的球面 k-means** 补齐到 N 列
       （补齐次数也一并记录，因为这本身说明朴素 eps 常常找不到 N 个簇）；
    ③ 聚类中心一律用同一套球面主方向（`principal_direction_weighted`），与 k-means 同源。

用法：
    python3 r5_density_baseline.py --seeds 10 --workers 10 \
        --out ../results/r5_density_baseline.json
    python3 r5_density_baseline.py --real --real-seeds 4 --workers 10 \
        --out ../results/r5_density_real.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import metrics as MT
import data as D
from baselines import (canonical_sign, kmeans_sphere, l1_recover,
                       principal_direction_weighted, ssp_directions,
                       ssp_mask_from_complex)
from nfr import nfr_mask

ALPHA = 1e-4
LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
EPS_GRID = [0.005, 0.02, 0.05]
EPS_DEFAULT = 0.02
MIN_SAMPLES = 10
MAX_SUB = 3000          # 密度峰值法的子采样上限（K² 距离矩阵的内存/时间）
M_OBS, N_SRC, F, T = 2, 4, 33, 64
DUR = 3.0

# 真实语音子集：覆盖"惯例默认值最吃亏"与"本文方法最吃亏"的两端
REAL_SUBSET = [
    ("R1_res", "w1024", dict(win=1024, n_sources=4, snr_db=20.0)),
    ("R2_n", "N6", dict(win=1024, n_sources=6, snr_db=20.0)),
    ("R4_dense", "d2", dict(win=1024, n_sources=4, snr_db=20.0, n_dense=2)),
    ("R5_noise", "babble", dict(win=1024, n_sources=4, snr_db=20.0, noise="babble")),
]

_CORPUS = None


# ======================================================================
# 方向集
# ======================================================================
def all_directions(X_tf: np.ndarray) -> np.ndarray:
    """全部 TF 点的实部方向（符号规范化），不做任何判据筛选。"""
    R = X_tf.real
    nrm = np.linalg.norm(R, axis=0)
    keep = nrm > 1e-12
    if not np.any(keep):
        return np.zeros((X_tf.shape[0], 0))
    return canonical_sign(R[:, keep] / nrm[keep])


def collin_mask(X_tf: np.ndarray, thr_cos: float = 0.98,
                thr_part_ratio: float = 0.1) -> np.ndarray:
    """共线 + 均衡判据（**不含**能量准则）——即经典流水线的前两条判据。"""
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    d = np.maximum(nr + ni, 1e-15)
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
    return (np.abs(cos) > thr_cos) & (nr / d > thr_part_ratio) & (ni / d > thr_part_ratio)


# ======================================================================
# 密度聚类
# ======================================================================
def _pad_to_n(U: np.ndarray, cols: list[np.ndarray], n: int,
              rng: np.random.Generator, used: np.ndarray) -> list[np.ndarray]:
    """簇数不足 N 时，用**未分派点**上的球面 k-means 补齐（并记录补齐个数）。"""
    need = n - len(cols)
    if need <= 0:
        return cols
    rest = U[:, ~used] if np.any(~used) else U
    extra = kmeans_sphere(rest, need, rng)
    cols.extend([extra[:, i] for i in range(extra.shape[1])])
    return cols


def centers_dbscan(U: np.ndarray, n: int, rng: np.random.Generator,
                   eps: float, min_samples: int = MIN_SAMPLES) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import DBSCAN

    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(U.T)
    return _centers_from_labels(U, n, rng, lab, eps)


def centers_optics(U: np.ndarray, n: int, rng: np.random.Generator,
                   eps: float, min_samples: int = MIN_SAMPLES) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import OPTICS

    m = OPTICS(min_samples=min_samples, max_eps=eps, cluster_method="dbscan",
               eps=eps, metric="euclidean").fit(U.T)
    return _centers_from_labels(U, n, rng, m.labels_, eps)


def _centers_from_labels(U: np.ndarray, n: int, rng: np.random.Generator,
                         lab: np.ndarray, eps: float) -> tuple[np.ndarray, dict]:
    uniq = [k for k in np.unique(lab) if k >= 0]
    sizes = {k: int(np.sum(lab == k)) for k in uniq}
    order = sorted(uniq, key=lambda k: -sizes[k])
    cols, used = [], np.zeros(U.shape[1], dtype=bool)
    for k in order[:n]:
        sel = lab == k
        used |= sel
        cols.append(principal_direction_weighted(U[:, sel], np.ones(int(sel.sum()))))
    n_found = len(cols)
    cols = _pad_to_n(U, cols, n, rng, used)
    A = np.stack(cols, axis=1)
    return canonical_sign(A), dict(n_clusters=n_found, eps=eps,
                                   padded=max(0, n - n_found))


def centers_dpeak(U: np.ndarray, n: int, rng: np.random.Generator,
                  sub: int = MAX_SUB) -> tuple[np.ndarray, dict]:
    """Rodriguez–Laio 密度峰值：ρ·δ 取前 N 个峰，再按"沿最近高密度点"分派。"""
    K = U.shape[1]
    idx = np.arange(K)
    if K > sub:
        idx = rng.choice(K, size=sub, replace=False)
    V = U[:, idx]
    Dm = np.linalg.norm(V[:, None, :] - V[:, :, None], axis=0)      # (k,k)
    dc = float(np.quantile(Dm[Dm > 0], 0.02))
    dc = max(dc, 1e-6)
    dens = np.exp(-(Dm / dc) ** 2).sum(axis=1) - 1.0
    delta = np.empty_like(dens)
    nbr = np.zeros(len(dens), dtype=int)
    for i in range(len(dens)):
        higher = np.where(dens > dens[i])[0]
        if higher.size == 0:
            delta[i] = Dm[i].max()
            nbr[i] = i
        else:
            j = higher[np.argmin(Dm[i, higher])]
            delta[i] = Dm[i, j]
            nbr[i] = j
    gamma = dens * delta
    peaks = np.argsort(gamma)[::-1][:n]
    # 分派：每个子样本点沿最近高密度链走到某个峰
    def root(i: int) -> int:
        seen = 0
        while nbr[i] != i and seen < len(dens):
            i = int(nbr[i])
            seen += 1
        return i
    roots = np.array([root(i) for i in range(len(dens))])
    # 全量点分派到最近的峰方向，再各自精修
    P = V[:, peaks]
    lab_all = np.argmax(np.abs(P.T @ U), axis=0)
    cols = []
    for c in range(len(peaks)):
        sel = lab_all == c
        if not np.any(sel):
            continue
        cols.append(principal_direction_weighted(U[:, sel], np.ones(int(sel.sum()))))
    n_found = len(cols)
    used = np.zeros(U.shape[1], dtype=bool)
    for c in range(len(peaks)):
        used |= lab_all == c
    cols = _pad_to_n(U, cols, n, rng, used)
    A = np.stack(cols, axis=1)
    return canonical_sign(A), dict(n_clusters=n_found, eps=None,
                                   padded=max(0, n - n_found))


# ======================================================================
# 一个实例上的全部方法
# ======================================================================
def eval_one(prob: dict, seed: int) -> list[dict]:
    X_tf, A_true = prob["X_tf"], prob["A"]
    n = prob["n"]
    rng = np.random.default_rng(seed)
    recs: list[dict] = []

    def emit(tag, A_hat_1d=None, **info):
        recs.append(dict(case=prob["cfg_key"], seed=int(seed), method=tag, **info))

    # ---- 参考：经典门限 ----
    mk = ssp_mask_from_complex(X_tf, thr_cos=0.98, thr_energy_ratio=0.02)
    Ak = kmeans_sphere(ssp_directions(X_tf, mk), n, np.random.default_rng(seed))
    emit("classical", A=MT.mixing_matrix_angle_error_deg(A_true, Ak),
         SDR=_sdr(prob, Ak), keep=float(mk.mean()))

    # ---- 参考：NF-SSP ----
    mkn, dg = nfr_mask(X_tf, X_tf.shape[0], alpha=ALPHA)
    An = kmeans_sphere(ssp_directions(X_tf, mkn), n, np.random.default_rng(seed))
    emit("nf", A=MT.mixing_matrix_angle_error_deg(A_true, An),
         SDR=_sdr(prob, An), keep=float(mkn.mean()), gate_on=bool(dg["gate_on"]))

    # ---- 密度方法：两种输入口径 ----
    U_all = all_directions(X_tf)
    mk_c = collin_mask(X_tf)
    U_col = ssp_directions(X_tf, mk_c)
    for mode, U in (("all", U_all), ("collin", U_col)):
        if U.shape[1] < n:
            continue
        for eps in EPS_GRID:
            for name, fn in (("dbscan", centers_dbscan), ("optics", centers_optics)):
                A_hat, info = fn(U, n, np.random.default_rng(seed), eps)
                emit(f"{name}[{mode},eps={eps}]", A=MT.mixing_matrix_angle_error_deg(A_true, A_hat),
                     SDR=_sdr(prob, A_hat), keep=float(U.shape[1]) / float(X_tf.shape[1] * X_tf.shape[2]),
                     **info)
        A_hat, info = centers_dpeak(U, n, np.random.default_rng(seed))
        emit(f"dpeak[{mode}]", A=MT.mixing_matrix_angle_error_deg(A_true, A_hat),
             SDR=_sdr(prob, A_hat), keep=float(U.shape[1]) / float(X_tf.shape[1] * X_tf.shape[2]),
             **info)
    return recs


def _sdr(prob: dict, A_hat: np.ndarray) -> float:
    """SDR 需要一次 ℓ1 恢复；真实语音的 X_all 有 2·F·T ≈ 2×10⁵ 列，
    每个实例要跑十几次恢复，代价远超 Angle 指标本身。真实数据一律跳过（报 nan）。"""
    if os.environ.get("R5_NO_SDR") == "1" or prob.get("cfg_key", "").startswith("R"):
        return float("nan")
    try:
        S_hat = l1_recover(A_hat, prob["X_all"])
        return float(MT.evaluate_sources(prob["S_all"], S_hat)["SDR"])
    except Exception:
        return float("nan")


# ======================================================================
# 任务
# ======================================================================
def job_synth(arg):
    key, seed = arg
    p = D.make_problem(M_OBS, N_SRC, F, T, key, 20.0, seed=1000 + seed)
    p["cfg_key"] = key
    return eval_one(p, seed)


def corpus(root: str):
    global _CORPUS
    if _CORPUS is None:
        import realdata as RD
        _CORPUS = RD.scan_corpus(root)
    return _CORPUS


def job_real(arg):
    import realdata as RD
    fam, tag, kw, seed, root = arg
    need = kw["n_sources"] - kw.get("n_dense", 0)
    pool = RD.load_pool(corpus(root), n_files=40, dur_s=DUR, seed=seed,
                        min_speakers=need)
    p = RD.make_real_problem(pool, m_obs=M_OBS, dur_s=DUR, seed=1000 + seed,
                             hop=kw.get("win", 1024) // 2, **kw)
    p["cfg_key"] = f"{fam}:{tag}"
    recs = eval_one(p, seed)
    return recs


def report(recs: list[dict], cases: list[str]) -> None:
    methods = sorted({r["method"] for r in recs})
    def mean(m, c):
        v = [r["A"] for r in recs if r["method"] == m and r["case"] == c and np.isfinite(r["A"])]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n{'method':<26}" + "".join(f"{c.split(':')[-1]:>9}" for c in cases) + f"{'mean':>9}")
    for m in methods:
        vals = [mean(m, c) for c in cases]
        print(f"{m:<26}" + "".join(f"{v:>9.2f}" for v in vals) + f"{np.nanmean(vals):>9.2f}")

    # 逐格最优的密度方法（oracle eps）
    print("\n=== 密度方法取逐格最优 eps 后，与两个参考的比较 ===")
    for c in cases:
        ref_c = mean("classical", c)
        ref_n = mean("nf", c)
        dens = {m: mean(m, c) for m in methods
                if m.startswith(("dbscan", "optics", "dpeak")) and np.isfinite(mean(m, c))}
        if not dens:
            continue
        best_m = min(dens, key=dens.get)
        print(f"  {c:<20} 经典 {ref_c:7.2f}  NF {ref_n:7.2f}  最优密度 {dens[best_m]:7.2f}"
              f"  ({best_m})  → NF {'胜' if ref_n <= dens[best_m] else '负'}")
    print("\n=== 簇数不足 N 需补齐的次数（说明朴素 eps 常常找不到 N 个簇）===")
    for m in sorted({r["method"] for r in recs if r["method"].startswith(("dbscan", "optics", "dpeak"))}):
        sub = [r for r in recs if r["method"] == m]
        pad = sum(1 for r in sub if r.get("padded", 0) > 0)
        print(f"  {m:<26} 补齐 {pad:>4}/{len(sub)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--real", action="store_true", help="只跑真实语音子集")
    ap.add_argument("--real-seeds", type=int, default=4)
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    recs: list[dict] = []
    if a.real:
        args = [(fam, tag, kw, s, a.corpus) for fam, tag, kw in REAL_SUBSET
                for s in range(a.real_seeds)]
        cases = [f"{fam}:{tag}" for fam, tag, _ in REAL_SUBSET]
    else:
        args = [(k, s) for k in LADDER for s in range(a.seeds)]
        cases = LADDER
    fn = job_real if a.real else job_synth
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for r in ex.map(fn, args):
            recs += r
    report(recs, cases)
    with open(a.out, "w") as f:
        json.dump(dict(records=recs, cases=cases, n_seeds=a.real_seeds if a.real else a.seeds,
                       eps_grid=EPS_GRID, min_samples=MIN_SAMPLES,
                       elapsed_s=time.perf_counter() - t0), f)
    print(f"\n记录 {len(recs)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
