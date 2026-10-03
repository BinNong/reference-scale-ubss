"""R5-2  复数（卷积混合的 STFT 像）下：诊断可迁移、判据不可迁移。

审稿意见要求本文说明「噪声基底参考尺度漂移」这一现象在**复数混合矩阵**（卷积混合
在 STFT 域的表现）语境下是否依然存在，以及 NF-SSP 能否作为插件用于频点级复数 UBSS。

本脚本把这件事拆成三个可用数值回答的问题：

  (A) **诊断是否可迁移？**
      漂移 r(p)=median(e)/ν 是关于**逐点能量边缘分布**的陈述：噪声点仍是
      (σ²/2)χ²_{2M}，仅噪声点比例仍是 π₀=(1−p)^N。所以它与 A 的几何结构无关。
      实测三种 A：实值（论文设定）、复值固定、复值逐频点（卷积混合的像），
      与 §4.2 闭式逐档比较。

  (B) **判据是否可迁移？**
      §3.2 的实值精确重构 X_r = A S_r、X_i = A S_i 依赖 **A 为实**。A 复值时该
      恒等式失效，Prop. 2 的 |cos(X_r, X_i)| 判据随之失效。实测：在复 A 下该判据
      对**真单源点**的召回率（用真实激活标注）。

  (C) **门限是否仍有用？**
      用复值投影空间里的同一套机器（复主方向 + 球面式 k-means，距离
      d(x,y)² = 2 − 2|⟨x,y⟩|/(‖x‖‖y‖)，⟨x,y⟩ = Σ x_p y_p **不取共轭**）评价四种掩码：
      全点（无判据）/ 实值共线判据（朴素移植）/ 经典中位数能量门限 / NF 门限。

用法：
    python3 r5_complex.py --seeds 10 --workers 10 --out ../results/r5_complex.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

import metrics as MT
from nfr import (estimate_noise_power, mixture_median_ratio, nfr_mask,
                 point_energies, threshold_ratio)
from baselines import kmeans_sphere, ssp_directions, ssp_mask_from_complex

ALPHA = 1e-4
LADDER = [("tf_p02", 0.02), ("tf_p05", 0.05), ("tf_p10", 0.10),
          ("tf_p20", 0.20), ("tf_p40", 0.40), ("tf_gauss", 1.0)]
M_OBS, N_SRC, F, T = 2, 4, 33, 64
SNR = 20.0


# ======================================================================
# 生成：三种混合矩阵
# ======================================================================
def _complex_A(rng, min_angle_deg: float = 25.0) -> np.ndarray:
    """复值混合矩阵：独立复高斯列、单位范数，并用复值夹角约束最小分离度。

    夹角用**不取共轭**的内积 |aᴴ... 不对，用 |aᵀb|：复值方向上 a 与 βa 视为同一方向，
    故分离度是 |aᵀb|/(‖a‖‖b‖)。
    """
    A = None
    for _ in range(4000):
        A = rng.standard_normal((M_OBS, N_SRC)) + 1j * rng.standard_normal((M_OBS, N_SRC))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        G = np.abs(A.T @ A)
        np.fill_diagonal(G, 0.0)
        if np.degrees(np.arccos(np.clip(G.max(), 0.0, 1.0))) >= min_angle_deg:
            break
    return A


def make_problem(kind: str, cfg_key: str, seed: int, snr_db: float = SNR) -> dict:
    """kind ∈ {real, cfixed, cperfreq}。

    **源的实现体 (S_tf) 三种情形完全相同**：直接取自 `data.make_problem`，只把混合
    矩阵换成复值（而不是重新生成一套源），这样三种 A 的差异只有一个变量。
    SNR 约定与 §3.3 一致：σ² = mean|X|² / 10^(SNR/10)。
    """
    import data as D

    base = D.make_problem(M_OBS, N_SRC, F, T, cfg_key, snr_db, seed=seed)
    S_tf = base["S_tf"]
    rng = np.random.default_rng(seed + 777)
    p = (float(D.SOURCE_CONFIGS[cfg_key]["param"])
         if D.SOURCE_CONFIGS[cfg_key]["kind"] == "tf_sparse" else 1.0)

    if kind == "real":
        A = base["A"]
        X = np.einsum("mn,nft->mft", A, S_tf).astype(complex)
    elif kind == "cfixed":
        A = _complex_A(rng)
        X = np.einsum("mn,nft->mft", A, S_tf)
    elif kind == "cperfreq":
        A = np.empty((M_OBS, N_SRC, F), dtype=complex)
        for f in range(F):
            A[:, :, f] = _complex_A(rng)
        X = np.einsum("mnf,nft->mft", A, S_tf)
    else:
        raise ValueError(kind)

    sig_pow = float(np.mean(np.abs(X) ** 2))
    if snr_db is None:                       # 无噪声（仅用于机器一致性自检）
        noise_pow, noise, nu = 0.0, np.zeros_like(X), 1.0
    else:
        noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        noise = (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape)) \
            * np.sqrt(noise_pow / 2.0)
        nu = M_OBS * noise_pow
    return dict(kind=kind, cfg_key=cfg_key, p=p, seed=seed, A=A, S_tf=S_tf,
                X_tf=X + noise, noise_tf=noise, s2=noise_pow, nu=nu,
                J=np.sum(np.abs(S_tf) > 1e-12, axis=0))


# ======================================================================
# 复值投影空间上的机器
# ======================================================================
def _leading_complex(V: np.ndarray) -> np.ndarray:
    """复值"主方向"：Σ_i |⟨v_i, c⟩|²/‖c‖² 的最大化者。

    关键一步（**第一版写错过，代价是整张表都成了 20° 的垃圾**）：把目标写开
    ``Σ_i |v_iᵀc|² = cᴴ·conj(Σ v_i v_iᴴ)·c``，故最大化者是 **Hermitian** 矩阵
    ``G = Σ v_i v_iᴴ`` 的首特征向量的**共轭**。若直接对 ``Σ v_i v_iᵀ``（复对称、
    **非** Hermitian）调用 ``eigh``，得到的特征向量无意义——``eigh`` 假定矩阵 Hermitian。
    实值数据下两者退化为同一件事（conj 无作用），所以只有复值路径会暴露该错误。
    """
    G = V @ V.conj().T
    val, vec = np.linalg.eigh(G)
    c = np.conj(vec[:, -1])
    n = np.linalg.norm(c)
    return c / n if n > 1e-15 else c


def kmeans_complex(X: np.ndarray, k: int, rng: np.random.Generator,
                   n_init: int = 6, n_iter: int = 60) -> np.ndarray:
    """复值投影空间上的 k-means。

    距离 ``d(x,c)² = 2 − 2|⟨x,c⟩|/(‖x‖‖c‖)``，``⟨x,c⟩ = Σ x_p c_p``（**不取共轭**）——
    这正是复值域上"相差一个复标量即同一方向"的度量。
    """
    K = X.shape[1]
    if K < k:
        return np.zeros((X.shape[0], 0), dtype=complex)
    best, best_obj = None, -np.inf
    for _ in range(n_init):
        C = X[:, rng.choice(K, size=k, replace=False)].copy()
        C /= np.maximum(np.linalg.norm(C, axis=0, keepdims=True), 1e-15)
        for _ in range(n_iter):
            A_ = np.abs(C.T @ X)
            lab = np.argmax(A_, axis=0)
            newC = np.empty_like(C)
            for j in range(k):
                sel = lab == j
                if not np.any(sel):
                    newC[:, j] = X[:, rng.integers(K)]
                    continue
                newC[:, j] = _leading_complex(X[:, sel])
            newC /= np.maximum(np.linalg.norm(newC, axis=0, keepdims=True), 1e-15)
            moved = float(np.max(np.abs(np.abs(newC.T @ C) - 1.0)))
            C = newC
            if moved < 1e-9:
                break
        obj = float(np.sum(np.max(np.abs(C.T @ X), axis=0)))
        if obj > best_obj:
            best_obj, best = obj, C
    return best


def complex_angle_error_deg(A_true: np.ndarray, A_est: np.ndarray) -> float:
    """复值（或实值）列的最优匹配平均夹角（度）。维数不同时漏检按 90° 计。"""
    if A_est is None or A_est.size == 0:
        return 90.0
    At = A_true / np.maximum(np.linalg.norm(A_true, axis=0, keepdims=True), 1e-15)
    Ae = A_est / np.maximum(np.linalg.norm(A_est, axis=0, keepdims=True), 1e-15)
    nt, ne = At.shape[1], Ae.shape[1]
    G = np.abs(At.T @ Ae)                       # 不取共轭：复值方向比对
    n = max(nt, ne)
    cost = np.ones((n, n))
    cost[:nt, :ne] = 1.0 - G
    row, col = linear_sum_assignment(cost)
    angs = []
    for r, c in zip(row, col):
        g = G[r, c] if (r < nt and c < ne) else 0.0
        angs.append(float(np.degrees(np.arccos(np.clip(g, 0.0, 1.0)))))
    return float(np.mean(angs))


def flat(X: np.ndarray) -> np.ndarray:
    """C^{M×F×T} -> C^{M×FT}"""
    return X.reshape(X.shape[0], -1)


def collin_mask_complex(X_tf: np.ndarray) -> np.ndarray:
    """朴素移植 Prop. 2 的实值共线判据（实部/虚部夹角）。"""
    return ssp_mask_from_complex(X_tf, thr_cos=0.98, thr_part_ratio=0.1,
                                 thr_energy_ratio=0.0)


# ======================================================================
# 单个实例
# ======================================================================
def eval_one(kind: str, cfg_key: str, seed: int, krng: int | None = None) -> dict:
    """krng：聚类随机数种子。与论文主表一致时取 ``seed - 1000``，便于逐档核对。"""
    krng = seed - 1000 if krng is None else krng
    pr = make_problem(kind, cfg_key, seed)
    X_tf, A, J = pr["X_tf"], pr["A"], pr["J"]
    out = dict(kind=kind, cfg_key=cfg_key, p=pr["p"], seed=seed)

    # ---------------- (A) 漂移 ----------------
    E = point_energies(X_tf)
    out["r_emp"] = float(np.median(E) / pr["nu"])
    cf = mixture_median_ratio(pr["p"], N_SRC, M_OBS, SNR)
    out["r_closed"] = float(cf["ratio"])
    out["pi0"] = float(cf["pi0"])
    out["s2_hat_over_s2"] = float(estimate_noise_power(E, M_OBS)["s2"] / pr["s2"])

    # ---------------- (B) 实值共线判据在复 A 下的召回 ----------------
    single = J == 1
    noise = J == 0
    mk = collin_mask_complex(X_tf)
    out["collin_admit_frac"] = float(mk.mean())
    tp = float(np.sum(mk & single))
    out["collin_recall_single"] = tp / max(float(single.sum()), 1.0)
    out["collin_precision"] = tp / max(float(mk.sum()), 1.0)
    out["collin_fpr_noise"] = float(np.sum(mk & noise)) / max(float(noise.sum()), 1.0)

    # ---------------- (C) 门限在复值域是否仍有用 ----------------
    Xf = flat(X_tf)
    E = point_energies(X_tf)
    m_med = E > 0.02 * float(np.median(E))
    m_nf, dg = nfr_mask(X_tf, M_OBS, alpha=ALPHA)
    # 只保留**能量判据**（本文的贡献本身），去掉实值共线判据——在复 A 下后者无效，
    # 但能量判据与 A 的结构无关，因此可单独移植。
    tau = threshold_ratio(M_OBS, ALPHA)
    nu_hat = float(estimate_noise_power(E, M_OBS)["s2"]) * M_OBS
    m_nfE = E > tau * nu_hat
    # 自检（保留约束）的复值域版本：保留率过低即认为前提失效，直接退回"不过门限"。
    # 自检读的是**可观测后果**（保留率），与 A 的结构无关，因此与能量判据一同可迁移。
    m_nfE_sc = m_nfE if float(m_nfE.mean()) >= 0.035 else np.ones(Xf.shape[1], dtype=bool)
    variants = {
        "all": np.ones(Xf.shape[1], dtype=bool),
        "collin": mk.reshape(-1),
        "median0.02": m_med.reshape(-1),
        "nf": m_nf.reshape(-1),
        "nfE": m_nfE.reshape(-1),
        "nfE+sc": m_nfE_sc.reshape(-1),
    }
    for tag, m in variants.items():
        out[f"keep_{tag}"] = float(m.mean())
        if kind == "cperfreq":
            # 逐频点 A：各频点的方向不同，跨频点聚类本身不成立（见正文），故只报保留率
            out[f"A_cx_{tag}"] = float("nan")
            out[f"A_rp_{tag}"] = float("nan")
            continue
        U = Xf[:, m]
        A_ref = np.asarray(A)
        out[f"A_cx_{tag}"] = complex_angle_error_deg(
            A_ref, kmeans_complex(U, N_SRC, np.random.default_rng(krng)))
        # 对照：把复值观测**只取实部**、再走论文原有的实值球面 k-means（"朴素移植"）——
        # 复 A 下 Re X = A_r Re s − A_i Im s 不再是 A 乘任何东西，这条路按构造就失效。
        m2d = m.reshape(F, T)
        out[f"A_rp_{tag}"] = MT.mixing_matrix_angle_error_deg(
            np.asarray(A_ref).real,
            kmeans_sphere(ssp_directions(X_tf, m2d), N_SRC, np.random.default_rng(krng)))
    out["gate_on_nf"] = bool(dg["gate_on"])
    return out


# ======================================================================
# 交叉校验：两个聚类机器在**同一批实值点**上应给出同一结果
# ======================================================================
def selftest(seeds: int = 5) -> None:
    """把同一批实值单位方向分别喂给两个机器：差异应 ≈0。

    （注意：这只检验"机器实现一致"。复值 A 下把复值点直接交给复值机器，与论文把
    实值点交给球面 k-means 并不是同一个算法——这正是 (C) 要量化的那件事。）
    """
    e1s, e2s = [], []
    for s in range(seeds):
        pr = make_problem("real", "tf_p05", 4000 + s)
        X_tf, A = pr["X_tf"], np.asarray(pr["A"], dtype=float)
        mk, _ = nfr_mask(X_tf, M_OBS, alpha=ALPHA)
        U = ssp_directions(X_tf, mk)
        A1 = kmeans_sphere(U.copy(), N_SRC, np.random.default_rng(s))
        A2 = kmeans_complex(U.astype(complex), N_SRC, np.random.default_rng(s))
        e1s.append(MT.mixing_matrix_angle_error_deg(A, np.real(A1)))
        e2s.append(complex_angle_error_deg(A, A2))
    print(f"[自检] 同一批实值方向：球面 k-means {np.mean(e1s):.4f}° vs 复值机器 "
          f"{np.mean(e2s):.4f}°，逐档最大差 {max(abs(a-b) for a, b in zip(e1s, e2s)):.4f}°（应≈0）")


def job(arg):
    kind, cfg_key, seed, krng = arg
    return eval_one(kind, cfg_key, seed, krng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="实例种子起点；默认 1000 与论文主表 E1_sparsity 一致")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    selftest()

    args = [(k, ck, a.seed_base + s, s) for k in ("real", "cfixed", "cperfreq")
            for ck, _ in LADDER for s in range(a.seeds)]
    ctx = mp.get_context("spawn")
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for r in ex.map(job, args):
            recs.append(r)

    def m(kind, ck, key):
        v = [r[key] for r in recs if r["kind"] == kind and r["cfg_key"] == ck]
        return float(np.nanmean(v)) if v else float("nan")

    print("\n=== (A) 参考尺度漂移：三种混合矩阵（**同一套源实现**）vs §4.2 闭式 ===")
    print(f"{'regime':<10}{'实值':>9}{'复值固定':>10}{'复值逐频点':>12}{'闭式':>9}"
          f"{'σ̂²/σ²':>9}{'准入率':>9}{'单源召回':>10}")
    for ck, p in LADDER:
        print(f"{ck:<10}{m('real',ck,'r_emp'):>9.2f}{m('cfixed',ck,'r_emp'):>10.2f}"
              f"{m('cperfreq',ck,'r_emp'):>12.2f}{m('real',ck,'r_closed'):>9.2f}"
              f"{m('real',ck,'s2_hat_over_s2'):>9.2f}"
              f"{m('cfixed',ck,'collin_admit_frac'):>9.4f}"
              f"{m('cfixed',ck,'collin_recall_single'):>10.3f}")

    print("\n=== (B) Prop. 2 的实值共线判据在复 A 下失效（准入率/单源召回/查准/噪声误纳）===")
    for k in ("real", "cfixed", "cperfreq"):
        print(f"  {k:<10} 准入 {np.nanmean([r['collin_admit_frac'] for r in recs if r['kind']==k]):.4f}"
              f"  单源召回 {np.nanmean([r['collin_recall_single'] for r in recs if r['kind']==k]):.3f}"
              f"  查准 {np.nanmean([r['collin_precision'] for r in recs if r['kind']==k]):.3f}"
              f"  噪声误纳 {np.nanmean([r['collin_fpr_noise'] for r in recs if r['kind']==k]):.4f}")

    MASKS = ("all", "collin", "median0.02", "nf", "nfE", "nfE+sc")
    for kind, label in (("cfixed", "复值固定 A（卷积混合的像）"), ("real", "实值 A（论文设定，对照）")):
        print(f"\n=== (C) {label}：五种掩码 × 两种聚类机器（角度误差，度）===")
        print(f"{'regime':<10}" + "".join(f"{t:>21}" for t in MASKS))
        for ck, _ in LADDER:
            row = f"{ck:<10}"
            for t in MASKS:
                row += f"{m(kind, ck, f'A_cx_{t}'):>10.2f}/{m(kind, ck, f'A_rp_{t}'):<10.2f}"
            print(row)
        for t in MASKS:
            cx = np.nanmean([r[f"A_cx_{t}"] for r in recs if r["kind"] == kind])
            rp = np.nanmean([r[f"A_rp_{t}"] for r in recs if r["kind"] == kind])
            print(f"  均值 {t:<10} 复值机器 {cx:7.3f}   实值机器 {rp:7.3f}")

    print("\n=== 保留率（同一批掩码）===")
    for t in MASKS:
        print(f"  {t:<11} {np.nanmean([r[f'keep_{t}'] for r in recs if r['kind']=='cfixed']):.4f}")

    with open(a.out, "w") as f:
        json.dump(dict(records=recs, ladder=[t for t, _ in LADDER], n_seeds=a.seeds,
                       elapsed_s=time.perf_counter() - t0), f)
    print(f"\n记录 {len(recs)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
