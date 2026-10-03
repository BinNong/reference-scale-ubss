"""
评测指标模块。

包含：
  - BSS_EVAL 风格的 SDR / SIR / SAR（Vincent et al., 2006）
  - 输出 SNR（工程类 UBSS 论文常用）
  - 混合矩阵估计误差（含最优排列匹配）
  - 源数目估计误差
  - 排列/尺度模糊的处理
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# 当某源完全未被恢复时的性能下限（dB），避免出现 -inf 破坏统计
SDR_FLOOR = -40.0


# --------------------------------------------------------------------------
# BSS_EVAL 指标
# --------------------------------------------------------------------------

def _project(est: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """把 est 投影到 basis 张成的方向上（基向量为单向量）。"""
    denom = float(np.dot(basis, basis))
    if denom < 1e-15:
        return np.zeros_like(est)
    return (float(np.dot(est, basis)) / denom) * basis


def _orthonormal_basis(rows: np.ndarray) -> np.ndarray:
    """对行向量集合做正交化，返回行空间的一组标准正交基（形状 (rank, L)）。"""
    if rows is None or rows.shape[0] == 0:
        return np.zeros((0, 0))
    # 行空间的标准正交基 = 右奇异向量 Vh 的前 rank 行
    _, s, vh = np.linalg.svd(rows, full_matrices=False)
    tol = max(rows.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    rank = int(np.sum(s > tol))
    return vh[:rank]


def bss_eval_pair(
    est: np.ndarray,
    ref: np.ndarray,
    interferers: np.ndarray | None = None,
    noise: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """对单个估计源计算 (SDR, SIR, SAR)，单位 dB。

    采用**正交化后的显式分解**，避免非正交参考源导致的虚假性能上限：

        e_target = proj(est, ref)                    ← 目标分量（含最小二乘标定）
        r        = est - e_target                    ← 全部残余
        e_interf = proj(r, span⊥{其它源})            ← 干扰分量
        e_artif  = r - e_interf                      ← 伪影/噪声分量

    于是 SDR = 10log10(‖e_target‖²/‖r‖²)，完美分离时 r → 0，各指标同时趋于无穷。

    注：noise 须为**源域**一维信号；观测域加噪无法直接投影到源域，
    故本流程不传 noise，噪声引起的误差体现于 artifacts（BSS_EVAL 在无噪声参考时的标准行为）。
    """
    est = np.asarray(est, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if noise is not None:
        noise = np.asarray(noise, dtype=np.float64).ravel()
        assert noise.shape == est.shape, (
            f"noise 必须是源域一维信号；得到 {noise.shape} vs {est.shape}"
        )

    e_target = _project(est, ref)
    r = est - e_target

    if interferers is not None and interferers.shape[0] > 0:
        Q = _orthonormal_basis(np.atleast_2d(interferers))
        e_interf = (Q @ r) @ Q if Q.shape[0] > 0 else np.zeros_like(est)
    else:
        e_interf = np.zeros_like(est)

    e_artif = r - e_interf

    p_target = float(np.sum(e_target ** 2))
    p_interf = float(np.sum(e_interf ** 2))
    p_artif = float(np.sum(e_artif ** 2))
    p_res = float(np.sum(r ** 2))

    eps = 1e-30

    def db(num: float, den: float) -> float:
        num = max(num, 0.0)
        den = max(den, 0.0)
        if num <= eps:
            return -300.0                    # 目标分量为零 → 完全失败
        if den <= eps * num:
            return 300.0                     # 分母可忽略 → 视为完美
        return float(10.0 * np.log10(num / max(den, eps)))

    sdr = db(p_target, p_res)
    sir = db(p_target, p_interf)
    sar = db(p_target + p_interf, p_artif)
    return sdr, sir, sar


def best_permutation(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """基于归一化互相关，用匈牙利算法求最优源配对。

    Returns
    -------
    perm : 长度 n_ref 的数组，perm[i] 是匹配到 ref[i] 的 est 下标；
           若 ref[i] 未被匹配则为 -1。
    """
    n_ref = ref.shape[0]
    n_est = est.shape[0]

    def _nz(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-15 else v

    C = np.zeros((n_ref, n_est))
    for i in range(n_ref):
        ri = _nz(ref[i])
        for j in range(n_est):
            C[i, j] = -abs(float(np.dot(ri, _nz(est[j]))))   # 负相关 → 最小化代价

    n = max(n_ref, n_est)
    Cb = np.zeros((n, n))
    Cb[:n_ref, :n_est] = C
    Cb[:n_ref, n_est:] = 1.0      # 真实源未被匹配 → 惩罚
    Cb[n_ref:, :] = 0.0           # 多余估计源行设为哑行（不计代价）
    Cb[n_ref:, n_est:] = 0.0

    row, col = linear_sum_assignment(Cb)
    perm = np.full(n_ref, -1, dtype=int)
    for r, c in zip(row, col):
        if r < n_ref and c < n_est:
            perm[r] = c
    return perm


def evaluate_sources(
    S_true: np.ndarray,
    S_est: np.ndarray,
    noise: np.ndarray | None = None,
) -> dict:
    """整体评估：先求最优排列，再逐源计算 SDR/SIR/SAR。

    数量不等时，未被恢复的真实源按 SDR_FLOOR 计入。
    """
    N_true = S_true.shape[0]
    n_est = S_est.shape[0]
    perm = best_permutation(S_true, S_est)          # 长度 N_true，-1 表示未匹配

    sdrs, sirs, sars = [], [], []
    matched = 0

    for j in range(N_true):
        ref = S_true[j]
        others = np.delete(S_true, j, axis=0)
        jj = int(perm[j])
        if jj >= 0:
            matched += 1
            sdr, sir, sar = bss_eval_pair(S_est[jj], ref, others, noise)
        else:
            sdr = sir = sar = SDR_FLOOR
        sdrs.append(max(sdr, SDR_FLOOR))
        sirs.append(max(sir, SDR_FLOOR))
        sars.append(max(sar, SDR_FLOOR))

    # 估计出的多余源视为伪源：不并入平均，但记录其数量
    n_spurious = max(0, n_est - matched)

    return {
        "SDR": float(np.mean(sdrs)),
        "SIR": float(np.mean(sirs)),
        "SAR": float(np.mean(sars)),
        "SDR_per_source": sdrs,
        "SIR_per_source": sirs,
        "SAR_per_source": sars,
        "n_spurious": int(n_spurious),
        "perm": perm.tolist(),
    }


# --------------------------------------------------------------------------
# 输出 SNR（工程界常用，对幅值缩放敏感，故先做最小二乘标定）
# --------------------------------------------------------------------------

def output_snr(S_true: np.ndarray, S_est: np.ndarray) -> float:
    """按最优排列配对后，做最小二乘标定再计算平均 SNR_out (dB)。"""
    perm = best_permutation(S_true, S_est)
    vals = []
    for j in range(S_true.shape[0]):
        ref = S_true[j]
        if int(perm[j]) >= 0:
            est = S_est[int(perm[j])]
            alpha = float(np.dot(est, ref)) / max(float(np.dot(est, est)), 1e-30)
            est = alpha * est
            p_sig = float(np.sum(ref ** 2))
            p_err = float(np.sum((ref - est) ** 2))
            vals.append(10.0 * np.log10((p_sig + 1e-30) / (p_err + 1e-30)))
        else:
            vals.append(SDR_FLOOR)
    return float(np.mean(np.maximum(vals, SDR_FLOOR)))


# --------------------------------------------------------------------------
# 混合矩阵估计误差
# --------------------------------------------------------------------------

def _normalize_cols(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=0, keepdims=True)
    return A / np.maximum(n, 1e-15)


def mixing_matrix_nmse(A_true: np.ndarray, A_est: np.ndarray, allow_subset: bool = True) -> float:
    """混合矩阵估计的归一化均方误差（列单位化 + 最优排列匹配）。

    allow_subset=True 时，若估计列数少于真实列数，允许部分匹配
    （未匹配的真实列计入误差），以公平反映"漏检源"的代价。
    """
    At = _normalize_cols(np.asarray(A_true, dtype=np.float64))
    Ae = _normalize_cols(np.asarray(A_est, dtype=np.float64))
    n_true, n_est = At.shape[1], Ae.shape[1]

    # 代价：列间欧氏距离平方，并考虑 ± 符号等价（A 的列符号与源的符号可互换）
    D = np.zeros((n_true, n_est))
    for i in range(n_true):
        d_plus = np.sum((Ae - At[:, [i]]) ** 2, axis=0)
        d_minus = np.sum((Ae + At[:, [i]]) ** 2, axis=0)
        D[i] = np.minimum(d_plus, d_minus)

    n = max(n_true, n_est)
    Db = np.full((n, n), np.nan)
    Db[:n_true, :n_est] = D
    if allow_subset:
        # 未匹配的真实列 → 代价 2（列方向完全错误），未使用的估计列 → 代价 0
        Db[np.isnan(Db)] = 0.0
        penalty = 2.0
        Db[n_true:, :] = 0.0
        Db[:, n_est:] = penalty
        Db[n_true:, n_est:] = 0.0
        # 使匹配矩阵为方阵并保证每行每列各用一次
        M = np.where(np.arange(n)[:, None] < n_true, 1, 0)
        cost = np.where((np.arange(n)[None, :] < n_est) & (M == 1), Db, 0.0)
        cost[:n_true, n_est:] = penalty
        cost[n_true:, :] = 0.0
    else:
        cost = np.nan_to_num(Db, nan=0.0)

    row, col = linear_sum_assignment(cost)
    total = 0.0
    for r, c in zip(row, col):
        if r < n_true and c < n_est:
            total += float(D[r, c])
        elif r < n_true:
            total += 2.0
    return float(total / (4.0 * n_true))  # 归一化：每列最大误差 2*2=4


def mixing_matrix_angle_error_deg(A_true: np.ndarray, A_est: np.ndarray) -> float:
    """最优匹配后各列的平均夹角误差（度）。比 NMSE 更直观。"""
    At = _normalize_cols(np.asarray(A_true, dtype=np.float64))
    Ae = _normalize_cols(np.asarray(A_est, dtype=np.float64))
    n_true, n_est = At.shape[1], Ae.shape[1]
    D = np.zeros((n_true, n_est))
    for i in range(n_true):
        D[i] = 1.0 - np.abs(At[:, i] @ Ae)
    n = max(n_true, n_est)
    cost = np.zeros((n, n))
    cost[:n_true, :n_est] = D
    cost[:n_true, n_est:] = 1.0   # 漏检惩罚（90 度）
    row, col = linear_sum_assignment(cost)
    angs = []
    for r, c in zip(row, col):
        if r < n_true:
            cos_sim = 1.0 - (D[r, c] if c < n_est else 1.0)   # 该列最优匹配后的 |cos|
            angs.append(float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))))
    return float(np.mean(angs)) if angs else 90.0


# --------------------------------------------------------------------------
# 源数目估计
# --------------------------------------------------------------------------

def source_number_metrics(n_true: int, n_est: int) -> dict:
    return {
        "n_true": int(n_true),
        "n_est": int(n_est),
        "abs_err": int(abs(int(n_est) - int(n_true))),
        "exact": bool(int(n_est) == int(n_true)),
    }


# --------------------------------------------------------------------------
# 汇总工具
# --------------------------------------------------------------------------

def summarize(runs: list[dict]) -> dict:
    """把多次独立实验的结果汇总为 均值±标准差。"""
    if not runs:
        return {}
    out = {}
    keys = [k for k in runs[0].keys() if isinstance(runs[0][k], (int, float))]
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=np.float64)
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    exact = [r.get("exact", None) for r in runs if "exact" in r]
    if exact and all(e is not None for e in exact):
        out["n_exact_rate"] = float(np.mean([bool(e) for e in exact]))
    return out
