"""
基线算法集合。

统一接口：
    est = run_method(name, problem, **kwargs)
    est = {"A_hat": (M×N̂), "S_hat": (N̂×L), "n_hat": int, "time": float, "ssp_count": int}

实现的方法
----------
1. oracle_a_l1   —— 已知真实 A 的 ℓ1 恢复（性能上界）
2. ssp_kmeans_l1 —— 单源点检测 + 球面 K-means + ℓ1 恢复（经典两步法）
3. ssp_fcm_sp    —— 单源点检测 + 模糊 C 均值 + OMP 子空间投影恢复
4. pf_auto_l1    —— 势函数峰值检测自动定阶 + ℓ1 恢复（无需真实源数目）
5. sl0_l1        —— 平滑 ℓ0（σ 退火，UBSS-SAF 风格）+ ℓ1 恢复
6. duet          —— DUET 幅度比直方图（M=2）

关键实现说明
------------
* ℓ1 恢复采用「FISTA 求解 LASSO + 支撑集去偏（debiasing）」两段式。
  去偏是消除 ℓ1 幅值收缩偏差的标准做法；基线也享受该处理，故对比是公平的。
* 除 pf_auto_l1 外，其余方法均被赋予**真实源数目 N**（对基线有利）。
"""

from __future__ import annotations

import time

import numpy as np


# ==========================================================================
# 单位方向工具
# ==========================================================================

def canonical_sign(U: np.ndarray) -> np.ndarray:
    """符号规范化：每列绝对值最大的分量取正，消除 ± 二义性。"""
    if U.size == 0:
        return U
    U = np.atleast_2d(U)
    idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[idx, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    return U * signs


def unit_cols(A: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return A
    return A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-15)


def principal_direction(V: np.ndarray) -> np.ndarray:
    if V.shape[1] == 0:
        return np.zeros(V.shape[0])
    v = np.linalg.svd(V, full_matrices=False)[0][:, 0]
    return v / (np.linalg.norm(v) + 1e-15)


def principal_direction_weighted(U: np.ndarray, w: np.ndarray) -> np.ndarray:
    """加权主方向：最大化 Σ_k w_k <v, u_k>²，即 (U diag(w) Uᵀ) 的主特征向量。"""
    Gw = (U * w[None, :]) @ U.T
    _, evecs = np.linalg.eigh(Gw)
    v = evecs[:, -1]
    return v / (np.linalg.norm(v) + 1e-15)


# ==========================================================================
# 单源点检测（实部-虚部夹角法，UBSS 经典方法）
# ==========================================================================

def ssp_mask_from_complex(
    X_tf: np.ndarray,
    thr_cos: float = 0.98,
    thr_part_ratio: float = 0.1,
    thr_energy_ratio: float = 0.02,
) -> np.ndarray:
    """基于实部与虚部夹角一致性的单源点检测。

    A 为实矩阵 ⇒ 单源点上 X = a_n·s，故 Re(X) = a_n·Re(s)、Im(X) = a_n·Im(s)
    严格同向或反向，|cos(Re(X), Im(X))| → 1；多源叠加时该量下降。
    thr_part_ratio 用于剔除实/虚部接近于零（夹角无定义）的点。
    """
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    inner = np.sum(Xr * Xi, axis=0)

    tot = nr ** 2 + ni ** 2
    eps = 1e-15
    cos = inner / np.maximum(nr * ni, eps)
    denom = np.maximum(nr + ni, eps)
    both_parts = (nr / denom > thr_part_ratio) & (ni / denom > thr_part_ratio)
    med = np.median(tot[tot > 0]) if np.any(tot > 0) else 1.0
    energetic = tot > thr_energy_ratio * med
    return (np.abs(cos) > thr_cos) & both_parts & energetic


def ssp_directions(X_tf: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """从单源点提取符号规范化后的单位方向 (M×K)。"""
    Xr = X_tf.real[:, mask]
    if Xr.shape[1] == 0:
        return np.zeros((X_tf.shape[0], 0))
    nrm = np.linalg.norm(Xr, axis=0)
    keep = nrm > 1e-12
    return canonical_sign(Xr[:, keep] / nrm[keep])


# ==========================================================================
# 聚类
# ==========================================================================

def kmeans_sphere(U: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10) -> np.ndarray:
    """球面 |cos| 度量下的 K-means（等价于外积特征空间中的欧氏 K-means）。"""
    from sklearn.cluster import KMeans

    M = U.shape[0]
    if U.shape[1] < k:
        C = np.concatenate([U, rng.standard_normal((M, k - U.shape[1]))], axis=1)
        return unit_cols(canonical_sign(C))

    feats = np.einsum("ik,jk->ijk", U, U).reshape(M * M, -1).T
    km = KMeans(n_clusters=k, n_init=n_init, random_state=int(rng.integers(1 << 31)))
    lab = km.fit_predict(feats)

    C = np.zeros((M, k))
    for c in range(k):
        sel = U[:, lab == c]
        C[:, c] = principal_direction(sel) if sel.shape[1] > 0 else rng.standard_normal(M)
    return unit_cols(canonical_sign(C))


def fcm_sphere(
    U: np.ndarray, k: int, rng: np.random.Generator, m_fuzz: float = 2.0, n_iter: int = 100
) -> np.ndarray:
    """余弦距离下的模糊 C 均值，返回单位化中心 (M×k)。"""
    M = U.shape[0]
    if U.shape[1] < k:
        C = np.concatenate([U, rng.standard_normal((M, k - U.shape[1]))], axis=1)
        return unit_cols(canonical_sign(C))

    idx = rng.choice(U.shape[1], size=k, replace=False)
    C = unit_cols(U[:, idx].copy())
    for _ in range(n_iter):
        D = np.maximum(1.0 - np.abs(C.T @ U), 1e-10)
        w = 1.0 / (D ** (2.0 / (m_fuzz - 1.0)))
        w /= (np.sum(w, axis=0, keepdims=True) + 1e-30)
        C_new = np.stack([principal_direction_weighted(U, w[c]) for c in range(k)], axis=1)
        C_new = unit_cols(canonical_sign(C_new))
        if np.max(np.abs(np.abs(np.sum(C_new * C, axis=0)) - 1.0)) < 1e-9:
            C = C_new
            break
        C = C_new
    return C


def potential_function_A(
    X_tf: np.ndarray,
    n_grid: int = 1440,
    h: float | None = None,
    peak_ratio: float = 0.2,
) -> np.ndarray:
    """势函数（核密度）峰值法，自动确定源数目。

    在投影单位球面上计算高斯核密度，取显著局部极大作为混合矩阵列。
    M=2 时退化为 [0,π) 上的角度网格搜索，稳健且无需预设类别数——
    这是 UBSS 领域最早的自动定阶思路之一。

    h=None 时按样本散布自适应设定带宽。
    """
    U = ssp_directions(X_tf, ssp_mask_from_complex(X_tf))
    if U.shape[1] < 2:
        U = _all_directions(X_tf)
    if U.shape[1] == 0:
        return np.zeros((X_tf.shape[0], 0))

    M = X_tf.shape[0]
    if h is None:
        # 自适应带宽：取最近邻角距的中位数
        C = np.abs(U.T @ U)
        np.fill_diagonal(C, 0.0)
        nn = 1.0 - np.max(C, axis=1)
        h = float(np.clip(np.median(nn[nn > 0]) if np.any(nn > 0) else 0.05, 0.01, 0.3))

    if M == 2:
        th = np.linspace(0.0, np.pi, n_grid, endpoint=False)
        V = np.stack([np.cos(th), np.sin(th)], axis=0)              # (2, G)
        # 核密度（投影球面：取 |cos| 距离）
        d = 1.0 - np.abs(V.T @ U)                                   # (G, K)
        rho = np.exp(-(d ** 2) / (2 * h ** 2)).sum(axis=1)
        # 局部极大
        is_pk = (rho > np.roll(rho, 1)) & (rho >= np.roll(rho, -1))
        cand = np.where(is_pk)[0]
        if cand.size == 0:
            cand = np.array([int(np.argmax(rho))])
        thr = peak_ratio * rho.max()
        cand = cand[rho[cand] > thr]
        if cand.size == 0:
            cand = np.array([int(np.argmax(rho))])
        # 抑制相近峰
        order = cand[np.argsort(rho[cand])[::-1]]
        sel: list[int] = []
        min_sep = max(int(0.5 * np.deg2rad(8.0) / (np.pi / n_grid)), 3)
        for i in order:
            if all(min(abs(i - j), n_grid - abs(i - j)) > min_sep for j in sel):
                sel.append(int(i))
        # 精修：对每个峰做局部加权主方向
        cols = []
        for i in sel:
            w = np.exp(-(d[i] ** 2) / (2 * h ** 2))
            v = principal_direction_weighted(U, w)
            cols.append(v)
        A = np.stack(cols, axis=1)
    else:
        # M>2：多起点梯度上升找局部极大（简化实现）
        rng = np.random.default_rng(0)
        starts = U[:, rng.choice(U.shape[1], size=min(400, U.shape[1]), replace=False)]
        cols = []
        for s in range(starts.shape[1]):
            v = starts[:, s].copy()
            for _ in range(120):
                d = 1.0 - np.abs(v @ U)
                w = np.exp(-(d ** 2) / (2 * h ** 2))
                v = principal_direction_weighted(U, w)
            if all(abs(float(v @ c)) < 1.0 - 1e-3 for c in cols):
                cols.append(v)
        if not cols:
            return np.zeros((X_tf.shape[0], 0))
        A = np.stack(cols, axis=1)

    return unit_cols(canonical_sign(A))


def _all_directions(X_tf: np.ndarray, thr_energy: float = 0.02) -> np.ndarray:
    """所有有效时频点的单位方向（实虚部合并后投影）。"""
    M = X_tf.shape[0]
    V = np.concatenate([X_tf.real.reshape(M, -1), X_tf.imag.reshape(M, -1)], axis=1)
    e = np.sum(V ** 2, axis=0)
    if not np.any(e > 0):
        return np.zeros((M, 0))
    med = np.median(e[e > 0])
    V = V[:, e > thr_energy * med]
    nrm = np.linalg.norm(V, axis=0)
    V = V[:, nrm > 1e-12]
    return canonical_sign(V / np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-15))


# ==========================================================================
# 源恢复：FISTA-LASSO + 支撑集去偏
# ==========================================================================

def _ls_on_supports(A: np.ndarray, X: np.ndarray, supports: list[np.ndarray]) -> np.ndarray:
    """按支撑集分组做最小二乘（同支撑只算一次），返回 (N, L)。"""
    groups: dict[tuple, list[int]] = {}
    for l, idx in enumerate(supports):
        if idx.size == 0:
            continue
        groups.setdefault(tuple(idx.tolist()), []).append(l)

    S = np.zeros((A.shape[1], X.shape[1]))
    for key, cols in groups.items():
        idx = np.array(key)
        As = A[:, idx]
        pinv = np.linalg.pinv(As)                 # (k, M)
        S[np.ix_(idx, cols)] = pinv @ X[:, cols]
    return S


def l1_recover(
    A_hat: np.ndarray,
    X: np.ndarray,
    lam_ratio: float = 0.005,
    n_iter: int = 800,
    debias: bool = True,
    support_rel: float = 0.1,
) -> np.ndarray:
    """ℓ1 恢复 = FISTA 求解 LASSO + 支撑集去偏。

    LASSO:  min_S 0.5||A S - X||_F² + λ||S||₁ ,  λ = lam_ratio · max|AᵀX|
    去偏:   取支撑集后在支撑集上重解最小二乘，消除 ℓ1 的幅值收缩偏差。
    """
    A_hat = np.asarray(A_hat, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    N = A_hat.shape[1]
    if N == 0:
        return np.zeros((0, X.shape[1]))

    AtX = A_hat.T @ X
    G = A_hat.T @ A_hat
    Lc = float(np.linalg.norm(G, 2)) or 1.0
    lam = lam_ratio * float(np.max(np.abs(AtX)) + 1e-15)

    step = 1.0 / Lc
    thr = lam * step
    S = np.zeros((N, X.shape[1]))
    Y = S.copy()
    t = 1.0
    for _ in range(n_iter):
        Z = Y - step * (A_hat.T @ (A_hat @ Y - X))
        S_new = np.sign(Z) * np.maximum(np.abs(Z) - thr, 0.0)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        Y = S_new + ((t - 1.0) / t_new) * (S_new - S)
        S, t = S_new, t_new

    if not debias:
        return S

    mag = np.abs(S)
    mx = mag.max(axis=0, keepdims=True)
    supports = [np.nonzero(mag[:, l] > support_rel * mx[0, l])[0] for l in range(X.shape[1])]
    return _ls_on_supports(A_hat, X, supports)


def omp_recover(A_hat: np.ndarray, X: np.ndarray, max_active: int | None = None) -> np.ndarray:
    """批量化 OMP（正交匹配追踪）子空间投影恢复。"""
    A_hat = np.asarray(A_hat, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    M, N = A_hat.shape
    L = X.shape[1]
    if max_active is None:
        max_active = M
    max_active = int(min(max_active, M, N))
    if N == 0 or max_active == 0:
        return np.zeros((N, L))

    An = unit_cols(A_hat)
    R = X.copy()
    supports = [np.zeros(0, dtype=int) for _ in range(L)]
    for _ in range(max_active):
        corr = np.abs(An.T @ R)
        j = np.argmax(corr, axis=0)
        for l in range(L):
            supports[l] = np.append(supports[l], j[l])
        S = _ls_on_supports(A_hat, X, supports)
        R = X - A_hat @ S
    return S


# ==========================================================================
# 平滑 ℓ0（σ 退火，UBSS-SAF 风格）
# ==========================================================================

def _sl0_gradient_ascent(
    U: np.ndarray, w: np.ndarray, sigma: float, v0: np.ndarray, n_iter: int = 120, lr: float = 0.05
) -> np.ndarray:
    """单位球面上的**归一化**梯度上升，最大化加权核密度。

    目标：  max_v  Σ_k w_k · exp(-‖v - u_k‖² / (2σ²))

    两点关键实现：
    * 梯度量级随 1/σ² 变化，固定学习率在 σ 较小时会发散 —— 故对梯度做归一化，
      使步长与 σ 无关。
    * `_all_directions` 已做符号规范化，所有 u_k 落在同一半球，故无需 ± 对偶项
      （该项在 σ 较大时正负相消会使梯度消失）。
    """
    v = v0 / (np.linalg.norm(v0) + 1e-15)
    for _ in range(n_iter):
        dpos = v[:, None] - U
        wk = w * np.exp(-np.sum(dpos ** 2, axis=0) / (2 * sigma ** 2))
        grad = dpos @ (-wk / sigma ** 2)
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        v = v + lr * grad / gn
        v = v / (np.linalg.norm(v) + 1e-15)
    return v


def estimate_A_sl0(
    X_tf: np.ndarray,
    n_sources: int,
    rng: np.random.Generator | None = None,
    sigmas: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05),
    n_start: int = 6,
) -> np.ndarray:
    """平滑 ℓ0 逐列提取混合矩阵（σ 退火 + 多起点 + 软去相关）。"""
    if rng is None:
        rng = np.random.default_rng(0)
    U = _all_directions(X_tf)
    if U.shape[1] == 0:
        return np.zeros((X_tf.shape[0], 0))

    w = np.ones(U.shape[1])
    cols = []
    for _ in range(n_sources):
        cands = U[:, rng.choice(U.shape[1], size=min(n_start, U.shape[1]), replace=False)]
        best_v, best_val = None, -np.inf
        for s in range(cands.shape[1]):
            v = cands[:, s]
            for sg in sigmas:                       # σ 退火：由粗到细
                v = _sl0_gradient_ascent(U, w, sg, v)
            d2 = np.sum((v[:, None] - U) ** 2, axis=0)
            sg = sigmas[-1]
            val = float(np.sum(w * np.exp(-d2 / (2 * sg ** 2))))
            if val > best_val:
                best_val, best_v = val, v.copy()
        cols.append(best_v)
        # 软去相关：抑制已被解释的方向
        d2 = np.sum((best_v[:, None] - U) ** 2, axis=0)
        w = w * (1.0 - 0.95 * np.exp(-d2 / (2 * sigmas[-1] ** 2)))

    return unit_cols(canonical_sign(np.stack(cols, axis=1)))


# ==========================================================================
# DUET
# ==========================================================================

def duet(X_tf: np.ndarray, n_sources: int) -> np.ndarray:
    """DUET 幅值比直方图法（瞬时无延迟退化版，仅适用 M=2）。"""
    assert X_tf.shape[0] == 2, "DUET 仅适用于双通道"
    x1 = np.concatenate([X_tf[0].real.ravel(), X_tf[0].imag.ravel()])
    x2 = np.concatenate([X_tf[1].real.ravel(), X_tf[1].imag.ravel()])
    e = x1 ** 2 + x2 ** 2
    med = np.median(e[e > 0]) if np.any(e > 0) else 1.0
    keep = e > 0.02 * med
    x1, x2 = x1[keep], x2[keep]

    alpha = x2 / (np.abs(x1) + np.abs(x2) + 1e-15)
    hist, edges = np.histogram(alpha, bins=201, range=(-1, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])

    order = np.argsort(hist)[::-1]
    peaks: list[int] = []
    for i in order:
        if hist[i] <= 0:
            break
        if all(abs(centers[i] - centers[p]) > 0.1 for p in peaks):
            peaks.append(int(i))
        if len(peaks) >= n_sources:
            break

    A_hat = np.zeros((2, len(peaks)))
    for k, pi in enumerate(peaks):
        a = centers[pi]
        A_hat[0, k] = a
        A_hat[1, k] = 1.0 - abs(a)
    return unit_cols(canonical_sign(A_hat))


# ==========================================================================
# 统一入口
# ==========================================================================

BASELINE_NAMES = ["duet", "ssp_kmeans_l1", "ssp_fcm_sp", "pf_auto_l1", "sl0_l1", "oracle_a_l1"]

BASELINE_LABELS = {
    "duet": "DUET",
    "ssp_kmeans_l1": "SCA-L1 (SSP+K-means)",
    "ssp_fcm_sp": "SCA-SP (SSP+FCM+OMP)",
    "pf_auto_l1": "PF-auto-L1 (势函数定阶)",
    "sl0_l1": "SL0-L1 (UBSS-SAF)",
    "oracle_a_l1": "Oracle-A (真值A)",
}

# 使用自动定阶（不喂真实源数目）的方法
AUTO_N_METHODS = {"pf_auto_l1"}


def run_method(name: str, problem: dict, **kwargs) -> dict:
    """在给定问题上运行指定方法，返回 A_hat / S_hat / n_hat / time。"""
    X_all = problem["X_all"]
    X_tf = problem["X_tf"]
    n_true = problem["n"]
    rng = np.random.default_rng(int(kwargs.get("seed", 0)))
    t0 = time.perf_counter()

    mask = kwargs.get("ssp_mask")
    if mask is None:
        mask = ssp_mask_from_complex(X_tf, kwargs.get("thr_cos", 0.98))

    if name == "oracle_a_l1":
        A_hat, n_hat = problem["A"].copy(), n_true
    elif name == "ssp_kmeans_l1":
        A_hat = kmeans_sphere(ssp_directions(X_tf, mask), n_true, rng)
        n_hat = n_true
    elif name == "ssp_fcm_sp":
        A_hat = fcm_sphere(ssp_directions(X_tf, mask), n_true, rng)
        n_hat = n_true
    elif name == "pf_auto_l1":
        A_hat = potential_function_A(X_tf, kwargs.get("n_grid", 1440), kwargs.get("h"),
                                     kwargs.get("peak_ratio", 0.2))
        n_hat = A_hat.shape[1]
    elif name == "sl0_l1":
        A_hat = estimate_A_sl0(X_tf, n_true, rng)
        n_hat = n_true
    elif name == "duet":
        A_hat = duet(X_tf, n_true)
        n_hat = A_hat.shape[1]
    else:
        raise ValueError(f"未知方法: {name}")

    rec_kw = {k: kwargs[k] for k in ("lam_ratio", "n_iter", "support_rel") if k in kwargs}
    if name == "ssp_fcm_sp":
        S_hat = omp_recover(A_hat, X_all, kwargs.get("max_active", problem["m"]))
    else:
        S_hat = l1_recover(A_hat, X_all, **rec_kw)

    return {
        "A_hat": A_hat,
        "S_hat": S_hat,
        "n_hat": int(n_hat),
        "time": float(time.perf_counter() - t0),
        "ssp_count": int(np.sum(mask)),
    }
