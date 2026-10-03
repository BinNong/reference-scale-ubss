"""
数据生成模块：时频域欠定盲分离的标准仿真模型。

问题模型
--------
观测（时频域）：   X(f,t) = A · S(f,t) + N(f,t),   f=1..F, t=1..T
其中 A ∈ R^{M×N}（实数瞬时混合），S(f,t) ∈ C^N，且 M < N（欠定）。

实值化重构
----------
记 X = Xr + jXi，S = Sr + jSi。由 A 为实矩阵可得
    Xr = A·Sr,   Xi = A·Si
两式共享同一个 A，因此将实部与虚部沿样本维拼接后，复数 UBSS 问题
**精确等价于**一个实数 UBSS 问题：
    X_all = A · S_all,  X_all ∈ R^{M×2FT},  S_all ∈ R^{N×2FT}
这一重构是精确的（非近似），且不改变稀疏结构：某时频点若只有一个源活跃，
则其实部、虚部同样只有该源活跃。全篇方法均在此实数等价形式上实现。

稀疏度控制
----------
核心难度变量是**同时活跃源数**（overlap），而非单纯的边缘分布。
所有源类型都通过 TF 域的激活概率 p 或分布形状直接控制 overlap。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft as scipy_stft, istft as scipy_istft


# ==========================================================================
# 稀疏度 / 难度度量
# ==========================================================================

def gini_index(x: np.ndarray) -> float:
    """Gini 指数（0=完全不稀疏，1=最稀疏）。"""
    v = np.abs(np.asarray(x)).ravel()
    if v.size == 0 or np.all(v == 0):
        return 1.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    if cum[-1] <= 0:
        return 1.0
    return float((n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n)


def active_ratio(x: np.ndarray, tol: float = 1e-8) -> float:
    v = np.abs(np.asarray(x)).ravel()
    return float(np.mean(v > tol))


def wdo_violation_rate(p: float, n_sources: int) -> float:
    """W-不交正交（WDO）违背率：单时频点至少两个源同时活跃的概率。

    对激活概率为 p 的 Bernoulli 源，该式为解析解，直接作为难度标签：
      P(>=2 active) = 1 - (1-p)^N - N·p·(1-p)^{N-1}
    """
    p = float(np.clip(p, 0.0, 1.0))
    return float(1.0 - (1 - p) ** n_sources - n_sources * p * (1 - p) ** (n_sources - 1))


def participation_ratio(S: np.ndarray) -> np.ndarray:
    """逐时频点的「有效活跃源数」（参与比 / 逆 Simpson 指数）。

        PR(f,t) = (Σ_n e_n)² / Σ_n e_n²,   e_n = |s_n(f,t)|²

    PR ≈ 1 表示该点被单一源支配（满足 WDO）；PR = k 表示 k 个源等强度共存。
    比"非零计数"更能反映稀疏性的真实难度，因为它对幅度衰减不敏感。
    """
    e = np.abs(np.asarray(S)) ** 2
    num = np.sum(e, axis=0) ** 2
    den = np.sum(e ** 2, axis=0) + 1e-30
    pr = num / den
    # 能量极低的点无意义，置为 0
    pr = np.where(np.sum(e, axis=0) > 1e-12 * (np.max(np.sum(e, axis=0)) + 1e-30), pr, 0.0)
    return pr


def empirical_overlap(S: np.ndarray, tol: float = 1e-8) -> float:
    """实测平均有效活跃源数（参与比，仅统计有效时频点）。"""
    pr = participation_ratio(S)
    pr = pr[pr > 0]
    return float(np.mean(pr)) if pr.size else 0.0


def empirical_wdo_violation(S: np.ndarray, pr_thr: float = 1.5) -> float:
    """实测 WDO 违背率：有效活跃源数超过 pr_thr 的时频点占比。"""
    pr = participation_ratio(S)
    pr = pr[pr > 0]
    return float(np.mean(pr > pr_thr)) if pr.size else 0.0


# ==========================================================================
# 时频域源信号生成（直接给定稀疏结构，稀疏度精确可控）
# ==========================================================================

def _unit_power_tf(S: np.ndarray) -> np.ndarray:
    p = np.mean(np.abs(S) ** 2, axis=(1, 2), keepdims=True)
    p = np.maximum(p, 1e-15)
    return S / np.sqrt(p)


def gen_tf_sources(
    n_sources: int,
    n_freq: int,
    n_frames: int,
    kind: str = "tf_sparse",
    param: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """直接生成时频域的稀疏表示 S ∈ C^{N×F×T}（复值）。

    kind / param：
      'tf_sparse'  —— Bernoulli-复高斯，param = 激活概率 p ∈ (0,1]（严格稀疏来源）
      'tf_ggauss'  —— 复广义高斯，param = 形状参数 β（β<1 稀疏，β=2 高斯）
      'tf_laplace' —— 复拉普拉斯（近似稀疏）
      'tf_gauss'   —— 稠密复高斯（完全不稀疏，p≡1）
    """
    if rng is None:
        rng = np.random.default_rng()
    N, F, T = n_sources, n_freq, n_frames
    shape = (N, F, T)

    if kind == "tf_sparse":
        p = float(np.clip(param, 1e-5, 1.0))
        mask = rng.random(shape) < p
        mag = rng.rayleigh(1.0, size=shape)
        pha = rng.uniform(0, 2 * np.pi, size=shape)
        S = mask * mag * np.exp(1j * pha)

    elif kind == "tf_ggauss":
        beta = float(max(param, 1e-2))
        # 复广义高斯：幅度服从广义高斯，相位均匀
        from scipy.stats import gennorm
        mag = gennorm.rvs(beta, size=shape, random_state=rng)
        pha = rng.uniform(0, 2 * np.pi, size=shape)
        S = mag * np.exp(1j * pha)

    elif kind == "tf_laplace":
        scale = float(max(param, 1e-3))
        re = rng.laplace(0.0, scale / np.sqrt(2), size=shape)
        im = rng.laplace(0.0, scale / np.sqrt(2), size=shape)
        S = re + 1j * im

    elif kind == "tf_gauss":
        re = rng.standard_normal(shape) / np.sqrt(2)
        im = rng.standard_normal(shape) / np.sqrt(2)
        S = re + 1j * im

    else:
        raise ValueError(f"未知的 TF 源类型: {kind}")

    return _unit_power_tf(S)


def gen_sources_time_domain(
    n_sources: int,
    n_samples: int,
    kind: str = "sinusoid",
    param: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """生成实值时域源信号（用于构造"结构化非稀疏"源：正弦/调频/调幅调频）。"""
    if rng is None:
        rng = np.random.default_rng()
    N, T = n_sources, n_samples
    t = np.linspace(0.0, 1.0, T, endpoint=False)
    S = np.zeros((N, T))

    if kind == "sinusoid":
        for n in range(N):
            for _ in range(int(rng.integers(3, 6))):
                f = rng.uniform(1.0, 60.0)
                S[n] += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    elif kind == "chirp":
        for n in range(N):
            for _ in range(int(rng.integers(2, 4))):
                f0, f1 = rng.uniform(1.0, 10.0), rng.uniform(30.0, 90.0)
                ph = np.pi * (f1 - f0) * t ** 2 + 2 * np.pi * f0 * t
                S[n] += rng.uniform(0.5, 1.5) * np.sin(ph + rng.uniform(0, 2 * np.pi))

    elif kind == "amfm":
        for n in range(N):
            fc, fm = rng.uniform(5.0, 40.0), rng.uniform(0.5, 4.0)
            am = 0.3 + 0.7 * (1.0 + np.sin(2 * np.pi * fm * t)) / 2.0
            S[n] = am * np.sin(2 * np.pi * fc * t + 2.0 * np.sin(2 * np.pi * fm * t))

    elif kind == "impulse":  # 时域脉冲串（时域稀疏，频域不稀疏）
        for n in range(N):
            p = float(param) if param > 0 else 0.05
            mask = rng.random(T) < p
            S[n] = mask * rng.standard_normal(T)

    else:
        raise ValueError(f"未知的时域源类型: {kind}")

    p = np.mean(S ** 2, axis=1, keepdims=True)
    return S / np.sqrt(np.maximum(p, 1e-15))


def stft_sources(
    S_time: np.ndarray,
    nperseg: int = 64,
    noverlap: int | None = None,
) -> tuple[np.ndarray, dict]:
    """对时域源做 STFT，得到时频域表示。返回 (S_tf, meta)。"""
    if noverlap is None:
        noverlap = nperseg * 3 // 4
    _, _, Z = scipy_stft(S_time, nperseg=nperseg, noverlap=noverlap, axis=-1, boundary=None)
    meta = {"nperseg": nperseg, "noverlap": noverlap}
    return Z, meta


def istft_sources(S_tf: np.ndarray, meta: dict, length: int | None = None) -> np.ndarray:
    _, s = scipy_istft(S_tf, nperseg=meta["nperseg"], noverlap=meta["noverlap"], boundary=None)
    if length is not None and s.shape[-1] > length:
        s = s[..., :length]
    return s


# ==========================================================================
# 源类型配置表（按预期难度从易到难）
# ==========================================================================

SOURCE_CONFIGS: dict[str, dict] = {
    "tf_p02":    {"family": "tf",   "kind": "tf_sparse",  "param": 0.02, "label": "TF 严格稀疏 (p=0.02)"},
    "tf_p05":    {"family": "tf",   "kind": "tf_sparse",  "param": 0.05, "label": "TF 严格稀疏 (p=0.05)"},
    "tf_p10":    {"family": "tf",   "kind": "tf_sparse",  "param": 0.10, "label": "TF 中等稀疏 (p=0.10)"},
    "tf_p20":    {"family": "tf",   "kind": "tf_sparse",  "param": 0.20, "label": "TF 弱稀疏 (p=0.20)"},
    "tf_p40":    {"family": "tf",   "kind": "tf_sparse",  "param": 0.40, "label": "TF 接近稠密 (p=0.40)"},
    "tf_b06":    {"family": "tf",   "kind": "tf_ggauss",  "param": 0.6,  "label": "复广义高斯 (β=0.6)"},
    "tf_lap":    {"family": "tf",   "kind": "tf_laplace", "param": 0.3,  "label": "复拉普拉斯（近似稀疏）"},
    "tf_b15":    {"family": "tf",   "kind": "tf_ggauss",  "param": 1.5,  "label": "复广义高斯 (β=1.5)"},
    "tf_gauss":  {"family": "tf",   "kind": "tf_gauss",   "param": 0.0,  "label": "稠密复高斯（非稀疏）"},
    "td_sinusoid": {"family": "td", "kind": "sinusoid",   "param": 0.0,  "label": "正弦叠加（结构化）"},
    "td_chirp":    {"family": "td", "kind": "chirp",      "param": 0.0,  "label": "线性调频（结构化）"},
    "td_amfm":     {"family": "td", "kind": "amfm",       "param": 0.0,  "label": "调幅调频（结构化）"},
    "td_impulse":  {"family": "td", "kind": "impulse",    "param": 0.05, "label": "时域脉冲串（频域不稀疏）"},
}

# 主实验使用的稀疏度阶梯（由易到难）
SPARSITY_LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]


# ==========================================================================
# 混合矩阵
# ==========================================================================

def gen_mixing_matrix(
    m_obs: int,
    n_sources: int,
    rng: np.random.Generator | None = None,
    min_angle_deg: float = 12.0,
    max_tries: int = 5000,
) -> np.ndarray:
    """列单位化的实混合矩阵 A ∈ R^{M×N}，任意两列夹角 ≥ min_angle_deg。

    方向过近会使任何算法都无法区分，因此施加最小夹角约束，
    保证各方法在公平条件下比较。
    """
    if rng is None:
        rng = np.random.default_rng()
    cos_min = np.cos(np.deg2rad(min_angle_deg))
    for _ in range(max_tries):
        A = rng.standard_normal((m_obs, n_sources))
        A /= np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-15)
        G = np.abs(A.T @ A)
        np.fill_diagonal(G, 0.0)
        if G.max() <= cos_min:
            return A
    # 约束过紧时（如 M 很小时）退化为仅单位化
    A = rng.standard_normal((m_obs, n_sources))
    return A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-15)


# ==========================================================================
# 观测组装（含实值化）
# ==========================================================================

def assemble_observation(
    A: np.ndarray,
    S_tf: np.ndarray,
    snr_db: float | None = 30.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """在时频域组装观测，并给出精确的实值等价形式。

    Returns
    -------
    dict：
      X_tf   : C^{M×F×T}  频域观测
      S_all  : R^{N×2FT}  实值化源
      X_all  : R^{M×2FT}  实值化观测
      noise_tf : C^{M×F×T} 或 None
    """
    if rng is None:
        rng = np.random.default_rng()
    M, N, F, T = A.shape[0], S_tf.shape[0], S_tf.shape[1], S_tf.shape[2]

    # 逐频点混合：X[:, f, :] = A @ S[:, f, :]
    X_tf = np.einsum("mn,nft->mft", A, S_tf)

    noise_tf = None
    if snr_db is not None:
        sig_pow = float(np.mean(np.abs(X_tf) ** 2))
        noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        noise_tf = (
            rng.standard_normal(X_tf.shape) + 1j * rng.standard_normal(X_tf.shape)
        ) * np.sqrt(noise_pow / 2.0)
        X_tf = X_tf + noise_tf

    S_all = tf_to_real(S_tf)
    X_all = tf_to_real(X_tf)
    return {"X_tf": X_tf, "S_all": S_all, "X_all": X_all, "noise_tf": noise_tf}


def tf_to_real(Z: np.ndarray) -> np.ndarray:
    """C^{N×F×T} → R^{N×2FT}：实部与虚部沿样本维拼接（精确等价变换）。"""
    n = Z.shape[0]
    return np.concatenate([Z.real.reshape(n, -1), Z.imag.reshape(n, -1)], axis=1)


def real_to_tf(Zr: np.ndarray, n_freq: int, n_frames: int) -> np.ndarray:
    """R^{N×2FT} → C^{N×F×T}：实值化的逆变换。"""
    n = Zr.shape[0]
    half = n_freq * n_frames
    re = Zr[:, :half].reshape(n, n_freq, n_frames)
    im = Zr[:, half:].reshape(n, n_freq, n_frames)
    return re + 1j * im


# ==========================================================================
# 统一入口
# ==========================================================================

def make_problem(
    m_obs: int = 3,
    n_sources: int = 5,
    n_freq: int = 33,
    n_frames: int = 64,
    cfg_key: str = "tf_p10",
    snr_db: float | None = 30.0,
    seed: int | None = None,
    min_angle_deg: float = 12.0,
    time_len: int = 4096,
    transform_meta: dict | None = None,
) -> dict:
    """生成一个完整的 UBSS 仿真问题实例（时频域）。

    对 family='tf' 的配置，直接在 TF 域生成稀疏源；
    对 family='td' 的配置，先生成时域信号再 STFT，得到结构化稀疏表示。
    """
    rng = np.random.default_rng(seed)
    cfg = SOURCE_CONFIGS[cfg_key]

    A = gen_mixing_matrix(m_obs, n_sources, rng, min_angle_deg)

    if cfg["family"] == "tf":
        S_tf = gen_tf_sources(n_sources, n_freq, n_frames, cfg["kind"], cfg["param"], rng)
        meta = {"source_family": "tf", "nperseg": None, "noverlap": None}
        S_time = None
    else:
        S_time = gen_sources_time_domain(n_sources, time_len, cfg["kind"], cfg["param"], rng)
        if transform_meta is None:
            transform_meta = {"nperseg": 64, "noverlap": 48}
        S_tf, meta = stft_sources(S_time, **transform_meta)
        meta["source_family"] = "td"
        n_freq, n_frames = S_tf.shape[1], S_tf.shape[2]
        # 重新生成源时域信号以匹配 STFT 帧数（能量归一）
        S_tf = _unit_power_tf(S_tf)

    obs = assemble_observation(A, S_tf, snr_db, rng)

    p_eff = float(cfg["param"]) if cfg["kind"] == "tf_sparse" else float(active_ratio(S_tf))
    return {
        "A": A,
        "S_tf": S_tf,
        "S_time": S_time,
        "X_tf": obs["X_tf"],
        "S_all": obs["S_all"],
        "X_all": obs["X_all"],
        "noise_tf": obs["noise_tf"],
        "noise_all": tf_to_real(obs["noise_tf"]) if obs["noise_tf"] is not None else None,
        "m": m_obs,
        "n": n_sources,
        "F": S_tf.shape[1],
        "T": S_tf.shape[2],
        "cfg_key": cfg_key,
        "cfg_label": cfg["label"],
        "snr_db": snr_db,
        "gini": gini_index(S_tf),
        "active_ratio": active_ratio(S_tf),
        "overlap": empirical_overlap(S_tf),
        "wdo_violation_emp": empirical_wdo_violation(S_tf),
        "wdo_violation": wdo_violation_rate(p_eff, n_sources),
        "transform_meta": meta,
        "time_len": S_time.shape[-1] if S_time is not None else None,
        "seed": seed,
    }


def normalize_obs(X: np.ndarray) -> np.ndarray:
    """把观测归一化到单位 RMS（稳定数值；A 误差与 SDR 对整体尺度不敏感）。"""
    s = float(np.sqrt(np.mean(X ** 2))) + 1e-12
    return X / s
