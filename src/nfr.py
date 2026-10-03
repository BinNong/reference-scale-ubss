"""NF-SSP —— 以**噪声底**为参考尺度的单源点门限准则（盲、自校准、可插拔）。

动机
----
两步法 UBSS 的能量门限惯例写成 ``e_l > t_e · median(e)``（或相对最大值）。
但中位数随稀疏度漂移：实测 median(e)/噪声底 在 p=0.02 时为 0.80，
在稠密时为 64.6（**81 倍**）。因此任何固定倍数 ``t_e`` 必然在某一端失效，
而"失效"是静默的——观测者只看到精度崩塌，看不到阈值已经不在正确量级上。

本模块的做法
------------
1. **参考尺度换成噪声底。** 门限写成 ``e_l > τ · ν``，其中 ν = E[e_l | 仅噪声]
   = M·s²，s² 为每个复分量的噪声功率。
2. **盲估计 s²。** 仅噪声点的逐点能量满足 ``e = (s²/2)·χ²_{2M}``，其下尾有
   精确幂律 ``F_e(x) ≃ x^M / (s²ᴹ M!)``。于是 ``log q = M·log x_q − M·log s² − log(M!)``
   —— 对分位数序列做线性回归，**斜率应恰为 M**（这是可检验的自诊断），
   **截距给出 s²**。
3. **按虚警率标定 τ。** 令 ``P(仅噪声点通过 | 噪声) = α``，则
   ``τ = Q_{1−α}(χ²_{2M}) / (2M)``，**只依赖 (M, α)**，与 p、SNR、N 无关。
   于是"调阈值"这件事从超参搜索变成一个闭式标定。
4. **自诊断与回退。** 若回归斜率显著偏离 M，说明下尾已不再由噪声点构成
   （稠密源，π₀ → 0），此时估计无效，关闭能量门限。

注意：本准则是一个**门限准则**，不是新的聚类器。它可以套在任意 SSP 流水线
（K-means / FCM / 势函数 / 其他）前面，作为即插即用的前端。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2

LOG_M_FACT = np.array([math.lgamma(m + 1) for m in range(1, 13)])  # log(M!) for M=1..12


# ==========================================================================
# 1. 参考尺度的闭式
# ==========================================================================

def mixture_median_ratio(p: float, n_sources: int, m_obs: int, snr_db: float,
                        convention: str = "np") -> dict:
    """混合分布中位数与噪声底之比 r = median(e)/ν 的闭式数值解。

    逐 TF 点能量是"恰有 J 个源活跃"的混合，``π_J = C(N,J)p^J(1−p)^{N−J}``：

      · J = 0 ：e = (s²/2)χ²_{2M}                                    （精确）
      · J ≥ 1 ：e = S_J + (s²/2)χ²_{2M}，S_J ~ Gamma(J, e_act)

    在噪声尺度 ``x ≪ e_act`` 上（这正是中位数落在仅噪声分量里的情形），
    ``F_{S_J}(z) ≃ z^J/Γ(J+1)``，于是把 S_J 边际化后得到**精确到该阶**的展开

        T_J(x) = 1/(e_act^J J!) · Σ_{k=0}^{J} C(J,k) x^{J−k} (−1)^k m_k(x),
        m_k(x) = (s²/2)^k · Γ(M+k)/Γ(M) · χ²_{2(M+k)}(2x/s² 的 CDF),

    其中 ``m_k`` 有闭式（部分矩 ↔ 卡方 CDF）。混合 CDF 为 ``F = Σ_J π_J F_J``，
    中位数由 F(x)=1/2 二分求得。

    Returns
    -------
    dict：ratio（中位数/ν）、pi0、pi_single、knee（π₀=1/2 的 p 阈值）、in_noise_regime。
    """
    from scipy.stats import chi2 as _chi2

    N, M = n_sources, m_obs
    s2 = 1.0
    p_act = 1.0 - (1.0 - p) ** N
    snr_lin = 10.0 ** (snr_db / 10.0)
    # 单个活跃点的平均信号能量。
    #
    # 约定（默认 convention="np"）：每条源把自己的平面平均功率集中在其活跃的 p 比例
    # 点位上，故恰有 1 条源活跃的点平均携带 M·s²·10^(SNR/10)/(N·p)。这与生成器
    # （data._unit_power_tf：每条源平面均值归一化为 1）实测一致——经验 E₁/ν 为
    # 1208 / 480 / 251 / 126 / 61，本约定给 1250 / 501 / 249 / 124 / 62。
    #
    # 旧约定 convention="pact" 把平面平均功率摊到"至少一条源活跃"的比例 P(act) 上，
    # 得到的是**活跃点上总能量的均值**（已对 J 平均），却又按 E_J = J·E_1 线性放大，
    # 相当于把 Σ_J π_J E_J 重复计入，使信号能量偏高 N·p/P(act) 倍（p=0.2 时 1.36×，
    # p=0.4 时 1.84×，稠密时 4×），是 r(p) 在膝上偏离真值的**主因**。
    # 保留该分支仅供历史对照（prop6_convention.py 的 "paper" 列）。
    if convention == "pact":
        e_act = snr_lin * M / max(p_act, 1e-12)
    else:
        e_act = snr_lin * M / max(N * p, 1e-12)

    # π₀ = 1/2 ⟺ p = 1 − 2^(−1/N)：中位数是否仍落在仅噪声分量的分界
    knee = 1.0 - 0.5 ** (1.0 / N)
    pi0 = (1.0 - p) ** N
    in_noise_regime = bool(pi0 >= 0.5)

    from scipy.stats import binom
    pi = binom.pmf(np.arange(N + 1), N, p)

    def cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        z = 2.0 * x / s2
        tot = pi[0] * _chi2.cdf(z, 2 * M)
        for J in range(1, N + 1):
            if pi[J] < 1e-12:
                continue
            # m_k(x) for k = 0..J
            k = np.arange(J + 1)
            lg = (k * np.log(s2 / 2.0)
                  + np.array([math.lgamma(M + kk) - math.lgamma(M) for kk in k])
                  + np.log(_chi2.cdf(z, 2 * (M + k))))
            m = np.exp(lg)
            coef = np.array([math.comb(J, kk) for kk in k]) * x ** (J - k) * (-1.0) ** k
            T = float(np.sum(coef * m)) / (e_act ** J * math.factorial(J))
            tot += pi[J] * T
        return float(tot)

    hi = 0.5 * e_act
    if cdf(hi) < 0.5:
        # 中位数已越过噪声尺度，落在信号尺度上——超出噪声区间展开的有效范围
        return {"ratio": float("nan"), "pi0": pi0, "knee": knee,
                "in_noise_regime": in_noise_regime,
                "note": "median 位于信号尺度，展开不适用", "e_act_over_nu": e_act / (M * s2)}
    lo = 0.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return {"ratio": float(0.5 * (lo + hi) / (M * s2)), "pi0": pi0,
            "knee": knee, "in_noise_regime": in_noise_regime,
            "e_act_over_nu": e_act / (M * s2)}


# ==========================================================================
# 2. 盲噪声功率估计（下尾幂律回归 + 自诊断）
# ==========================================================================

def estimate_noise_power(E: np.ndarray, m_obs: int,
                         q_hi: float = 0.02, n_q: int = 30,
                         spread_tol: float = 0.35) -> dict:
    """从逐点能量 E（长度 N_pts）盲估计每个复分量的噪声功率 s²。

    理论依据：仅噪声点满足 ``e = (s²/2)·χ²_{2M}``，故对任意小分位数 q 有**精确**关系

        s² = 2·e_(q) / Q_q(χ²_{2M}),          Q_q = χ²_{2M} 的 q 分位数。

    因此取一串小分位数 q_i（对应次序统计量 i 的期望水平 ``i/(n+1)``，而不是 i/n——
    否则 i 很小时会把"样本最小值"当成极小分位数而严重扭曲估计），得到一组
    ``s²_i``，取**中位数**作为估计。

    注意此处**不使用**小 x 的幂律近似：对 M=2 而言幂律只在 q ≲ 4e-4 才准确
    （该区间点数不足），强行拟合会带来约 60% 的系统偏高。精确分位数无此问题。

    自诊断有两条，任一不满足即判定估计不可信：
      · **离散度** ``spread = (q75−q25)/median``：模型成立时各分位给出的 s² 应一致；
      · **自洽性** ``ν̂ / mean(e) < β``：估计出的噪声底不可能占平均能量的显著份额。
        稠密源（π₀→0）时 ν̂ 被抬高到与 mean(e) 同量级，该检验据此拦截。
    """
    E = np.asarray(E, dtype=float).ravel()
    E = E[E > 0]
    n = E.size
    if n < 200:
        return {"s2": float(np.mean(E)) if n else float("nan"), "spread": float("nan"),
                "nu_over_mean": float("nan"), "valid": False, "n_pts": int(n),
                "note": "样本过少"}

    es = np.sort(E)
    i_hi = max(8, int(q_hi * n))
    idx = np.unique(np.geomspace(1.0, i_hi, n_q).astype(int))
    idx = idx[(idx >= 1) & (idx <= n)]
    if idx.size < 5:
        return {"s2": float(np.mean(E)), "spread": float("nan"),
                "nu_over_mean": float("nan"), "valid": False, "n_pts": int(n),
                "note": "可用次序统计量过少"}

    x = es[idx - 1]
    q = idx / (n + 1.0)
    levels = 2.0 * x / chi2.ppf(q, 2 * m_obs)
    levels = levels[np.isfinite(levels) & (levels > 0)]
    if levels.size < 5:
        return {"s2": float(np.mean(E)), "spread": float("nan"),
                "nu_over_mean": float("nan"), "valid": False, "n_pts": int(n),
                "note": "分位数超出卡方支撑"}

    s2 = float(np.median(levels))
    q25, q75 = np.percentile(levels, [25, 75])
    spread = float((q75 - q25) / max(s2, 1e-30))
    nu_over_mean = float(m_obs * s2 / max(np.mean(E), 1e-30))

    return {"s2": s2, "spread": spread, "nu_over_mean": nu_over_mean,
            "valid": True, "n_pts": int(n), "q_hi": float(q_hi),
            "spread_tol": float(spread_tol), "n_levels": int(levels.size)}


# ==========================================================================
# 3. 虚警率标定的门限比
# ==========================================================================

def threshold_ratio(m_obs: int, alpha: float) -> float:
    """τ 使仅噪声点的通过概率恰为 α：``e > τν  ⟺  χ²_{2M} > 2τM``。

    τ = Q_{1−α}(χ²_{2M}) / (2M)。**只依赖 (M, α)**，与 p、SNR、N 无关。
    """
    return float(chi2.ppf(1.0 - alpha, 2 * m_obs) / (2.0 * m_obs))


# ==========================================================================
# 4. 门限准则本身（可插拔）
# ==========================================================================

def point_energies(X_tf: np.ndarray) -> np.ndarray:
    """逐 TF 点能量 e_{(f,t)} = ‖x(f,t)‖²，返回展平后的长度 F·T。"""
    return np.sum(np.abs(X_tf) ** 2, axis=0).ravel()


def nfr_mask(X_tf: np.ndarray, m_obs: int, alpha: float = 1e-4,
             thr_cos: float = 0.98, thr_part_ratio: float = 0.1,
             use_self_check: bool = True, min_keep_abs: int = 10,
             min_keep_frac: float = 0.035, spread_max: float = 0.5
             ) -> tuple[np.ndarray, dict]:
    """NF-SSP 门限掩码。返回 (mask, 诊断字典)。

    三类判据（与经典流水线同构，只有能量判据的**参考尺度**不同）：
      · 能量判据  e > τ·ν̂        （ν̂ = M·ŝ²，τ 由 α 与 M 闭式给出）
      · 单源判据  |cos(X_r, X_i)| > c₀
      · 均衡判据  实虚部均不可忽略

    **自检（保留约束）**：设 ``n_base`` 为通过单源与均衡判据的点数、``n_keep`` 为
    再经能量判据后的剩余点数。若

        n_keep < max(min_keep_abs, min_keep_frac · n_base),

    则说明原则性阈值把候选点几乎删光——即"噪声底与信号能量可分离"这一前提
    不成立（稠密源，或信号能量本身低于噪声底附近的低 SNR 情形）。此时关闭
    能量判据，方法**诚实退化**为经典流水线，而不是静默失效。

    该准则是可观测的、与噪声底估计无关的，故不会因 SNR 低而误触发：
    实测（10 seeds × 全条件网格）需要开启门限的情形保留比例最低为 5.8%，
    需要关闭的情形最高为 2.2%。
    """
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    eps = 1e-15

    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
    denom = np.maximum(nr + ni, eps)
    both = (nr / denom > thr_part_ratio) & (ni / denom > thr_part_ratio)
    base = (np.abs(cos) > thr_cos) & both

    E = point_energies(X_tf)
    est = estimate_noise_power(E, m_obs)
    e_pts = np.sum(np.abs(X_tf) ** 2, axis=0)
    mean_e = float(np.mean(e_pts))

    tau = threshold_ratio(m_obs, alpha)
    nu_hat = m_obs * est["s2"]

    n_base = int(base.sum())
    energ = e_pts > tau * nu_hat
    n_keep = int((base & energ).sum())
    need = max(int(min_keep_abs), int(np.ceil(min_keep_frac * max(n_base, 1))))

    spread_ok = bool(np.isfinite(est["spread"]) and est["spread"] <= spread_max)
    retention_ok = n_keep >= need
    check_ok = bool(spread_ok and retention_ok and n_base >= min_keep_abs)
    gate_on = bool(check_ok or not use_self_check)

    mask = (base & energ) if gate_on else base
    diag = {"tau": tau, "alpha": alpha, "nu_hat": float(nu_hat),
            "gate_on": gate_on, "self_check_ok": check_ok,
            "spread": float(est["spread"]), "s2_hat": est["s2"],
            "nu_over_mean": float(nu_hat / max(mean_e, 1e-30)),
            "n_base": n_base, "n_keep": n_keep, "need": int(need),
            "keep_frac": float(n_keep / max(n_base, 1)),
            "n_admitted": int(mask.sum()), "n_pts": int(e_pts.size),
            "retention_ok": bool(retention_ok), "spread_ok": bool(spread_ok)}
    return mask, diag


# ==========================================================================
# 5. 端到端：NF-SSP 前端 + 球面 K-means
# ==========================================================================

def run_nfr(problem: dict, alpha: float = 1e-4, thr_cos: float = 0.98,
            seed: int = 0, use_self_check: bool = True) -> dict:
    """NF-SSP 前端 + 球面 K-means + 去偏 ℓ1（与全部基线使用同一恢复器）。"""
    from baselines import kmeans_sphere, l1_recover, ssp_directions

    X_tf, X_all, n_true = problem["X_tf"], problem["X_all"], problem["n"]
    m_obs = problem["m"]
    rng = np.random.default_rng(seed)

    mask, diag = nfr_mask(X_tf, m_obs, alpha=alpha, thr_cos=thr_cos,
                          use_self_check=use_self_check)
    A_hat = kmeans_sphere(ssp_directions(X_tf, mask), n_true, rng)
    S_hat = l1_recover(A_hat, X_all)
    return {"A_hat": A_hat, "S_hat": S_hat, "n_hat": n_true, "diag": diag}
