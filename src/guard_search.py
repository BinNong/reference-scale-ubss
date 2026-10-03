"""为自检找判别量：需要在 (稀疏/低SNR → 应开门) 与 (稠密 → 应关门) 之间有清晰间隔。

候选：
  pi0_hat(ν̂) = F_emp(ν̂) / P(χ²_{2M} ≤ 2M)       — 下尾中噪声点的占比估计
  pi0_hat(2ν̂), pi0_hat(0.5ν̂)                    — 同上，不同水平
  conc        = λ₁ / mean(λ) of admitted directions  — 方向是否真的成簇
  conc_ratio  = conc(开门) / conc(关门)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

import data as D
from nfr import estimate_noise_power, point_energies, threshold_ratio

CASES = [("tf_p02", 20.0), ("tf_p05", 20.0), ("tf_p10", 20.0), ("tf_p20", 20.0),
         ("tf_p40", 20.0), ("tf_gauss", 20.0),
         ("tf_p05", 0.0), ("tf_p05", 5.0), ("tf_p20", 0.0), ("tf_p20", 5.0),
         ("tf_gauss", 0.0),
         ("td_chirp", 20.0), ("td_impulse", 20.0)]


def mask_of(X_tf, M, alpha, gate, guess=None):
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0); ni = np.linalg.norm(Xi, axis=0)
    d = np.maximum(nr + ni, 1e-15)
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
    base = (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)
    if not gate:
        return base
    est = estimate_noise_power(point_energies(X_tf), M)
    tau = threshold_ratio(M, alpha)
    e = np.sum(np.abs(X_tf) ** 2, axis=0)
    return base & (e > tau * M * est["s2"])


def concentration(X_tf, mask):
    from baselines import ssp_directions
    U = ssp_directions(X_tf, mask)
    if U.shape[1] < 3:
        return float("nan")
    G = U @ U.T
    w = np.linalg.eigvalsh(G)[::-1]
    return float(w[0] / max(np.mean(w), 1e-30))


def main():
    alpha = 1e-4
    print(f"{'case':<22s}{'ŝ²/s²':>8s}{'π̂₀(ν̂)':>9s}{'π̂₀(2ν̂)':>10s}{'π̂₀(.5ν̂)':>11s}"
          f"{'conc_ON':>9s}{'conc_OFF':>9s}{'λ₁/均(ON)':>10s}{'λ₁/均(OFF)':>11s}")
    for key, snr in CASES:
        a, b, c, d1, d2, l1, l2 = [], [], [], [], [], [], []
        for s in range(5):
            p = D.make_problem(2, 4, 33, 64, key, snr, seed=(1000 if snr == 20.0 else 2000) + s)
            s2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
            E = point_energies(p["X_tf"])
            est = estimate_noise_power(E, 2)
            nu = 2 * est["s2"]
            Es = np.sort(E)
            f0 = lambda x: chi2.cdf(2 * x / est["s2"], 4)
            a.append(est["s2"] / s2)
            for x, buf in ((nu, b), (2 * nu, c), (0.5 * nu, d1)):
                Femp = np.searchsorted(Es, x) / Es.size
                buf.append(Femp / max(f0(x), 1e-12))
            m_on = mask_of(p["X_tf"], 2, alpha, True)
            m_off = mask_of(p["X_tf"], 2, alpha, False)
            l1.append(concentration(p["X_tf"], m_on))
            l2.append(concentration(p["X_tf"], m_off))
            from baselines import ssp_directions
            U = ssp_directions(p["X_tf"], m_on)
            if U.shape[1] >= 3:
                w = np.linalg.eigvalsh(U @ U.T)[::-1]
                d2.append(float(w[0] / np.mean(w)))
        print(f"{key+'@'+str(int(snr))+'dB':<22s}{np.mean(a):>8.2f}{np.mean(b):>9.3f}"
              f"{np.mean(c):>10.3f}{np.mean(d1):>11.3f}{'':>9s}{'':>9s}"
              f"{np.mean(l1):>10.3f}{np.mean(l2):>11.3f}")


if __name__ == "__main__":
    main()
