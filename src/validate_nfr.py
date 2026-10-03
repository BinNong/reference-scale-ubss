"""NF-SSP 核心量的数值验证（纯 CPU，可重复）。

验证三件事，任一不成立则方法不成立：
  V1  闭式参考尺度 r(p) = median(e)/ν 是否与实测一致
  V2  盲噪声功率估计 ŝ² 是否无偏；下尾幂律斜率是否等于 M（自诊断有效性）
  V3  闭式门限比 τ(M, α) 下，仅噪声点的实际通过率是否等于 α
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

import data as D
from nfr import (LOG_M_FACT, estimate_noise_power, point_energies,
                 reference_scale_ratio, threshold_ratio)

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]


def v1_reference_scale():
    print("=" * 88)
    print("V1  闭式 r(p)=median(e)/ν  vs  实测")
    print("=" * 88)
    print(f"  {'cfg':<10s}{'N':>3s}{'M':>3s}{'实测 r':>10s}{'闭式 r':>10s}{'相对误差':>10s}")
    for key in LADDER:
        for N in [4]:
            for M in [2, 3]:
                meas, pred = [], []
                for s in range(5):
                    p = D.make_problem(M, N, 33, 64, key, 20.0, seed=1000 + s)
                    s2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
                    e = point_energies(p["X_tf"])
                    meas.append(float(np.median(e)) / (M * s2))
                    pf = float(p["cfg_key"].split("_p")[-1]) / 100 if "p0" in p["cfg_key"] or "p1" in p["cfg_key"] or "p2" in p["cfg_key"] or "p4" in p["cfg_key"] else None
                    pred.append(reference_scale_ratio(
                        p["active_ratio"] if pf is None else pf, N, M, 20.0))
                m_, p_ = np.mean(meas), np.mean(pred)
                print(f"  {key:<10s}{N:>3d}{M:>3d}{m_:>10.3f}{p_:>10.3f}{(p_-m_)/m_:>10.1%}")
    print()
    print("  说明：闭式用『全平面平均信号功率』的 SNR 约定（与 data.py 一致），")
    print("        多源点用 2 倍单源能量近似，故稠密端偏差最大。")


def v2_noise_estimator():
    print()
    print("=" * 88)
    print("V2  盲噪声功率估计 ŝ² 与下尾幂律斜率（自诊断）")
    print("=" * 88)
    print(f"  {'cfg':<10s}{'M':>3s}{'真值 s2':>10s}{'ŝ²':>10s}{'比值':>8s}"
          f"{'斜率':>8s}{'目标':>6s}{'R²':>8s}{'自检':>6s}")
    for key in LADDER:
        for M in [2, 3]:
            r_true, r_hat, sl, r2, ok = [], [], [], [], []
            for s in range(5):
                p = D.make_problem(M, 4, 33, 64, key, 20.0, seed=1000 + s)
                st = float(np.mean(np.abs(p["noise_tf"]) ** 2))
                est = estimate_noise_power(point_energies(p["X_tf"]), M)
                r_true.append(st); r_hat.append(est["s2"])
                sl.append(est["slope"]); r2.append(est["r2"]); ok.append(est["valid"])
            a, b = np.mean(r_true), np.mean(r_hat)
            print(f"  {key:<10s}{M:>3d}{a:>10.5f}{b:>10.5f}{b/a:>8.2f}"
                  f"{np.mean(sl):>8.2f}{float(M):>6.0f}{np.mean(r2):>8.4f}"
                  f"{str(bool(np.all(ok))):>6s}")


def v3_alpha_calibration():
    print()
    print("=" * 88)
    print("V3  闭式 τ(M,α) 下单点噪声通过率是否等于 α（蒙特卡洛，每格 2e6 点）")
    print("=" * 88)
    rng = np.random.default_rng(0)
    print(f"  {'M':>3s}{'alpha':>10s}{'τ':>10s}{'实测通过率':>12s}{'相对误差':>10s}")
    for M in [2, 3, 4]:
        n = 2_000_000
        s2 = 1.0
        z = (rng.standard_normal((M, n)) + 1j * rng.standard_normal((M, n))) * np.sqrt(s2 / 2.0)
        e = np.sum(np.abs(z) ** 2, axis=0)
        for alpha in [1e-2, 1e-3, 1e-4, 1e-5]:
            tau = threshold_ratio(M, alpha)
            emp = float(np.mean(e > tau * M * s2))
            print(f"  {M:>3d}{alpha:>10.0e}{tau:>10.3f}{emp:>12.3e}{(emp-alpha)/alpha:>10.1%}")


if __name__ == "__main__":
    v1_reference_scale()
    v2_noise_estimator()
    v3_alpha_calibration()
