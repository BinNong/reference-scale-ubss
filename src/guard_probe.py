"""综合诊断：闭式参考尺度 V1 + 盲噪声估计与失效判据 V2。

判据目标：门限应在 p≤0.40 开启、在稠密端关闭。
"""
from __future__ import annotations

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from nfr import (estimate_noise_power, mixture_median_ratio, nfr_mask,
                 point_energies, threshold_ratio)

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
P_OF = {"tf_p02": 0.02, "tf_p05": 0.05, "tf_p10": 0.10, "tf_p20": 0.20, "tf_p40": 0.40}


def classical_mask(X_tf, c0=0.98, pr=0.1):
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0); ni = np.linalg.norm(Xi, axis=0)
    d = np.maximum(nr + ni, 1e-15)
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
    return (np.abs(cos) > c0) & (nr / d > pr) & (ni / d > pr)


def main():
    print("=" * 96)
    print("V1  闭式 r(p)=median(e)/ν  vs  实测（含多源 Gamma 卷积）")
    print("=" * 96)
    print(f"  {'cfg':<10s}{'M':>3s}{'π₀':>7s}{'knee p*':>9s}{'实测 r':>11s}{'闭式 r':>11s}{'相对误差':>10s}{'噪声区间':>9s}")
    for key in LADDER:
        for M in [2, 3]:
            meas, pred, meta = [], [], []
            for s in range(5):
                p = D.make_problem(M, 4, 33, 64, key, 20.0, seed=1000 + s)
                s2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
                meas.append(float(np.median(point_energies(p["X_tf"]))) / (M * s2))
                pv = P_OF.get(key, p["active_ratio"])
                r = mixture_median_ratio(pv, 4, M, 20.0)
                pred.append(r["ratio"]); meta.append(r)
            m_, p_ = np.mean(meas), np.mean(pred)
            pv = P_OF.get(key, 1.0)
            r0 = meta[0]
            pstr = f"{p_:>11.3f}" if np.isfinite(p_) else f"{'—':>11s}"
            estr = f"{(p_-m_)/m_:>10.1%}" if np.isfinite(p_) else f"{'—':>10s}"
            print(f"  {key:<10s}{M:>3d}{(1-pv)**4:>7.3f}{r0['knee']:>9.3f}{m_:>11.3f}{pstr}{estr}"
                  f"{str(r0['in_noise_regime']):>9s}")

    print()
    print("=" * 96)
    print("V2  盲噪声估计、失效判据与端到端 A 误差")
    print("=" * 96)
    for M in [2, 3]:
        tau = threshold_ratio(M, 1e-4)
        print(f"\n  --- M={M}, alpha=1e-4, tau={tau:.3f} ---")
        print(f"  {'cfg':<10s}{'s2真':>9s}{'ŝ²':>9s}{'比':>7s}{'spread':>8s}"
              f"{'ν̂/均':>8s}{'自检':>6s}{'门开误差':>9s}{'门关误差':>9s}")
        for key in LADDER:
            rows = []
            for s in range(5):
                p = D.make_problem(M, 4, 33, 64, key, 20.0, seed=1000 + s)
                st = float(np.mean(np.abs(p["noise_tf"]) ** 2))
                rng = np.random.default_rng(s)
                mk_on, dg = nfr_mask(p["X_tf"], M, alpha=1e-4, use_self_check=False)
                A_on = kmeans_sphere(ssp_directions(p["X_tf"], mk_on), 4, rng)
                mk_off = classical_mask(p["X_tf"])
                A_off = kmeans_sphere(ssp_directions(p["X_tf"], mk_off), 4, rng)
                rows.append((st, dg, MT.mixing_matrix_angle_error_deg(p["A"], A_on),
                             MT.mixing_matrix_angle_error_deg(p["A"], A_off)))
            st = np.mean([r[0] for r in rows]); sh = np.mean([r[1]["s2_hat"] for r in rows])
            sp = np.mean([r[1]["spread"] for r in rows])
            nm = np.mean([r[1]["nu_over_mean"] for r in rows])
            ok = all(r[1]["self_check_ok"] for r in rows)
            eon = np.mean([r[2] for r in rows]); eoff = np.mean([r[3] for r in rows])
            print(f"  {key:<10s}{st:>9.5f}{sh:>9.5f}{sh/st:>7.2f}{sp:>8.3f}"
                  f"{nm:>8.3f}{str(ok):>6s}{eon:>9.2f}{eoff:>9.2f}")


if __name__ == "__main__":
    main()
