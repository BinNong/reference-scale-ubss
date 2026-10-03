"""测量门限开启后的保留比例，为"保留约束"自检定阈值。

判据形式：若 τ·ν̂ 之上的点数少于 k_min，说明估计出的"噪声底"高于数据主体，
自相矛盾 → 判定估计无效，关闭能量门限。
"""
from __future__ import annotations

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from nfr import estimate_noise_power, point_energies, threshold_ratio

CASES = [(k, s) for k in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
         for s in [20.0]]
CASES += [("tf_p05", 0.0), ("tf_p05", 10.0), ("tf_p20", 0.0), ("tf_p20", 10.0),
          ("tf_gauss", 0.0), ("tf_gauss", 10.0),
          ("td_chirp", 20.0), ("td_sinusoid", 20.0), ("td_amfm", 20.0), ("td_impulse", 20.0)]
CASES += [(k, 20.0) for k in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]]


def main():
    alpha = 1e-4
    print(f"{'case':<20s}{'N_pts':>7s}{'keep':>7s}{'keep%':>8s}{'τν̂':>9s}"
          f"{'A(ON)':>8s}{'A(OFF)':>8s}  推断")
    worst = (1e9, None)
    for key, snr in CASES:
        ks, tots, aon, aoff = [], [], [], []
        for s in range(5):
            p = D.make_problem(2, 4, 33, 64, key, snr, seed=(1000 if snr == 20.0 else 2000) + s)
            E = point_energies(p["X_tf"])
            est = estimate_noise_power(E, 2)
            tau = threshold_ratio(2, alpha)
            thr = tau * 2 * est["s2"]
            Xr, Xi = p["X_tf"].real, p["X_tf"].imag
            nr = np.linalg.norm(Xr, axis=0); ni = np.linalg.norm(Xi, axis=0)
            d = np.maximum(nr + ni, 1e-15)
            cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
            base = (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)
            e = np.sum(np.abs(p["X_tf"]) ** 2, axis=0)
            mk_on = base & (e > thr)
            ks.append(int(mk_on.sum())); tots.append(int(base.sum()))
            for gate, buf in ((True, aon), (False, aoff)):
                rng = np.random.default_rng(s)
                mk = (base & (e > thr)) if gate else base
                A = kmeans_sphere(ssp_directions(p["X_tf"], mk), 4, rng)
                buf.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
        kpc = np.mean(ks) / np.mean(tots)
        on_ok = np.mean(aon) < np.mean(aoff)
        if kpc < worst[0]:
            worst = (kpc, key + "@" + str(int(snr)))
        print(f"{key+'@'+str(int(snr))+'dB':<20s}{np.mean(tots):>7.0f}{np.mean(ks):>7.1f}"
              f"{kpc:>8.1%}{thr:>9.4f}{np.mean(aon):>8.2f}{np.mean(aoff):>8.2f}"
              f"  {'开门更优' if on_ok else '开关门更优'}")
    print(f"\n最小保留比例 = {worst[0]:.1%}  ({worst[1]})")


if __name__ == "__main__":
    main()
