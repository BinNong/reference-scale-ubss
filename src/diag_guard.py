"""诊断自检判据：找出只在"下尾非噪声"时才触发、而不误伤低 SNR 的准则。

对每个条件比较三种模式：
  auto   —— 当前实现（自检决定开关）
  ON     —— 强制开启能量门限
  OFF    —— 强制关闭（等价经典流水线）
并同时给出真值 s²、估计 ŝ²、spread、ν̂/mean。
"""
from __future__ import annotations

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, l1_recover, ssp_directions
from nfr import estimate_noise_power, point_energies, threshold_ratio

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
STRUCT = ["td_chirp", "td_sinusoid", "td_amfm", "td_impulse"]


def mask_forced(X_tf, M, alpha, gate: bool, thr_cos=0.98, thr_part_ratio=0.1):
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0); ni = np.linalg.norm(Xi, axis=0)
    d = np.maximum(nr + ni, 1e-15)
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
    base = (np.abs(cos) > thr_cos) & (nr / d > thr_part_ratio) & (ni / d > thr_part_ratio)
    if not gate:
        return base
    est = estimate_noise_power(point_energies(X_tf), M)
    tau = threshold_ratio(M, alpha)
    e = np.sum(np.abs(X_tf) ** 2, axis=0)
    return base & (e > tau * M * est["s2"])


def main():
    alpha = 1e-4
    print("=" * 104)
    print("A. 稀疏度阶梯（M=2, N=4, SNR=20）")
    print("=" * 104)
    print(f"  {'cfg':<11s}{'s2真':>9s}{'ŝ²/s²':>8s}{'spread':>8s}{'ν̂/均':>8s}"
          f"{'A(auto)':>9s}{'A(ON)':>8s}{'A(OFF)':>8s}")
    for key in LADDER:
        st, rt, spp, nmm, ea, eo, ef = [], [], [], [], [], [], []
        for s in range(10):
            p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
            s2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
            est = estimate_noise_power(point_energies(p["X_tf"]), 2)
            st.append(s2); rt.append(est["s2"] / s2); spp.append(est["spread"])
            nmm.append(est["nu_over_mean"] if "nu_over_mean" in est else 2 * est["s2"] / np.mean(point_energies(p["X_tf"])))
            for tag, gate, buf in (("a", True, ea), ("o", True, eo), ("f", False, ef)):
                rng = np.random.default_rng(s)
                mk = mask_forced(p["X_tf"], 2, alpha, gate)
                A = kmeans_sphere(ssp_directions(p["X_tf"], mk), 4, rng)
                buf.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
        print(f"  {key:<11s}{np.mean(st):>9.5f}{np.mean(rt):>8.2f}{np.mean(spp):>8.3f}"
              f"{np.mean(nmm):>8.3f}{np.mean(ea):>9.2f}{np.mean(eo):>8.2f}{np.mean(ef):>8.2f}")

    print()
    print("=" * 104)
    print("B. SNR 扫描（关键：低 SNR 是否被自检误伤）")
    print("=" * 104)
    print(f"  {'cfg':<11s}{'SNR':>5s}{'ŝ²/s²':>8s}{'spread':>8s}{'ν̂/均':>8s}"
          f"{'e_act/ν':>9s}{'A(auto)':>9s}{'A(ON)':>8s}{'A(OFF)':>8s}")
    for key in ["tf_p05", "tf_p20"]:
        for snr in [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]:
            st, rt, spp, nmm, ea, eo, ef, ratio = [], [], [], [], [], [], [], []
            for s in range(10):
                p = D.make_problem(2, 4, 33, 64, key, snr, seed=2000 + s)
                s2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
                est = estimate_noise_power(point_energies(p["X_tf"]), 2)
                E = point_energies(p["X_tf"])
                st.append(s2); rt.append(est["s2"] / s2); spp.append(est["spread"])
                nmm.append(2 * est["s2"] / np.mean(E))
                # e_act/ν：用真值估计活跃点能量 / 噪声底
                pv = {"tf_p05": 0.05, "tf_p20": 0.20}.get(key, 1.0)
                p_act = 1 - (1 - pv) ** 4
                e_act = 10 ** (snr / 10) * 2 / p_act
                ratio.append(e_act / (2 * s2))
                for gate, buf in ((True, eo), (False, ef)):
                    rng = np.random.default_rng(s)
                    mk = mask_forced(p["X_tf"], 2, alpha, gate)
                    A = kmeans_sphere(ssp_directions(p["X_tf"], mk), 4, rng)
                    buf.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
            print(f"  {key:<11s}{snr:>5.0f}{np.mean(rt):>8.2f}{np.mean(spp):>8.3f}"
                  f"{np.mean(nmm):>8.3f}{np.mean(ratio):>9.1f}"
                  f"{'':>9s}{np.mean(eo):>8.2f}{np.mean(ef):>8.2f}")

    print()
    print("=" * 104)
    print("C. 结构化源（M=2, N=4, SNR=20）")
    print("=" * 104)
    print(f"  {'cfg':<13s}{'π₀实':>7s}{'overlap':>9s}{'ŝ²/s²':>8s}{'spread':>8s}{'ν̂/均':>8s}"
          f"{'A(ON)':>8s}{'A(OFF)':>8s}")
    for key in STRUCT:
        pi0, ov, rt, spp, nmm, eo, ef = [], [], [], [], [], [], []
        for s in range(10):
            p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=8000 + s)
            s2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
            est = estimate_noise_power(point_energies(p["X_tf"]), 2)
            E = point_energies(p["X_tf"])
            pi0.append(p["active_ratio"]); ov.append(p["overlap"])
            rt.append(est["s2"] / s2); spp.append(est["spread"])
            nmm.append(2 * est["s2"] / np.mean(E))
            for gate, buf in ((True, eo), (False, ef)):
                rng = np.random.default_rng(s)
                mk = mask_forced(p["X_tf"], 2, alpha, gate)
                A = kmeans_sphere(ssp_directions(p["X_tf"], mk), 4, rng)
                buf.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
        print(f"  {key:<13s}{np.mean(pi0):>7.2f}{np.mean(ov):>9.2f}{np.mean(rt):>8.2f}"
              f"{np.mean(spp):>8.3f}{np.mean(nmm):>8.3f}{np.mean(eo):>8.2f}{np.mean(ef):>8.2f}")


if __name__ == "__main__":
    main()
