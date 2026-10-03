"""真实 vs 合成：单源点方向的精度如何随能量变化（解释"越激进越好"的机理）。

动机
----
真实数据实验发现：最优门限远比任何虚警率标定给出的都激进（等价于"只留最强的一小撮点"）。
这需要一个解释。直觉是：加性噪声下，方向估计的角误差随该点信噪比下降，
因此在 i.i.d. 合成模型里"能量高的点方向更准"应同样成立——差别在于真实语音中
**低能量点并非纯噪声**（残留泄漏、呼吸、房间噪声），其方向系统性地偏离真实混合方向，
从而污染聚类；而合成模型的低能量点只是纯噪声，方向均匀随机、不偏向任何源，
K-means 对均匀噪声相对稳健。

本脚本量化这一差别：把通过单源判据的点按能量分箱，统计
  · 方向角误差（与最近的**真值**混合方向比）
  · 该点真正是单源点的比例（用真值源的支配性判断）
  · 该点的局部信噪比

用法:
    ../.venv/bin/python real_mechanism.py --out ../results/real_mechanism.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import data as D
import metrics as MT
import realdata as RD
from baselines import canonical_sign, ssp_mask_from_complex


def direction_stats(prob):
    """对通过单源判据的每个点，返回 (能量, 方向角误差°, 是否真单源, 局部SNR dB)。"""
    X_tf = prob["X_tf"]
    A = prob["A"]
    A = A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-30)
    mask = ssp_mask_from_complex(X_tf, thr_cos=0.98, thr_energy_ratio=0.0)

    Xr = X_tf.real[:, mask]                                  # (M, K)
    nrm = np.linalg.norm(Xr, axis=0)
    keep = nrm > 1e-12
    Xr, idx = Xr[:, keep], np.where(mask.ravel())[0][keep]
    U = canonical_sign(Xr / nrm[keep])                       # (M, K)

    cos = np.abs(A.T @ U)                                    # (N, K)
    best = cos.max(axis=0)
    best[best > 1.0] = 1.0
    ang = np.degrees(np.arccos(best))

    # 该点是否真单源：由真值源的支配性判断
    F, T = X_tf.shape[1], X_tf.shape[2]
    S = prob["S_tf"].reshape(prob["n"], -1)[:, idx]
    v = np.abs(S) ** 2
    dom = v.max(axis=0) / np.maximum(v.sum(axis=0), 1e-300)
    single = dom >= 0.9

    e = np.sum(np.abs(X_tf) ** 2, axis=0).ravel()[idx]
    # 单源点上的局部信噪比：最强源功率 / 该点噪声功率估计
    nu = prob.get("nu_true", np.nan)
    s2 = (nu / prob["m"]) if (nu and np.isfinite(nu) and nu > 0) else np.nan
    loc_snr = 10.0 * np.log10(np.maximum(v.max(axis=0), 1e-300) / max(s2, 1e-300)) \
        if np.isfinite(s2) else np.full(idx.size, np.nan)
    return dict(e=e, ang=ang, single=single, loc_snr=loc_snr,
                n_base=int(idx.size), f_max=float(e.max()))


def binned(stats, n_bins=8):
    """按能量分位分箱，给出每箱的角误差中位数、单源比例、局部SNR 中位数。"""
    e, ang, single = stats["e"], stats["ang"], stats["single"]
    if e.size < 2 * n_bins:
        return []
    qs = np.quantile(e, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    out = []
    for b in range(n_bins):
        m = (e >= qs[b]) & (e < qs[b + 1])
        if m.sum() < 5:
            continue
        out.append(dict(bin=b, n=int(m.sum()),
                        e_lo=float(np.min(e[m])), e_hi=float(np.max(e[m])),
                        e_over_max=float(np.median(e[m]) / max(stats["f_max"], 1e-300)),
                        ang_med=float(np.median(ang[m])),
                        ang_mean=float(np.mean(ang[m])),
                        single_frac=float(np.mean(single[m])),
                        local_snr=float(np.nanmedian(stats["loc_snr"][m]))
                        if np.any(np.isfinite(stats["loc_snr"][m])) else float("nan")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--out", default="../results/real_mechanism.json")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    payload = {}
    pool = RD.load_pool(RD.scan_corpus(a.corpus), n_files=40, dur_s=3.0, seed=0,
                        min_speakers=4)

    print("=" * 100)
    print("方向精度随能量的变化（真实语音 vs i.i.d. 合成）")
    print("=" * 100)
    hdr = (f"{'语料':<22s}{'能量分位':>10s}{'点数':>8s}{'e/max':>10s}"
           f"{'方向角误差中位':>14s}{'真单源比例':>11s}{'局部SNR':>9s}")
    print(hdr); print("-" * len(hdr))

    real_cases = [("real w1024 S20", dict(win=1024, snr_db=20.0, n_sources=4)),
                  ("real w1024 S30", dict(win=1024, snr_db=30.0, n_sources=4)),
                  ("real w1024 N3", dict(win=1024, snr_db=20.0, n_sources=3))]
    for nm, kw in real_cases:
        p = RD.make_real_problem(pool, m_obs=2, dur_s=3.0, hop=512, seed=41, **kw)
        st = direction_stats(p)
        bins = binned(st)
        payload[f"real|{nm}"] = bins
        for b in bins:
            print(f"{nm:<22s}{b['bin']+1:>10d}{b['n']:>8d}{b['e_over_max']:>10.1e}"
                  f"{b['ang_med']:>14.2f}{b['single_frac']:>11.3f}{b['local_snr']:>9.1f}")

    print()
    syn_cases = [("synth tf_p02", "tf_p02"), ("synth tf_p10", "tf_p10"),
                 ("synth tf_p40", "tf_p40")]
    for nm, key in syn_cases:
        p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000)
        st = direction_stats(p)
        bins = binned(st)
        payload[f"synth|{nm}"] = bins
        for b in bins:
            print(f"{nm:<22s}{b['bin']+1:>10d}{b['n']:>8d}{b['e_over_max']:>10.1e}"
                  f"{b['ang_med']:>14.2f}{b['single_frac']:>11.3f}{b['local_snr']:>9.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
