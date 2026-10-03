"""机理验证：按能量加权方向能否闭合真实数据上的差距？

真实数据诊断显示，通过单源判据的点里方向精度强烈依赖能量（最低能量档中位角误差
11.6°、真单源比例 28%；最高档 1.8°、94%）。硬阈值只是"按能量筛点"的粗糙做法，
其代价是丢掉大量点数（方差上升）。因此自然的改法是**不筛点、而按能量加权**。

本脚本在真实语音上比较：
  · NF-SSP（硬门限，不加权）                     —— 现方法
  · NF-SSP 门限 + 能量加权（w ∝ e 与 w 饱和两种） —— 本文思路的自然延伸
  · 基础掩码 + 能量加权（完全不做能量门限）
  · top-5% / max 参考（激进筛选的两个代表）

用法: ../.venv/bin/python energy_weight_test.py --out ../results/energy_weight.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.cluster import KMeans

import metrics as MT
import realdata as RD
from baselines import (canonical_sign, kmeans_sphere, ssp_directions,
                       ssp_mask_from_complex, unit_cols)
from nfr import nfr_mask, threshold_ratio


def kmeans_weighted(U: np.ndarray, k: int, rng, w: np.ndarray, n_init: int = 10):
    """带样本权重的球面 K-means（在外积特征空间做加权 K-means）。"""
    M = U.shape[0]
    if U.shape[1] < k:
        return None
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    feats = np.einsum("ik,jk->ijk", U, U).reshape(M * M, -1).T
    km = KMeans(n_clusters=k, n_init=n_init, random_state=int(rng.integers(1 << 31)))
    lab = km.fit_predict(feats, sample_weight=w)
    C = np.zeros((M, k))
    for c in range(k):
        sel = U[:, lab == c]
        sw = w[lab == c]
        if sel.shape[1] == 0:
            C[:, c] = rng.standard_normal(M)
        else:
            # 加权主方向 = 加权散点矩阵的主特征向量
            G = (sel * sw) @ sel.T
            ev, evec = np.linalg.eigh(G)
            C[:, c] = evec[:, -1]
    return unit_cols(canonical_sign(C))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="../results/energy_weight.json")
    a = ap.parse_args()

    pool = RD.load_pool(RD.scan_corpus(a.corpus), n_files=40, dur_s=3.0, seed=0,
                        min_speakers=4)
    CASES = [("w512 S20", dict(win=512, snr_db=20.0, n_sources=4)),
             ("w1024 S20", dict(win=1024, snr_db=20.0, n_sources=4)),
             ("w1024 S30", dict(win=1024, snr_db=30.0, n_sources=4)),
             ("w1024 S10", dict(win=1024, snr_db=10.0, n_sources=4)),
             ("w1024 d1", dict(win=1024, snr_db=20.0, n_sources=4, n_dense=1)),
             ("w1024 N3", dict(win=1024, snr_db=20.0, n_sources=3))]

    print("=" * 104)
    print("按能量加权方向 vs 硬门限筛点（真实语音，5 seeds 均值，A 误差 °）")
    print("=" * 104)
    h = (f"{'case':<11s}{'NF硬门限':>10s}{'NF+线性权':>11s}{'NF+饱和权':>11s}"
         f"{'基础+线性权':>12s}{'基础+饱和权':>12s}{'top-5%':>9s}{'max c=.05':>11s}"
         f"{'各法最优':>10s}")
    print(h); print("-" * len(h))
    payload = {}
    for nm, kw in CASES:
        acc = {k: [] for k in ["nf", "nf_lin", "nf_sat", "base_lin", "base_sat", "topk", "maxc"]}
        nkeep = []
        for s in range(a.seeds):
            p = RD.make_real_problem(pool, m_obs=2, dur_s=3.0, hop=kw.get('win', 1024) // 2,
                                     seed=51 + s, **kw)
            X_tf, n = p["X_tf"], p["n"]
            e_full = np.sum(np.abs(X_tf) ** 2, axis=0)
            mk_nf, dg = nfr_mask(X_tf, p["m"], alpha=1e-4)
            mk_base = ssp_mask_from_complex(X_tf, thr_cos=0.98, thr_energy_ratio=0.0)
            e = e_full.ravel()

            def err(mask, weight=None):
                U = ssp_directions(X_tf, mask)
                if U.shape[1] < n:
                    return np.nan
                rng = np.random.default_rng(s)
                if weight is None:
                    A = kmeans_sphere(U, n, rng)
                else:
                    w = weight[mask.ravel()]
                    A = kmeans_weighted(U, n, rng, w)
                    if A is None:
                        return np.nan
                return MT.mixing_matrix_angle_error_deg(p["A"], A)

            tau_nu = threshold_ratio(p["m"], 1e-4) * dg["nu_hat"]
            acc["nf"].append(err(mk_nf))
            acc["nf_lin"].append(err(mk_nf, e))
            acc["nf_sat"].append(err(mk_nf, e / (e + tau_nu)))
            acc["base_lin"].append(err(mk_base, e))
            acc["base_sat"].append(err(mk_base, e / (e + tau_nu)))
            thr = np.quantile(e_full, 0.95)
            acc["topk"].append(err(mk_base & (e_full > thr)))
            acc["maxc"].append(err(mk_base & (e_full >= 0.05 * e_full.max())))
            nkeep.append(dg["keep_frac"])

        m = {k: float(np.nanmean(v)) for k, v in acc.items()}
        best = min(m.values())
        payload[nm] = dict(mean=m, keep_frac=float(np.mean(nkeep)))
        print(f"{nm:<11s}{m['nf']:>10.2f}{m['nf_lin']:>11.2f}{m['nf_sat']:>11.2f}"
              f"{m['base_lin']:>12.2f}{m['base_sat']:>12.2f}{m['topk']:>9.2f}"
              f"{m['maxc']:>11.2f}{best:>10.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
