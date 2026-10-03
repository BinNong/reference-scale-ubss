"""能量加权方向估计：把硬门限换成（或叠加）按质量加权的球面 K-means。

动机（可写进论文的推导）
------------------------
在加性噪声模型下，一个候选点的方向 x/‖x‖ 的**角误差方差 ∝ σ²/‖x‖²**，因此关于
真实混合方向的 Fisher 信息 ∝ ‖x‖²/σ²。若把每条方向视为对同一个簇中心的独立观测，
则簇中心的最大似然估计不是"等权平均"，而是**以 ‖x‖² 为权重**的加权主方向。

经典流水线的做法是硬阈值：把 ‖x‖² 低于某个参考尺度若干倍的点**直接丢掉**。这是在用
0/1 权重近似上式中的连续权重，代价有两条：
  · 丢掉的信息无法找回（估计方差变大）；
  · 阈值必须相对某个参考统计量来定，而这个参考量随工况漂移（第 4 章）。

本模块检验两个变体：
  NF-W    —— 保留 NF 门限（去掉纯噪声点），但聚类改为按能量加权；
  base-W  —— **完全不做能量门限**，只在基础单源判据之上按能量加权。
若 base-W 能与逐档最优的硬阈值打平，则"门限"这一操作整体上是可被加权替代的，
参考尺度问题也随之消失。

用法:
    ../.venv/bin/python weighted_variant.py --stage 1        # 快速判定
    ../.venv/bin/python weighted_variant.py --stage 2 --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.cluster import KMeans

import data as D
import metrics as MT
import realdata as RD
from baselines import (canonical_sign, kmeans_sphere, l1_recover, ssp_directions,
                       ssp_mask_from_complex, unit_cols)
from nfr import nfr_mask, threshold_ratio

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
REAL_CFGS = [
    ("R1_res", "w256", dict(win=256, n_sources=4, snr_db=20.0)),
    ("R1_res", "w512", dict(win=512, n_sources=4, snr_db=20.0)),
    ("R1_res", "w1024", dict(win=1024, n_sources=4, snr_db=20.0)),
    ("R1_res", "w2048", dict(win=2048, n_sources=4, snr_db=20.0)),
    ("R2_n", "N3", dict(win=1024, n_sources=3, snr_db=20.0)),
    ("R2_n", "N5", dict(win=1024, n_sources=5, snr_db=20.0)),
    ("R2_n", "N6", dict(win=1024, n_sources=6, snr_db=20.0)),
    ("R3_snr", "snr00", dict(win=1024, n_sources=4, snr_db=0.0)),
    ("R3_snr", "snr10", dict(win=1024, n_sources=4, snr_db=10.0)),
    ("R3_snr", "snr30", dict(win=1024, n_sources=4, snr_db=30.0)),
    ("R3_snr", "snr40", dict(win=1024, n_sources=4, snr_db=40.0)),
    ("R4_dense", "d1", dict(win=1024, n_sources=4, snr_db=20.0, n_dense=1)),
    ("R4_dense", "d2", dict(win=1024, n_sources=4, snr_db=20.0, n_dense=2)),
    ("R5_noise", "babble", dict(win=1024, n_sources=4, snr_db=20.0, noise="babble")),
    ("R5_noise", "clean", dict(win=1024, n_sources=4, snr_db=None, noise=None)),
]
MED_GRID = [0.02, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
_CORPUS = None


# ==========================================================================
# 加权球面 K-means
# ==========================================================================

def principal_direction_weighted(sel: np.ndarray, w: np.ndarray) -> np.ndarray:
    """加权主方向：以 w 加权的散点矩阵 (Σ w x xᵀ) 的主特征向量。"""
    G = (sel * w) @ sel.T
    ev, evec = np.linalg.eigh(G)
    return evec[:, -1]


def kmeans_sphere_weighted(U: np.ndarray, k: int, rng: np.random.Generator,
                           w: np.ndarray | None = None, n_init: int = 10,
                           n_iter_center: int = 1) -> np.ndarray | None:
    """球面 |cos| 度量下的**加权** K-means（外积特征空间中的加权欧氏 K-means）。

    w 为每条方向的权重；w=None 时与 ``kmeans_sphere`` 等价（用于回归测试）。
    分配步由 sklearn 的加权 KMeans 完成，中心步用加权主方向（而非简单平均）更新。
    """
    M = U.shape[0]
    if U.shape[1] < k:
        return None
    if w is None:
        w = np.ones(U.shape[1])
    w = np.maximum(np.asarray(w, dtype=float), 1e-300)

    feats = np.einsum("ik,jk->ijk", U, U).reshape(M * M, -1).T
    km = KMeans(n_clusters=k, n_init=n_init, random_state=int(rng.integers(1 << 31)))
    lab = km.fit_predict(feats, sample_weight=w)

    C = np.zeros((M, k))
    for c in range(k):
        m = lab == c
        C[:, c] = (principal_direction_weighted(U[:, m], w[m]) if m.sum() > 0
                   else rng.standard_normal(M))
    return unit_cols(canonical_sign(C))


# ==========================================================================
# 掩码与权重
# ==========================================================================

def base_mask(prob):
    return ssp_mask_from_complex(prob["X_tf"], thr_cos=0.98, thr_energy_ratio=0.0)


def weights_for(prob, mask, kind):
    """给定掩码下的权重向量。"""
    e = np.sum(np.abs(prob["X_tf"]) ** 2, axis=0).ravel()[mask.ravel()]
    if kind == "e":
        return e
    if kind == "e_cos4":                      # 叠加共线性的置信度
        Xr, Xi = prob["X_tf"].real, prob["X_tf"].imag
        nr = np.linalg.norm(Xr, axis=0).ravel()[mask.ravel()]
        ni = np.linalg.norm(Xi, axis=0).ravel()[mask.ravel()]
        cos = np.abs(np.sum(Xr * Xi, axis=0).ravel()[mask.ravel()]
                     / np.maximum(nr * ni, 1e-300))
        return e * np.maximum(cos, 1e-6) ** 4
    raise ValueError(kind)


# ==========================================================================
# 评测
# ==========================================================================

def eval_synth_one(seed):
    """合成阶梯的**单个**问题实例：比较硬门限族与加权族。"""
    out = []
    for key in LADDER:
        p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + seed)
        n, rng0 = p["n"], np.random.default_rng(seed)
        bm = base_mask(p)
        mk_nf, dg = nfr_mask(p["X_tf"], p["m"], alpha=1e-4)
        med_e = float(np.median(np.sum(np.abs(p["X_tf"]) ** 2, axis=0).ravel()))
        cand = {
            "NF":        (mk_nf, None),
            "NF-W":      (mk_nf, "e"),
            "NF-Wc":     (mk_nf, "e_cos4"),
            "base-W":    (bm, "e"),
            "base-Wc":   (bm, "e_cos4"),
            "SCA-med":   (ssp_mask_from_complex(p["X_tf"], thr_cos=0.98, thr_energy_ratio=0.02), None),
            "SCA-max":   (ssp_mask_from_complex(p["X_tf"], thr_cos=0.98, thr_energy_ratio=0.0), "MAXREF"),
        }
        for name, (mk, wk) in cand.items():
            if wk == "MAXREF":
                e_full = np.sum(np.abs(p["X_tf"]) ** 2, axis=0)
                mk = mk & (e_full >= 0.05 * e_full.max())
                wk = None
            rng = np.random.default_rng(seed)
            U = ssp_directions(p["X_tf"], mk)
            if U.shape[1] < n:
                out.append(dict(dom="synth", case=key, seed=seed, method=name,
                                A=np.nan, SDR=np.nan, n_keep=int(mk.sum())))
                continue
            w = None if wk is None else weights_for(p, mk, wk)
            A = (kmeans_sphere(U, n, rng) if w is None
                 else kmeans_sphere_weighted(U, n, rng, w))
            if A is None:
                continue
            S = l1_recover(A, p["X_all"])
            out.append(dict(dom="synth", case=key, seed=seed, method=name,
                            A=float(MT.mixing_matrix_angle_error_deg(p["A"], A)),
                            SDR=float(MT.evaluate_sources(p["S_all"], S)["SDR"]),
                            n_keep=int(mk.sum()),
                            t_equiv=float(threshold_ratio(p["m"], 1e-4) * dg["nu_hat"]
                                          / max(med_e, 1e-30))))
        # 逐档硬阈值最优（中位数族）
        best = min(MT.mixing_matrix_angle_error_deg(
            p["A"], kmeans_sphere(ssp_directions(
                p["X_tf"], ssp_mask_from_complex(p["X_tf"], thr_cos=0.98, thr_energy_ratio=te)),
                n, np.random.default_rng(seed))) for te in MED_GRID)
        out.append(dict(dom="synth", case=key, seed=seed, method="hard-opt",
                        A=float(best), SDR=float("nan"), n_keep=0))
    return out


def eval_synth(n_seeds):
    """合成阶梯：对 n_seeds 个**不同**问题实例重复，返回全部记录。

    修正记录：早期版本写成 eval_synth(a.seeds)，把 8 当成**单个种子值**传进来，
    于是只评测了 seed=1000+8=1008 这一个问题实例——合成对照实为单次实现，
    而日志/输出里却打着 "seeds=8"。主实验 experiments_nfr.py 的约定是
    for s in range(n_seeds) 且每个种子生成一个独立问题，此处对齐。
    """
    out = []
    for seed in range(n_seeds):
        out += eval_synth_one(seed)
    return out


def eval_real(job):
    global _CORPUS
    corpus, exp, tag, kw, seeds, dur = job
    if _CORPUS is None:
        _CORPUS = RD.scan_corpus(corpus)
    case = f"{exp}:{tag}"
    out = []
    for s in range(seeds):
        pool = RD.load_pool(_CORPUS, n_files=40, dur_s=dur, seed=s,
                            min_speakers=kw["n_sources"] - kw.get("n_dense", 0))
        p = RD.make_real_problem(pool, m_obs=2, dur_s=dur, seed=1000 + s,
                                 hop=kw.get("win", 1024) // 2, **kw)
        n = p["n"]
        bm = base_mask(p)
        mk_nf, dg = nfr_mask(p["X_tf"], p["m"], alpha=1e-4)
        cand = {"NF": (mk_nf, None), "NF-W": (mk_nf, "e"), "NF-Wc": (mk_nf, "e_cos4"),
                "base-W": (bm, "e"), "base-Wc": (bm, "e_cos4"),
                "SCA-med": (ssp_mask_from_complex(p["X_tf"], thr_cos=0.98,
                                                 thr_energy_ratio=0.02), None)}
        for name, (mk, wk) in cand.items():
            U = ssp_directions(p["X_tf"], mk)
            if U.shape[1] < n:
                continue
            w = None if wk is None else weights_for(p, mk, wk)
            A = (kmeans_sphere(U, n, np.random.default_rng(s)) if w is None
                 else kmeans_sphere_weighted(U, n, np.random.default_rng(s), w))
            if A is None:
                continue
            S = l1_recover(A, p["X_all"])
            out.append(dict(dom="real", case=case, seed=s, method=name,
                            A=float(MT.mixing_matrix_angle_error_deg(p["A"], A)),
                            SDR=float(MT.evaluate_sources(p["S_all"], S)["SDR"]),
                            n_keep=int(mk.sum()),
                            gate_on=bool(dg["gate_on"])))
    return out


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--out", default="../results/weighted_variant.json")
    a = ap.parse_args()
    if a.stage == 2 and a.seeds == 5:
        a.seeds = 8

    t0 = time.perf_counter()
    recs = []
    print(f"stage={a.stage} seeds={a.seeds} dur={a.dur}")

    recs += eval_synth(a.seeds) if a.stage <= 2 else []
    cases = REAL_CFGS if a.stage == 2 else REAL_CFGS[:4] + REAL_CFGS[10:13]
    jobs = [(a.corpus, e, t, kw, a.seeds, a.dur) for e, t, kw in cases]
    if a.workers <= 1:
        for j in jobs:
            recs += eval_real(j)
    else:
        # 必须用 spawn：父进程在提交任务前已经跑过合成评测（sklearn/BLAS 线程池已初始化），
        # 此时再 fork 会继承被锁住的互斥量，子进程会永久阻塞——实测卡死 36 分钟只用 57s CPU。
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            for fu in as_completed([ex.submit(eval_real, j) for j in jobs]):
                recs += fu.result()
    print(f"共 {len(recs)} 条记录，用时 {time.perf_counter()-t0:.1f}s")

    MET = ["NF", "NF-W", "NF-Wc", "base-W", "base-Wc", "SCA-med", "SCA-max", "hard-opt"]

    def tbl(dom, cases_, metric):
        print(f"\n--- {dom} {metric} ---")
        print(f"{'case':<18s}" + "".join(f"{m:>9s}" for m in MET))
        for c in cases_:
            row = f"{c:<18s}"
            for m in MET:
                v = [r[metric] for r in recs if r["dom"] == dom and r["case"] == c
                     and r["method"] == m and np.isfinite(r[metric])]
                row += f"{np.mean(v):>9.2f}" if v else f"{'—':>9s}"
            print(row)
        print(f"{'MEAN':<18s}" + "".join(
            f"{np.mean([r[metric] for r in recs if r['dom']==dom and r['method']==m and np.isfinite(r[metric])]):>9.2f}"
            for m in MET))

    synth_cases = LADDER
    real_cases = [f"{e}:{t}" for e, t, _ in cases]
    tbl("synth", synth_cases, "A")
    tbl("synth", synth_cases, "SDR")
    tbl("real", real_cases, "A")
    tbl("real", real_cases, "SDR")

    # 相对逐档硬阈值最优的比值
    print("\n--- 相对 hard-opt 的比值（越低越好；<1 表示优于逐档最优的硬阈值）---")
    for dom, cs in [("synth", synth_cases), ("real", real_cases)]:
        print(f"  {dom}:")
        for m in ["NF", "NF-W", "NF-Wc", "base-W", "base-Wc", "SCA-med", "SCA-max"]:
            rs = []
            for c in cs:
                a1 = [r["A"] for r in recs if r["dom"] == dom and r["case"] == c
                      and r["method"] == m and np.isfinite(r["A"])]
                b1 = [r["A"] for r in recs if r["dom"] == dom and r["case"] == c
                      and r["method"] == "hard-opt" and np.isfinite(r["A"])]
                if a1 and b1:
                    rs.append(np.mean(a1) / np.mean(b1))
            if rs:
                print(f"    {m:<9s} 均值 {np.mean(rs):5.2f}  最差 {np.max(rs):5.2f}  最优 {np.min(rs):5.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"stage": a.stage, "seeds": a.seeds, "records": recs}, f)
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
