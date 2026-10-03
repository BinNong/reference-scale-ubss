"""纯度加权：把质量权重从"能量"升级为"能量 × 与方向估计的一致性"。

问题
----
硬门限（第 5 章）与能量加权（第 9.5 节）都只用**能量**判断一个候选点值不值得信。
这在真实语音上有效（Table 16：能量越高，单源比例越高、方向越准），但在 i.i.d. 合成
模型上会反噬：稠密档里**能量最高的点优先是多源叠加**（p=0.40 时单源比例随能量
从 0.52 降到 0.29），把它们的权重抬高等于把簇中心往错误方向拉。

理论上正确的权重
----------------
设 x_l 是真实单源点 x = a·u + n（u 为单位方向、n 为噪声），则方向估计的角误差方差
≈ σ²/‖a‖²，即关于 u 的 Fisher 信息 ∝ ‖x‖²/σ²。因此最大似然权重为

    w_l  ∝  P(该点是单源 | x_l) · ‖x_l‖²/σ²

第二因子就是能量（已有）；第一因子是"纯度"，需要盲估计。

盲估计纯度
----------
**两遍方案**：先用能量权重聚类得到中心 Ĉ，再令

    ρ_l = max_k |cos(x_l, ĉ_k)|²  ∈ [0,1]

一个真正的单源点严格落在某条混合方向上，故 ρ→1；而两源叠加点位于两条真实方向
**之间**，对任何中心都不对齐，ρ 明显小于 1。于是用 w ← w · ρ^p 重新聚类。

该量完全盲（只用观测与自己的估计），不需要真值，也不需要噪声功率；代价是聚类跑两遍。

用法:
    ../.venv/bin/python purity_weight.py --stage 1 --seeds 8 --workers 6
    ../.venv/bin/python purity_weight.py --stage 2 --seeds 8 --workers 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import data as D
import metrics as MT
import realdata as RD
from baselines import (canonical_sign, kmeans_sphere, l1_recover, ssp_directions,
                       ssp_mask_from_complex)
from nfr import nfr_mask
from weighted_variant import base_mask, kmeans_sphere_weighted, weights_for

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
_CORPUS = None


# ==========================================================================

def purity_two_pass(U, k, rng, w_base, p_purity, n_init=10, n_pass=2):
    """能量权重 → 纯度权重 的两遍加权球面 K-means。

    第一遍用 w_base 求中心；其后每遍令 w ← w_base · ρ^p，ρ 为该点与最近中心的
    一致性 max_k |cos(x, c_k)|²。ρ 度量纯度：单源点 ρ→1，多源叠加点 ρ<1。
    """
    C = kmeans_sphere_weighted(U, k, rng, w_base, n_init=n_init)
    if C is None:
        return None
    for _ in range(max(0, n_pass - 1)):
        rho = np.abs(C.T @ U).max(axis=0)          # (n,)
        w = w_base * np.maximum(rho, 1e-12) ** p_purity
        C2 = kmeans_sphere_weighted(U, k, rng, w, n_init=n_init)
        if C2 is None:
            break
        C = C2
    return C


VARIANTS = [
    ("NF", None, None),            # 无权重（基线）
    ("NF+e", "e", 0),              # 能量权重
    ("NF+e·cos4", "e_cos4", 0),    # 能量 × |cos(Re,Im)|⁴
    ("NF+e·ρ2", "e", 2),           # 能量 × 纯度²（两遍）
    ("NF+e·ρ4", "e", 4),
    ("NF+e·ρ8", "e", 8),
]


def eval_columns(prob, seed, tag_prefix=""):
    """返回该问题下各变体的记录。"""
    n = prob["n"]
    mk_nf, dg = nfr_mask(prob["X_tf"], prob["m"], alpha=1e-4)
    bm = base_mask(prob)
    med_e = float(np.median(np.sum(np.abs(prob["X_tf"]) ** 2, axis=0).ravel()))
    out = []
    for name, wkind, p_pur in VARIANTS:
        U = ssp_directions(prob["X_tf"], mk_nf)
        if U.shape[1] < n:
            continue
        w = None if wkind is None else weights_for(prob, mk_nf, wkind)
        rng = np.random.default_rng(seed)
        if w is None:
            A = kmeans_sphere(U, n, rng)
        elif p_pur == 0:
            A = kmeans_sphere_weighted(U, n, rng, w)
        else:
            A = purity_two_pass(U, n, rng, w, p_pur)
        if A is None:
            continue
        S = l1_recover(A, prob["X_all"])
        out.append(dict(case=prob["cfg_key"], seed=seed, method=name,
                        A=float(MT.mixing_matrix_angle_error_deg(prob["A"], A)),
                        SDR=float(MT.evaluate_sources(prob["S_all"], S)["SDR"]),
                        n_keep=int(mk_nf.sum()), gate_on=bool(dg["gate_on"])))
    # 无门限 + 纯度权重（检验门限是否仍必要）
    for name, wkind, p_pur in [("base+e", "e", 0), ("base+e·ρ4", "e", 4)]:
        U = ssp_directions(prob["X_tf"], bm)
        if U.shape[1] < n:
            continue
        w = weights_for(prob, bm, wkind)
        rng = np.random.default_rng(seed)
        A = (kmeans_sphere_weighted(U, n, rng, w) if p_pur == 0
             else purity_two_pass(U, n, rng, w, p_pur))
        if A is None:
            continue
        S = l1_recover(A, prob["X_all"])
        out.append(dict(case=prob["cfg_key"], seed=seed, method=name,
                        A=float(MT.mixing_matrix_angle_error_deg(prob["A"], A)),
                        SDR=float(MT.evaluate_sources(prob["S_all"], S)["SDR"]),
                        n_keep=int(bm.sum()), gate_on=True))
    return out, dg, med_e


def eval_synth(n_seeds):
    """合成阶梯：对 n_seeds 个**不同**问题实例重复。

    修正记录：早期版本写成 eval_synth(a.seeds)，把 8 当成**单个种子值**传进来，
    只评测了 seed=1000+8=1008 这一个问题实例，而日志里打着 "seeds=8"。
    现按主实验 experiments_nfr.py 的约定改为 for s in range(n_seeds)。
    """
    recs = []
    for key in LADDER:
        for s in range(n_seeds):
            p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000 + s)
            r, _, _ = eval_columns(p, s)
            for x in r:
                x["dom"] = "synth"; x["case"] = key
            recs += r
    return recs


def eval_real(job):
    global _CORPUS
    corpus, exp, tag, kw, seeds, dur = job
    if _CORPUS is None:
        _CORPUS = RD.scan_corpus(corpus)
    case = f"{exp}:{tag}"
    recs = []
    for s in range(seeds):
        pool = RD.load_pool(_CORPUS, n_files=40, dur_s=dur, seed=s,
                            min_speakers=kw["n_sources"] - kw.get("n_dense", 0))
        p = RD.make_real_problem(pool, m_obs=2, dur_s=dur, seed=1000 + s,
                                 hop=kw.get("win", 1024) // 2, **kw)
        r, _, _ = eval_columns(p, s)
        for x in r:
            x["dom"] = "real"; x["case"] = case
        recs += r
    return recs


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--out", default="../results/purity_weight.json")
    a = ap.parse_args()

    t0 = time.perf_counter()
    recs = []
    print(f"stage={a.stage} seeds={a.seeds}")
    if a.stage <= 1:
        recs += eval_synth(a.seeds)
    cases = REAL_CFGS if a.stage == 2 else REAL_CFGS[:4] + REAL_CFGS[11:13]
    jobs = [(a.corpus, e, t, kw, a.seeds, a.dur) for e, t, kw in cases]
    ctx = mp.get_context("spawn")            # 父进程已跑过 sklearn，必须 spawn
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        for fu in as_completed([ex.submit(eval_real, j) for j in jobs]):
            recs += fu.result()
    print(f"共 {len(recs)} 条记录，用时 {time.perf_counter()-t0:.1f}s")

    MET = [v[0] for v in VARIANTS] + ["base+e", "base+e·ρ4"]
    for dom, cs in [("synth", LADDER),
                    ("real", [f"{e}:{t}" for e, t, _ in cases])]:
        for metric in ["A", "SDR"]:
            print(f"\n--- {dom} {metric} ---")
            print(f"{'case':<18s}" + "".join(f"{m:>11s}" for m in MET))
            for c in cs:
                row = f"{c:<18s}"
                for m in MET:
                    v = [r[metric] for r in recs if r["dom"] == dom and r["case"] == c
                         and r["method"] == m and np.isfinite(r[metric])]
                    row += f"{np.mean(v):>11.2f}" if v else f"{'—':>11s}"
                print(row)
            print(f"{'MEAN':<18s}" + "".join(
                f"{np.mean([r[metric] for r in recs if r['dom']==dom and r['method']==m and np.isfinite(r[metric])]):>11.2f}"
                for m in MET))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"stage": a.stage, "seeds": a.seeds, "records": recs}, f)
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
