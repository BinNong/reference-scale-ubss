"""NF-SSP 论文的完整实验矩阵（自包含、CPU 可跑、可复现）。

实验清单
--------
E1  稀疏度阶梯：NF-SSP vs 逐档 oracle 调优的经典网格 vs 经典默认阈值 vs 各基线
E2  SNR 扫描（0–40 dB）
E3  传感器数 M ∈ {2,3,4}（欠定程度）
E4  源数目 N ∈ {3,4,5,6}
E5  单一常数自校准：全部条件下不换任何参数（关键论断）
E6  即插即用：把准则套到 K-means / FCM / 势函数 / SL0 四条流水线上
E7  α 敏感性：平台有多宽
E8  结构化源（chirp / 谐波 / AM-FM / 脉冲）：真实时频结构下是否成立

用法:
    ../.venv/bin/python experiments_nfr.py --seeds 10 --out ../results/nfr_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import stats

import data as D
import metrics as MT
from baselines import (fcm_sphere, kmeans_sphere, l1_recover, omp_recover,
                       potential_function_A, run_method, ssp_directions,
                       ssp_mask_from_complex, estimate_A_sl0)
from nfr import (estimate_noise_power, mixture_median_ratio, nfr_mask,
                 point_energies, threshold_ratio)

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
P_OF = {"tf_p02": 0.02, "tf_p05": 0.05, "tf_p10": 0.10, "tf_p20": 0.20, "tf_p40": 0.40}
STRUCT = ["td_chirp", "td_sinusoid", "td_amfm", "td_impulse"]
COS_GRID = [0.98, 0.99]
EN_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
ALPHA = 1e-4


# ==========================================================================
# 统一的评测原语
# ==========================================================================

def _mk(m_obs, n_src, key, snr, seed, n_freq=33, n_frames=64):
    return D.make_problem(m_obs, n_src, n_freq, n_frames, key, snr, seed=seed)


def _classical_A(prob, c0, te, rng):
    """经典流水线：硬 cos + 硬均衡 + 硬能量（相对中位数）。"""
    mask = ssp_mask_from_complex(prob["X_tf"], thr_cos=c0, thr_energy_ratio=te)
    return kmeans_sphere(ssp_directions(prob["X_tf"], mask), prob["n"], rng)


def _nfr_A(prob, alpha, rng, use_check=True):
    mask, diag = nfr_mask(prob["X_tf"], prob["m"], alpha=alpha,
                          use_self_check=use_check)
    A = kmeans_sphere(ssp_directions(prob["X_tf"], mask), prob["n"], rng)
    return A, diag


def _record(exp, method, prob, A_hat, S_hat, dt, extra=None, seed=None):
    m = MT.evaluate_sources(prob["S_all"], S_hat)
    rec = {"experiment": exp, "method": method, "cfg_key": prob["cfg_key"],
           "seed": int(seed) if seed is not None else None,
           "seed": int(seed) if seed is not None else None,
           "snr_db": float(prob["snr_db"]) if prob["snr_db"] is not None else None,
           "m_obs": int(prob["m"]), "n_true": int(prob["n"]),
           "A_angle_deg": MT.mixing_matrix_angle_error_deg(prob["A"], A_hat),
           "SDR": m["SDR"], "SIR": m["SIR"], "time_s": dt}
    if extra:
        rec.update(extra)
    return rec


def evals_nfr(exp, prob, alpha, seed, use_check=True):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    A, diag = _nfr_A(prob, alpha, rng, use_check)
    S = l1_recover(A, prob["X_all"])
    dt = time.perf_counter() - t0
    return _record(exp, "NF-SSP", prob, A, S, dt, seed=seed, extra={"gate_on": bool(diag["gate_on"]),
                          "spread": diag.get("spread"),
                          "nu_over_mean": diag.get("nu_over_mean")})


def evals_classical(exp, prob, seed, c0=0.98, te=0.02, tag="SCA-L1"):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    A = _classical_A(prob, c0, te, rng)
    S = l1_recover(A, prob["X_all"])
    return _record(exp, tag, prob, A, S, time.perf_counter() - t0, seed=seed)


def evals_oracle_grid(exp, prob, seed):
    """逐档取网格最优——经典流水线在同等调参预算下的上界。"""
    best, best_par = None, None
    for c0 in COS_GRID:
        for te in EN_GRID:
            r = evals_classical(exp, prob, seed, c0, te, tag="SCA-L1-tuned")
            if best is None or r["A_angle_deg"] < best["A_angle_deg"]:
                best, best_par = r, {"c0": c0, "te": te}
    best.update(best_par)
    return best


def evals_baselines(exp, prob, seed):
    out = []
    for name in ["ssp_kmeans_l1", "ssp_fcm_sp", "pf_auto_l1", "sl0_l1", "duet", "oracle_a_l1"]:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        r = run_method(name, prob, seed=seed)
        out.append(_record(exp, name, prob, r["A_hat"], r["S_hat"],
                           time.perf_counter() - t0, seed=seed))
    return out


# ==========================================================================
# E1 稀疏度阶梯
# ==========================================================================

def e1(n_seeds):
    recs = []
    print("E1 稀疏度阶梯 ...", flush=True)
    for key in LADDER:
        for s in range(n_seeds):
            prob = _mk(2, 4, key, 20.0, 1000 + s)
            recs.append(evals_nfr("E1_sparsity", prob, ALPHA, s))
            recs.append(evals_classical("E1_sparsity", prob, s, 0.98, 0.02, "SCA-default"))
            recs.append(evals_classical("E1_sparsity", prob, s, 0.98, 5.0, "SCA-fixed-te5"))
            recs.append(evals_oracle_grid("E1_sparsity", prob, s))
            recs.extend(evals_baselines("E1_sparsity", prob, s))
    return recs


# ==========================================================================
# E2 SNR 扫描
# ==========================================================================

def e2(n_seeds):
    recs = []
    print("E2 SNR 扫描 ...", flush=True)
    for key in ["tf_p05", "tf_p20", "tf_gauss"]:
        for snr in [0.0, 10.0, 20.0, 30.0, 40.0]:
            for s in range(n_seeds):
                prob = _mk(2, 4, key, snr, 2000 + s)
                recs.append(evals_nfr("E2_snr", prob, ALPHA, s))
                recs.append(evals_classical("E2_snr", prob, s, 0.98, 0.02, "SCA-default"))
                recs.append(evals_oracle_grid("E2_snr", prob, s))
    return recs


# ==========================================================================
# E3 传感器数 / E4 源数目
# ==========================================================================

def e3(n_seeds):
    recs = []
    print("E3 传感器数 M ...", flush=True)
    for M in [2, 3, 4]:
        for key in LADDER:
            for s in range(n_seeds):
                prob = _mk(M, 4, key, 20.0, 3000 + s)
                recs.append(evals_nfr("E3_M", prob, ALPHA, s))
                recs.append(evals_classical("E3_M", prob, s, 0.98, 0.02, "SCA-default"))
                recs.append(evals_classical("E3_M", prob, s, 0.98, 5.0, "SCA-fixed-te5"))
    return recs


def e4(n_seeds):
    recs = []
    print("E4 源数目 N ...", flush=True)
    for N in [3, 4, 5, 6]:
        for key in ["tf_p05", "tf_p20"]:
            for s in range(n_seeds):
                prob = _mk(2, N, key, 20.0, 4000 + s)
                recs.append(evals_nfr("E4_N", prob, ALPHA, s))
                recs.append(evals_classical("E4_N", prob, s, 0.98, 0.02, "SCA-default"))
                recs.append(evals_classical("E4_N", prob, s, 0.98, 5.0, "SCA-fixed-te5"))
    return recs


# ==========================================================================
# E6 即插即用：把同一准则套到不同聚类器上
# ==========================================================================

def _pipeline_A(prob, mask, clusterer, rng):
    U = ssp_directions(prob["X_tf"], mask)
    if clusterer == "kmeans":
        return kmeans_sphere(U, prob["n"], rng)
    if clusterer == "fcm":
        return fcm_sphere(U, prob["n"], rng)
    if clusterer == "pf":
        return potential_function_A(prob["X_tf"], 1440, None, 0.2)
    if clusterer == "sl0":
        return estimate_A_sl0(prob["X_tf"], prob["n"], rng)
    raise ValueError(clusterer)


def e6(n_seeds):
    recs = []
    print("E6 即插即用 ...", flush=True)
    for pipe in ["kmeans", "fcm", "pf", "sl0"]:
        for key in LADDER:
            for s in range(n_seeds):
                prob = _mk(2, 4, key, 20.0, 6000 + s)
                rng = np.random.default_rng(s)
                # 经典默认阈值
                mk_old = ssp_mask_from_complex(prob["X_tf"], thr_cos=0.98,
                                               thr_energy_ratio=0.02)
                t0 = time.perf_counter()
                A_old = _pipeline_A(prob, mk_old, pipe, rng)
                S_old = omp_recover(A_old, prob["X_all"]) if pipe == "fcm" else l1_recover(A_old, prob["X_all"])
                recs.append(_record("E6_plugin", f"{pipe}-classical", prob, A_old, S_old,
                                    time.perf_counter() - t0, seed=s))
                # 换成 NF 准则
                mk_new, dg = nfr_mask(prob["X_tf"], 2, alpha=ALPHA)
                t0 = time.perf_counter()
                A_new = _pipeline_A(prob, mk_new, pipe, rng)
                S_new = omp_recover(A_new, prob["X_all"]) if pipe == "fcm" else l1_recover(A_new, prob["X_all"])
                recs.append(_record("E6_plugin", f"{pipe}-NF", prob, A_new, S_new,
                                    time.perf_counter() - t0, seed=s, extra={"gate_on": bool(dg["gate_on"])}))
    return recs


# ==========================================================================
# E7 α 敏感性
# ==========================================================================

def e7(n_seeds):
    recs = []
    print("E7 alpha 敏感性 ...", flush=True)
    for alpha in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        for key in LADDER:
            for s in range(n_seeds):
                prob = _mk(2, 4, key, 20.0, 7000 + s)
                recs.append(evals_nfr(f"E7_alpha{alpha:.0e}", prob, alpha, s))
    return recs


# ==========================================================================
# E8 结构化源
# ==========================================================================

def e8(n_seeds):
    recs = []
    print("E8 结构化源 ...", flush=True)
    for key in STRUCT:
        for s in range(n_seeds):
            prob = _mk(2, 4, key, 20.0, 8000 + s)
            recs.append(evals_nfr("E8_structured", prob, ALPHA, s))
            recs.append(evals_classical("E8_structured", prob, s, 0.98, 0.02, "SCA-default"))
            recs.append(evals_classical("E8_structured", prob, s, 0.98, 5.0, "SCA-fixed-te5"))
            recs.append(evals_classical("E8_structured", prob, s, 0.99, 0.5, "SCA-tuned-ish"))
    return recs


# ==========================================================================
# 汇总
# ==========================================================================

def ms(recs, exp, method, cfg=None, **filt):
    v = []
    for r in recs:
        if r["experiment"] != exp or r["method"] != method:
            continue
        if cfg is not None and r["cfg_key"] != cfg:
            continue
        if any(abs(r[k] - val) > 1e-9 for k, val in filt.items() if isinstance(val, float)):
            continue
        if any(r[k] != val for k, val in filt.items() if not isinstance(val, float)):
            continue
        v.append(r["A_angle_deg"])
    a = np.asarray(v, float)
    return (float(np.mean(a)), float(np.std(a, ddof=1)) if a.size > 1 else 0.0, int(a.size))


def paired(recs, exp, m1, m2, cfg=None, metric="A_angle_deg"):
    d = {}
    for r in recs:
        if r["experiment"] != exp:
            continue
        if cfg is not None and r["cfg_key"] != cfg:
            continue
        if r["method"] not in (m1, m2):
            continue
        d.setdefault((r["cfg_key"], r["snr_db"], r["n_true"], r["m_obs"], r["seed"]), {})[r["method"]] = r[metric]
    a, b = [], []
    for k, v in d.items():
        if m1 in v and m2 in v:
            a.append(v[m1]); b.append(v[m2])
    if len(a) < 3:
        return float("nan"), float("nan")
    if np.allclose(a, b):
        return float("nan"), 1.0
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def mean_of(recs, exp, method, metric="A_angle_deg", **filt):
    """按实验/方法/附加过滤取均值；无数据返回 nan（便于 --only 分步跑）。"""
    v = []
    for r in recs:
        if r["experiment"] != exp or r["method"] != method:
            continue
        ok = True
        for k, val in filt.items():
            if isinstance(val, float):
                if not np.isfinite(r.get(k, np.nan)) or abs(r[k] - val) > 1e-9:
                    ok = False; break
            elif r.get(k) != val:
                ok = False; break
        if ok:
            v.append(r[metric])
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="../results/nfr_results.json")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    t_start = time.perf_counter()
    recs = []
    todo = a.only.split(",") if a.only else ["e1", "e2", "e3", "e4", "e6", "e7", "e8"]
    for tag, fn in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4),
                    ("e6", e6), ("e7", e7), ("e8", e8)]:
        if tag in todo:
            recs += fn(a.seeds)
    print(f"\n共 {len(recs)} 条记录，用时 {time.perf_counter()-t_start:.1f}s")

    have = lambda e: any(r["experiment"] == e for r in recs)

    if have("E1_sparsity"):
        print("\n=== E1  A 误差(deg)：NF-SSP vs 经典默认 vs 经典固定 te=5 vs 经典逐档最优 ===")
        print(f"  {'cfg':<10s}{'NF-SSP':>9s}{'SCA默认':>9s}{'SCA te=5':>10s}{'SCA逐档最优':>12s}{'Oracle':>8s}")
        for key in LADDER:
            m = ["NF-SSP", "SCA-default", "SCA-fixed-te5", "SCA-L1-tuned", "oracle_a_l1"]
            print(f"  {key:<10s}" + "".join(f"{mean_of(recs,'E1_sparsity',x,cfg_key=key):>10.2f}"
                                           for x in m))
        print("\n=== E1  SDR(dB) ===")
        print(f"  {'cfg':<10s}{'NF-SSP':>9s}{'SCA默认':>9s}{'SCA te=5':>10s}{'SCA逐档最优':>12s}{'Oracle':>8s}")
        for key in LADDER:
            m = ["NF-SSP", "SCA-default", "SCA-fixed-te5", "SCA-L1-tuned", "oracle_a_l1"]
            print(f"  {key:<10s}" + "".join(f"{mean_of(recs,'E1_sparsity',x,metric='SDR',cfg_key=key):>10.2f}"
                                           for x in m))
        print("\n=== E1  配对 t 检验 ===")
        for key in LADDER:
            _, p1 = paired(recs, "E1_sparsity", "NF-SSP", "SCA-default", key)
            _, p2 = paired(recs, "E1_sparsity", "NF-SSP", "SCA-fixed-te5", key)
            _, p3 = paired(recs, "E1_sparsity", "NF-SSP", "SCA-L1-tuned", key)
            print(f"  {key:<10s} vs 默认 p={p1:<9.3g} vs te=5 p={p2:<9.3g} vs 逐档最优 p={p3:.3g}")
        print("\n=== E1  运行时间 (s/问题) ===")
        for m in ["NF-SSP", "SCA-default", "SCA-fixed-te5", "SCA-L1-tuned", "pf_auto_l1", "sl0_l1"]:
            print(f"  {m:<16s}{mean_of(recs,'E1_sparsity',m,metric='time_s'):>9.4f}")

    if have("E3_M"):
        print("\n=== E3  A 误差(deg) vs 传感器数 M ===")
        print(f"  {'M':>3s}{'cfg':<10s}{'NF-SSP':>9s}{'SCA默认':>9s}{'SCA te=5':>10s}")
        for M in [2, 3, 4]:
            for key in LADDER:
                print(f"  {M:>3d}{key:<10s}"
                      + "".join(f"{mean_of(recs,'E3_M',x,cfg_key=key,m_obs=M):>10.2f}"
                                for x in ["NF-SSP", "SCA-default", "SCA-fixed-te5"]))

    if have("E4_N"):
        print("\n=== E4  A 误差(deg) vs 源数目 N ===")
        print(f"  {'N':>3s}{'cfg':<10s}{'NF-SSP':>9s}{'SCA默认':>9s}{'SCA te=5':>10s}")
        for N in [3, 4, 5, 6]:
            for key in ["tf_p05", "tf_p20"]:
                print(f"  {N:>3d}{key:<10s}"
                      + "".join(f"{mean_of(recs,'E4_N',x,cfg_key=key,n_true=N):>10.2f}"
                                for x in ["NF-SSP", "SCA-default", "SCA-fixed-te5"]))

    if have("E6_plugin"):
        print("\n=== E6  即插即用 ===")
        print("  (pf 内部自带门控、不使用掩码，故其两列相同——这本身是有效对照)")
        print(f"  {'pipeline':<10s}{'cfg':<10s}{'经典阈值':>10s}{'NF 准则':>9s}{'改善':>8s}")
        for pipe in ["kmeans", "fcm", "pf", "sl0"]:
            for key in LADDER:
                ma = mean_of(recs, "E6_plugin", f"{pipe}-classical", cfg_key=key)
                mb = mean_of(recs, "E6_plugin", f"{pipe}-NF", cfg_key=key)
                imp = (ma - mb) / ma if (np.isfinite(ma) and ma > 0) else float("nan")
                print(f"  {pipe:<10s}{key:<10s}{ma:>10.2f}{mb:>9.2f}{imp:>8.0%}")

    if any(r["experiment"].startswith("E7_alpha") for r in recs):
        print("\n=== E7  alpha 敏感性（A 误差均值 / 最差档，跨 6 档） ===")
        for alpha in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            v = [r["A_angle_deg"] for r in recs if r["experiment"] == f"E7_alpha{alpha:.0e}"]
            if v:
                print(f"  alpha={alpha:.0e}  均值 {np.mean(v):6.2f}   最差 {np.max(v):6.2f}")

    if have("E8_structured"):
        print("\n=== E8  结构化源（真实时频结构） ===")
        print(f"  {'cfg':<12s}{'NF-SSP':>9s}{'SCA默认':>9s}{'SCA te=5':>10s}{'SCA c0.99,te0.5':>16s}")
        for key in STRUCT:
            print(f"  {key:<12s}"
                  + "".join(f"{mean_of(recs,'E8_structured',x,cfg_key=key):>10.2f}"
                            for x in ["NF-SSP", "SCA-default", "SCA-fixed-te5"])
                  + f"{mean_of(recs,'E8_structured','SCA-tuned-ish',cfg_key=key):>16.2f}")

    if have("E2_snr"):
        print("\n=== E2  A 误差(deg) vs SNR ===")
        print(f"  {'cfg':<10s}{'SNR':>5s}{'NF-SSP':>9s}{'SCA默认':>9s}{'SCA逐档最优':>12s}")
        for key in ["tf_p05", "tf_p20", "tf_gauss"]:
            for snr in [0.0, 10.0, 20.0, 30.0, 40.0]:
                print(f"  {key:<10s}{snr:>5.0f}"
                      + "".join(f"{mean_of(recs,'E2_snr',x,cfg_key=key,snr_db=snr):>10.2f}"
                                for x in ["NF-SSP", "SCA-default", "SCA-L1-tuned"]))

    print("\n=== 自检 (gate_on) 触发统计 ===")
    for exp in ["E1_sparsity", "E2_snr", "E3_M", "E4_N", "E6_plugin", "E8_structured"]:
        rows = [r for r in recs if r["experiment"] == exp
                and (r["method"] == "NF-SSP" or r["method"].endswith("-NF"))]
        if not rows:
            continue
        off = [r for r in rows if not r.get("gate_on", True)]
        print(f"  {exp:<16s} 总 {len(rows):4d}  关闭 {len(off):4d} ({len(off)/len(rows):5.1%})"
              f"  涉及 {sorted({r['cfg_key'] for r in off})}")

    payload = {"n_seeds": a.seeds, "alpha": ALPHA, "n_records": len(recs),
               "records": recs, "wall_sec": time.perf_counter() - t_start}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f)
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
