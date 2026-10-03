#!/usr/bin/env python3
"""R4 / Minor 10：鲁棒（MAD 类）噪声底估计能否修复 babble？

审稿人问：NF-SSP 与已知的鲁棒噪声估计器（如 median absolute deviation 类）结合，
是否能修复 babble 噪声场景——目前该场景只被自检"识别"而无解法。

比较三个噪声底估计量（都用同一 τ 与同一 base 掩码，只换 ν̂）：
  paper  本文的 χ² 分位数反演（下尾 2% 的次序统计量 + 中位数合并）
  mad    实部 MAD：σ̂² = 2·(MAD(Re X)/0.6745)²
  q25    实部 25% 分位：σ̂² = 2·(q25(|Re X|)/0.6745)²
MAD 与 q25 是文献里最常见的鲁棒尺度估计，二者在纯高斯下与 σ 一致、对重尾不敏感。

同时测一个**可证的**诊断量：噪声在时频平面上的平稳性
（逐频点能量的中位数在频率间的离散度），用来区分"估计量不够稳健"与
"前提（噪声在 TF 上平稳）本身不成立"。

用法（服务器 src 目录）:
  ../.venv/bin/python r4_robust_floor.py --seeds 8 --out ../results/r4_robust_floor.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import metrics as MT
import realdata as RD
from baselines import kmeans_sphere, ssp_directions
from nfr import nfr_mask, point_energies, threshold_ratio
from r2_noise_family import make_noise

M_OBS, N_SRC, F, T = 2, 4, 33, 64
SNR_DB = 20.0
KINDS = ["gauss", "laplace", "impulsive", "colored", "uniform"]
P_LIST = [0.05, 0.40]
ALPHA = 1e-4
MAD_C = 0.6745      # |N(0,1)| 的中位数
Q25_C = 0.3186      # |N(0,1)| 的 25% 分位（二者不可混用！）


# ----------------------------------------------------------------- 估计量
def est_paper(X: np.ndarray, m: int) -> float:
    """本文的 χ² 分位数反演（返回 ν̂ = M σ̂²）。"""
    from nfr import estimate_noise_power
    return float(m * estimate_noise_power(point_energies(X), m)["s2"])


def est_mad(X: np.ndarray, m: int) -> float:
    r = X.real.ravel()
    mad = float(np.median(np.abs(r - np.median(r))))
    s2 = 2.0 * (mad / MAD_C) ** 2
    return float(m * s2)


def est_q25(X: np.ndarray, m: int) -> float:
    a = np.abs(X.real.ravel())
    q = float(np.quantile(a, 0.25))
    s2 = 2.0 * (q / Q25_C) ** 2      # 半正态的 25% 分位 = 0.3186·s
    return float(m * s2)


ESTS = {"paper": est_paper, "mad": est_mad, "q25": est_q25}


def base_mask(X: np.ndarray) -> np.ndarray:
    Xr, Xi = X.real, X.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
    d = np.maximum(nr + ni, 1e-15)
    return (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)


def tf_stationarity(Z: np.ndarray) -> float:
    """逐频点能量的时间中位数，在频率维上的 IQR/中位数。

    对**噪声本身**（而不是观测量）计算时，它是"噪声在每个 TF 点功率相同"这一前提的
    直接可证诊断：平稳白噪声下很小，按频率有色或按时间突发时很大。
    对观测量计算则被信号频谱主导（真实语音无论加什么噪声都会给出 5 左右），
    所以两列都报，但结论只依据 noise 列。
    """
    e = np.sum(np.abs(Z) ** 2, axis=0)               # F×T
    per_f = np.median(e, axis=1)
    med = float(np.median(per_f))
    return float((np.quantile(per_f, 0.75) - np.quantile(per_f, 0.25)) / max(med, 1e-30))


# ----------------------------------------------------------------- 作业
def job_synth(j):
    kind, pratio, s = j
    rng = np.random.default_rng(7000 + s)
    A = D.gen_mixing_matrix(M_OBS, N_SRC, rng, 12.0)
    S = D.gen_tf_sources(N_SRC, F, T, "tf_sparse", pratio, rng)
    Xc = np.einsum("mn,nft->mft", A, S)
    power = float(np.mean(np.abs(Xc) ** 2)) / (10.0 ** (SNR_DB / 10.0))
    noise = make_noise((M_OBS, F, T), power, kind, rng)
    X = Xc + noise
    s2_true = float(np.mean(np.abs(noise) ** 2))
    b = base_mask(X)
    e = np.sum(np.abs(X) ** 2, axis=0)

    rec = dict(dom="synth", kind=kind, p=pratio, seed=s,
               s2_true=s2_true, stat_x=tf_stationarity(X),
               stat_noise=tf_stationarity(noise),
               med_over_nu=float(np.median(e) / (M_OBS * s2_true)))
    for name, fn in ESTS.items():
        nu = fn(X, M_OBS)
        mk = b & (e > threshold_ratio(M_OBS, ALPHA) * nu)
        rec[f"nu_{name}"] = float(nu)
        rec[f"ratio_{name}"] = float(nu / (M_OBS * s2_true))
        rec[f"keep_{name}"] = float(mk.sum() / max(b.sum(), 1))
        U = ssp_directions(X, mk)
        rec[f"A_{name}"] = (float(MT.mixing_matrix_angle_error_deg(
            A, kmeans_sphere(U, N_SRC, np.random.default_rng(s))))
            if U.shape[1] >= N_SRC else float("nan"))
    return rec


def job_real(j):
    win, noise_kind, s, n_files, dur = j
    corpus = RD.scan_corpus("../data/ls_corpus")
    pool = RD.load_pool(corpus, n_files=n_files, dur_s=dur, seed=s, min_speakers=4)
    kw = dict(win=win, n_sources=4, hop=win // 2)
    if noise_kind is None:
        prob = RD.make_real_problem(pool, m_obs=M_OBS, dur_s=dur, seed=1000 + s,
                                    snr_db=None, noise=None, **kw)
    else:
        prob = RD.make_real_problem(pool, m_obs=M_OBS, dur_s=dur, seed=1000 + s,
                                    snr_db=SNR_DB, noise=noise_kind, **kw)
    X, A = prob["X_tf"], prob["A"]
    noise_tf = prob.get("noise_tf")
    s2_true = float(np.mean(np.abs(noise_tf) ** 2)) if noise_tf is not None else float("nan")
    b = base_mask(X)
    e = np.sum(np.abs(X) ** 2, axis=0)
    rec = dict(dom="real", kind=str(noise_kind), win=win, seed=s,
               s2_true=s2_true, stat_x=tf_stationarity(X),
               stat_noise=(tf_stationarity(noise_tf) if noise_tf is not None else float("nan")))
    for name, fn in ESTS.items():
        nu = fn(X, M_OBS)
        mk = b & (e > threshold_ratio(M_OBS, ALPHA) * nu)
        rec[f"nu_{name}"] = float(nu)
        rec[f"ratio_{name}"] = (float(nu / (M_OBS * s2_true))
                                if np.isfinite(s2_true) and s2_true > 0 else float("nan"))
        rec[f"keep_{name}"] = float(mk.sum() / max(b.sum(), 1))
        U = ssp_directions(X, mk)
        rec[f"A_{name}"] = (float(MT.mixing_matrix_angle_error_deg(
            A, kmeans_sphere(U, 4, np.random.default_rng(s))))
            if U.shape[1] >= 4 else float("nan"))
    # 自检是否触发（用本文估计量）
    _, dg = nfr_mask(X, M_OBS, alpha=ALPHA, use_self_check=True)
    rec["gate_on"] = bool(dg["gate_on"])
    rec["spread"] = float(dg["spread"])
    return rec


def selftest(reps: int = 40, tol: float = 0.04, verbose: bool = True) -> bool:
    """纯高斯噪声（无信号）下三个估计量都必须回到 σ²。

    单次抽样的分位数误差本身就有 3–6%（n=4224 时），所以必须对 **reps 次独立抽样取均值**
    再判——第一次写成了单次抽样 + 6% 容差，结果把一个正确的估计量判成常数写错。
    """
    rng = np.random.default_rng(0)
    sig2 = 0.01
    acc = {k: 0.0 for k in ESTS}
    for _ in range(reps):
        X = (rng.standard_normal((M_OBS, F, T)) + 1j * rng.standard_normal((M_OBS, F, T)))
        X *= np.sqrt(sig2 / 2.0)
        for name, fn in ESTS.items():
            acc[name] += fn(X, M_OBS) / (M_OBS * sig2)
    ok = True
    if verbose:
        print(f"=== 纯噪声自检（无信号，σ²=0.01；{reps} 次抽样取均值，容差 {tol:.0%}）===")
    for name, v in acc.items():
        r = v / reps
        flag = "" if abs(r - 1.0) < tol else "   <-- 常数可能写错"
        if flag:
            ok = False
        if verbose:
            print(f"  {name:<7} ratio = {r:.4f}{flag}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--real-seeds", type=int, default=4)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", default="../results/r4_robust_floor.json")
    a = ap.parse_args()

    if not selftest():
        raise SystemExit("自检未通过：估计量常数写错，先修再跑")

    ctx = mp.get_context("spawn")
    recs = []
    jobs = [(k, p, s) for k in KINDS for p in P_LIST for s in range(a.seeds)]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 4),
                             mp_context=ctx) as ex:
        recs += list(ex.map(job_synth, jobs))

    print("=== 合成：三个估计量（σ̂²/σ² 与下游角度误差）===")
    print(f"{'noise':<11}{'p':>5}{'stat_ns':>9}"
          + "".join(f"{n+'.ratio':>11}{n+'.A':>9}{n+'.keep':>9}" for n in ESTS))
    for k in KINDS:
        for p in P_LIST:
            sub = [r for r in recs if r["kind"] == k and r["p"] == p]
            line = f"{k:<11}{p:>5.2f}{np.mean([r['stat_noise'] for r in sub]):>9.2f}"
            for n in ESTS:
                line += (f"{np.mean([r[f'ratio_{n}'] for r in sub]):>11.3f}"
                         f"{np.nanmean([r[f'A_{n}'] for r in sub]):>9.2f}"
                         f"{np.mean([r[f'keep_{n}'] for r in sub]):>9.3f}")
            print(line)

    if a.real:
        print("\n（真实语音部分开始，需要语料与 soundfile）", flush=True)
        rjobs = [(1024, nk, s, 40, 6.0) for nk in ["babble", "gauss", None]
                 for s in range(a.real_seeds)]
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 4),
                                 mp_context=ctx) as ex:
            recs += list(ex.map(job_real, rjobs))
        print("\n=== 真实语音：babble / gauss / 无噪声（win=1024, N=4, SNR 20 dB）===")
        print(f"{'noise':<9}{'seeds':>6}{'stat_ns':>9}{'stat_x':>8}{'on':>5}"
              + "".join(f"{n+'.ratio':>11}{n+'.A':>8}{n+'.keep':>8}" for n in ESTS))
        for nk in ["babble", "gauss", "None"]:
            sub = [r for r in recs if r["dom"] == "real" and r["kind"] == nk]
            if not sub:
                continue
            line = (f"{nk:<9}{len(sub):>6}"
                    f"{np.nanmean([r['stat_noise'] for r in sub]):>9.2f}"
                    f"{np.mean([r['stat_x'] for r in sub]):>8.2f}"
                    f"{sum(r['gate_on'] for r in sub):>5}")
            for n in ESTS:
                line += (f"{np.nanmean([r[f'ratio_{n}'] for r in sub]):>11.3f}"
                         f"{np.nanmean([r[f'A_{n}'] for r in sub]):>8.2f}"
                         f"{np.mean([r[f'keep_{n}'] for r in sub]):>8.3f}")
            print(line)

    json.dump(dict(alpha=ALPHA, n_seeds=a.seeds, estimators=list(ESTS),
                   records=recs), open(a.out, "w"), indent=1)
    print(f"\n已写 {a.out}（{len(recs)} 条）")


if __name__ == "__main__":
    main()
