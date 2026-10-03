"""R2-3  噪声族系统性实验（审稿意见 MC5）。

把「下尾 χ² 模型在非高斯噪声下失效」从一个附加案例变成系统结论。
四类噪声，全部按**总噪声功率二阶矩匹配**（SNR 20 dB）：

  gauss      循环对称复高斯（模型假设，基准）
  laplace    复拉普拉斯（重尾但独立同分布）
  impulsive  Bernoulli-复高斯突发（20 dB 突发，ε=2%）——强脉冲性
  colored    AR(1) 沿频率相关的复高斯（有色）
  uniform    复均匀（下尾比 χ² 更轻，用来检验自检能否察觉模型失配）

报告：σ̂²/σ² 偏差与离散度、门限开启率、自检触发原因、
以及 NF / 经典默认 / 中位数 t_e=5 的角度误差。

用法：
    python3 r2_noise_family.py --seeds 8 --workers 10 --out ../results/r2_noise_family.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import data as D
import metrics as MT
from nfr import nfr_mask, point_energies, estimate_noise_power
from baselines import kmeans_sphere, ssp_directions

M_OBS, N_SRC, F, T = 2, 4, 33, 64
SNR_DB = 20.0
P_LIST = [0.05, 0.20, 0.40]
KINDS = ["gauss", "laplace", "impulsive", "colored", "uniform", "student"]


def make_noise(shape, power: float, kind: str, rng: np.random.Generator) -> np.ndarray:
    """生成噪声，二阶矩精确匹配到 power。"""
    if kind == "gauss":
        z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        z *= np.sqrt(power / 2.0)
    elif kind == "laplace":
        b = np.sqrt(power / 2.0) / np.sqrt(2.0)   # 实/虚各 Laplace(0,b)，var=2b²
        z = (rng.laplace(0.0, b, size=shape)
             + 1j * rng.laplace(0.0, b, size=shape))
    elif kind == "impulsive":
        eps, burst_db = 0.02, 20.0
        amp = np.where(rng.random(shape) < eps, np.sqrt(10.0 ** (burst_db / 10.0)), 1.0)
        z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * amp
    elif kind == "uniform":
        # 轻下尾：复均匀。二阶矩匹配，但小分位数比 χ² 大 -> σ̂² 系统性偏高
        a = math.sqrt(3.0 * power / 2.0)
        z = (rng.uniform(-a, a, size=shape) + 1j * rng.uniform(-a, a, size=shape))
    elif kind == "student":
        # 复 Student-t（ν=3）：重尾但非脉冲，与 Laplace 的区别是尾部衰减更慢、
        # 小分位数更小。用来补审稿人意欲的那一格（student-t 行）。
        nu = 3.0
        g = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        z = g / np.sqrt(rng.chisquare(nu, size=shape) / nu)
    elif kind == "colored":
        rho = 0.9
        w = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        z = np.empty_like(w)
        z[:, 0, :] = w[:, 0, :]
        for f in range(1, shape[1]):
            z[:, f, :] = rho * z[:, f - 1, :] + np.sqrt(1 - rho ** 2) * w[:, f, :]
    else:
        raise ValueError(kind)
    # 二阶矩匹配（colored 的逐频点功率分布不均，但总量匹配）
    z *= np.sqrt(power / max(float(np.mean(np.abs(z) ** 2)), 1e-300))
    return z


def base_mask(X: np.ndarray) -> np.ndarray:
    Xr, Xi = X.real, X.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    eps = 1e-15
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
    d = np.maximum(nr + ni, eps)
    return (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)


def job(j: tuple[str, float, int]) -> dict:
    kind, pratio, s = j
    rng = np.random.default_rng(7000 + s)
    A = D.gen_mixing_matrix(M_OBS, N_SRC, rng, 12.0)
    S = D.gen_tf_sources(N_SRC, F, T, "tf_sparse", pratio, rng)
    Xc = np.einsum("mn,nft->mft", A, S)
    power = float(np.mean(np.abs(Xc) ** 2)) / (10.0 ** (SNR_DB / 10.0))
    noise = make_noise((M_OBS, F, T), power, kind, rng)
    X = Xc + noise

    E = point_energies(X)
    sigma2 = float(np.mean(np.abs(noise) ** 2))
    est = estimate_noise_power(E, M_OBS)
    b = base_mask(X)
    e2 = np.sum(np.abs(X) ** 2, axis=0)
    m_nf, dg = nfr_mask(X, M_OBS, alpha=1e-4, use_self_check=True)
    med = float(np.median(E))
    m_cl = b & (e2 > 0.02 * med)
    m_t5 = b & (e2 > 5.0 * med)

    def err(mk):
        prob = {"X_tf": X, "A": A, "n": N_SRC}
        U = ssp_directions(X, mk)
        if U.shape[1] < N_SRC:
            return float("nan")
        C = kmeans_sphere(U, N_SRC, np.random.default_rng(s))
        return float(MT.mixing_matrix_angle_error_deg(A, C))

    return dict(kind=kind, p=pratio, seed=s,
                s2_ratio=float(est["s2"] / sigma2), spread=float(est["spread"]),
                gate_on=bool(dg["gate_on"]),
                spread_ok=bool(dg["spread_ok"]), retention_ok=bool(dg["retention_ok"]),
                keep_frac=float(dg["keep_frac"]), nu_over_mean=float(dg["nu_over_mean"]),
                A_nf=err(m_nf), A_classical=err(m_cl), A_te5=err(m_t5),
                med_over_mean_e=float(med / max(float(np.mean(E)), 1e-30)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="../results/r2_noise_family.json")
    a = ap.parse_args()

    jobs = [(k, p, s) for k in KINDS for p in P_LIST for s in range(a.seeds)]
    t0 = time.perf_counter()
    if a.workers <= 1:
        recs = [job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            recs = list(ex.map(job, jobs))

    print(f"\n=== 噪声族（{len(KINDS)} 类 × {len(P_LIST)} 档 × {a.seeds} seeds，"
          f"{time.perf_counter()-t0:.0f}s）===")
    print(f"{'noise':<11}{'p':>6}{'σ̂²/σ²':>10}{'spread':>9}{'门限开启':>10}"
          f"{'NF':>9}{'classical':>11}{'te5':>9}")
    print("-" * 76)
    for k in KINDS:
        for pr in P_LIST:
            sub = [r for r in recs if r["kind"] == k and r["p"] == pr]
            f = lambda x: float(np.nanmean([r[x] for r in sub]))
            print(f"{k:<11}{pr:>6.2f}{f('s2_ratio'):>10.3f}{f('spread'):>9.3f}"
                  f"{np.mean([r['gate_on'] for r in sub]):>10.2f}"
                  f"{f('A_nf'):>9.3f}{f('A_classical'):>11.3f}{f('A_te5'):>9.3f}")

    print("\n=== 自检为何触发（触发条件的平均）===")
    print(f"{'noise':<11}{'spread_ok':>11}{'retention_ok':>14}{'keep%':>9}{'ν̂/mean(e)':>12}")
    for k in KINDS:
        sub = [r for r in recs if r["kind"] == k]
        print(f"{k:<11}{np.mean([r['spread_ok'] for r in sub]):>11.2f}"
              f"{np.mean([r['retention_ok'] for r in sub]):>14.2f}"
              f"{np.mean([r['keep_frac'] for r in sub])*100:>9.2f}"
              f"{np.mean([r['nu_over_mean'] for r in sub]):>12.3f}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": vars(a), "kinds": KINDS, "p_list": P_LIST,
                   "snr_db": SNR_DB, "records": recs}, open(a.out, "w"))
        print(f"\n已写 {a.out}（{len(recs)} 条）")


if __name__ == "__main__":
    main()
