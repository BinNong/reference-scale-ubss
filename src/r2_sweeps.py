"""R2-2  敏感性扫描（审稿意见 MC1/MC2/MC3 与「必做实验 2、3」）。

三个部分，可单独运行：

  eta    η ∈ {0.005 … 0.1} 扫过——27 个条件网格上统计
         假开启（该关却开）、假回退（该开却关）、平均角度误差、门限使用率。
  qmax   q_max ∈ {0.005 … 0.05}——报告 σ̂²/σ² 偏差与下游角度误差，
         并给出「安全上界」q_safe(p)（偏差 ≤20% 的最大 q 水平），
         用来回答「不知道 p 时为什么 0.02 是安全的」。
  angle  最小列夹角 ∈ {5,10,12,20,30,45}°——量化 Proposition 6 的
         「活跃列正交」近似误差，以及下游角度误差随夹角的变化。

用法：
    python3 r2_sweeps.py --part eta   --out ../results/r2_eta_sweep.json
    python3 r2_sweeps.py --part qmax  --out ../results/r2_qmax_sweep.json
    python3 r2_sweeps.py --part angle --out ../results/r2_angle_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
from scipy.stats import chi2

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from nfr import nfr_mask, point_energies, threshold_ratio, estimate_noise_power

# --------------------------------------------------------------------------
# 共用
# --------------------------------------------------------------------------
M_OBS, N_SRC, NF_FREQ, NF_FRAME = 2, 4, 33, 64
ALPHA = 1e-4
GUARD_CASES = [(k, 20.0) for k in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]]
GUARD_CASES += [(k, s) for k in ["tf_p05", "tf_p20", "tf_gauss"] for s in [0.0, 5.0, 10.0, 30.0, 40.0]]
GUARD_CASES += [(k, 20.0) for k in ["td_chirp", "td_sinusoid", "td_amfm", "td_impulse"]]
GUARD_CASES += [(k, 20.0) for k in ["tf_b06", "tf_lap"]]

ETAS = [0.005, 0.01, 0.02, 0.03, 0.035, 0.05, 0.075, 0.1]
QMAXES = [0.005, 0.01, 0.02, 0.03, 0.05]
ANGLES = [5.0, 10.0, 12.0, 20.0, 30.0, 45.0]


def base_mask(X_tf: np.ndarray) -> np.ndarray:
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    eps = 1e-15
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
    d = np.maximum(nr + ni, eps)
    return (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)


def kmeans_err(p: dict, mask: np.ndarray, seed: int) -> float:
    A = kmeans_sphere(ssp_directions(p["X_tf"], mask), p["n"],
                      np.random.default_rng(seed))
    return float(MT.mixing_matrix_angle_error_deg(p["A"], A))


# ==========================================================================
# Part A — η 敏感性
# ==========================================================================
def eta_job(job: tuple[str, float, int]) -> dict:
    key, snr, s = job
    seed = (1000 if snr == 20.0 else 2000) + s
    p = D.make_problem(M_OBS, N_SRC, NF_FREQ, NF_FRAME, key, snr, seed=seed)
    X = p["X_tf"]
    _, dg_auto = nfr_mask(X, M_OBS, alpha=ALPHA, use_self_check=True)
    m_on, dg_on = nfr_mask(X, M_OBS, alpha=ALPHA, use_self_check=False)   # 强制 ON
    m_off = base_mask(X)                                                  # 强制 OFF
    return dict(case=key, snr_db=snr, seed=s,
                n_base=int(dg_on["n_base"]), n_keep=int(dg_on["n_keep"]),
                spread=float(dg_auto["spread"]), s2_ratio=float(
                    dg_on["s2_hat"] / max(np.mean(np.abs(p["noise_tf"]) ** 2), 1e-30)),
                A_on=kmeans_err(p, m_on, s), A_off=kmeans_err(p, m_off, s))


def run_eta(a) -> None:
    jobs = [(k, snr, s) for (k, snr) in GUARD_CASES for s in range(a.seeds)]
    t0 = time.perf_counter()
    recs: list[dict] = []
    if a.workers <= 1:
        recs = [eta_job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            recs = list(ex.map(eta_job, jobs))
    print(f"\n=== η 敏感性（{len(GUARD_CASES)} 条件 × {a.seeds} seeds，"
          f"{time.perf_counter()-t0:.0f}s）===")

    # 逐 (条件,seed) 的参考判定
    cond: dict[tuple[str, float], list[dict]] = {}
    for r in recs:
        cond.setdefault((r["case"], r["snr_db"]), []).append(r)
    for k, v in cond.items():
        for r in v:
            r["best"] = "ON" if r["A_on"] < r["A_off"] else "OFF"
            r["gain"] = abs(r["A_on"] - r["A_off"]) / max(r["A_off"], 1e-9)
            r["material"] = r["gain"] > a.gain_thr

    hdr = (f"{'η':>7}{'使用率':>9}{'假开启':>8}{'假回退':>8}{'总误判':>8}"
           f"{'平均A':>9}{'该开时A':>10}{'该关时A':>10}")
    print(hdr)
    print("-" * len(hdr))
    out_rows = []
    for eta in ETAS:
        usage = fa = ff = 0
        errs, errs_on, errs_off = [], [], []
        for (case, snr), rows in cond.items():
            for r in rows:
                need = max(10, math.ceil(eta * max(r["n_base"], 1)))
                ok = (r["spread"] <= 0.5) and (r["n_keep"] >= need) and (r["n_base"] >= 10)
                dec = "ON" if ok else "OFF"
                usage += int(dec == "ON")
                if dec == "ON" and r["best"] == "OFF" and r["material"]:
                    fa += 1
                if dec == "OFF" and r["best"] == "ON" and r["material"]:
                    ff += 1
                e = r["A_on"] if dec == "ON" else r["A_off"]
                errs.append(e)
                (errs_on if r["best"] == "ON" else errs_off).append(e)
        n = sum(len(v) for v in cond.values())
        row = dict(eta=eta, usage=usage / n, false_activation=fa, false_fallback=ff,
                   n_errors=fa + ff, mean_A=float(np.mean(errs)),
                   mean_A_should_on=float(np.mean(errs_on)),
                   mean_A_should_off=float(np.mean(errs_off)))
        out_rows.append(row)
        print(f"{eta:>7.3f}{usage/n:>9.1%}{fa:>8d}{ff:>8d}{fa+ff:>8d}"
              f"{np.mean(errs):>9.3f}{np.mean(errs_on):>10.3f}{np.mean(errs_off):>10.3f}")

    print("\n（条件级，与 Table 4 同口径：先对 10 个种子取平均再判定）")
    print(f"{'η':>7}{'条件级误判':>12}{'条件级使用率':>14}")
    for eta in ETAS:
        bad = 0
        used = 0
        for (case, snr), rows in cond.items():
            decs = []
            for r in rows:
                need = max(10, math.ceil(eta * max(r["n_base"], 1)))
                decs.append((r["spread"] <= 0.5) and (r["n_keep"] >= need) and (r["n_base"] >= 10))
            dec = sum(decs) > len(decs) / 2          # 多数表决
            used += int(dec)
            Aon = float(np.mean([r["A_on"] for r in rows]))
            Aoff = float(np.mean([r["A_off"] for r in rows]))
            best = "ON" if Aon < Aoff else "OFF"
            gain = abs(Aon - Aoff) / max(Aoff, 1e-9)
            if (("ON" if dec else "OFF") != best) and gain > a.gain_thr:
                bad += 1
        out_rows[ETAS.index(eta)]["cond_errors"] = bad
        out_rows[ETAS.index(eta)]["cond_usage"] = used / len(cond)
        print(f"{eta:>7.3f}{bad:>12d}{used/len(cond):>14.1%}")

    print(f"\n参考：该开时最优均值 "
          f"{np.mean([ (r['A_on'] if r['best']=='ON' else r['A_off']) for v in cond.values() for r in v if r['best']=='ON']):.3f}"
          f"；该关时最优均值 "
          f"{np.mean([ (r['A_on'] if r['best']=='ON' else r['A_off']) for v in cond.values() for r in v if r['best']=='OFF']):.3f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": vars(a), "etas": ETAS, "gain_thr": a.gain_thr,
                   "n_conditions": len(GUARD_CASES), "rows": out_rows,
                   "records": recs}, open(a.out, "w"))
        print(f"已写 {a.out}")


# ==========================================================================
# Part B — q_max 敏感性 + 安全上界
# ==========================================================================
def qmax_job(job: tuple[str, int]) -> dict:
    key, s = job
    p = D.make_problem(M_OBS, N_SRC, NF_FREQ, NF_FRAME, key, 20.0, seed=1000 + s)
    X = p["X_tf"]
    E = point_energies(X)
    sigma2 = float(np.mean(np.abs(p["noise_tf"]) ** 2))
    n = E.size
    es = np.sort(E[E > 0])
    nn = es.size

    # s²(q) 曲线：单个次序统计量，q 取期望水平 i/(n+1)
    qs = np.geomspace(1e-4, 0.1, 40)
    curve = []
    for q in qs:
        i = int(round(q * (nn + 1)))
        i = min(max(i, 1), nn)
        qq = i / (nn + 1.0)
        curve.append(dict(q=float(qq),
                          ratio=float(2.0 * es[i - 1] / chi2.ppf(qq, 2 * M_OBS) / sigma2)))

    # 各 q_max 下的中位数估计
    per = {}
    for qm in QMAXES:
        est = estimate_noise_power(E, M_OBS, q_hi=qm)
        per[str(qm)] = dict(s2_ratio=float(est["s2"] / sigma2),
                            spread=float(est["spread"]))
        # 下游
        tau = threshold_ratio(M_OBS, ALPHA)
        nu_hat = M_OBS * est["s2"]
        e_pts = np.sum(np.abs(X) ** 2, axis=0)
        b = base_mask(X)
        energ = e_pts > tau * nu_hat
        n_keep = int((b & energ).sum())
        need = max(10, math.ceil(0.035 * max(int(b.sum()), 1)))
        ok = (est["spread"] <= 0.5) and (n_keep >= need) and int(b.sum()) >= 10
        mask = (b & energ) if ok else b
        per[str(qm)]["A"] = kmeans_err(p, mask, s)
        per[str(qm)]["gate_on"] = bool(ok)

    # (a) 偏差口径的安全上界：s²(q)/σ²_true ≤ 1.20 的最大 q
    #     这是**分析**口径（用了真值），用来刻画估计量本身；不是算法的一部分。
    ok20 = [c["q"] for c in curve if c["ratio"] <= 1.20]
    # (b) 自适应口径：以保守端的平台为基准，取估计仍与之相符的最大 q。
    #     ref = q ∈ [1e-3, 3e-3] 上 s²(q) 的中位数（此时几乎无污染），
    #     q_adapt = 使 s²(q) ≤ 1.5·ref 的最大 q。完全不使用真值。
    plateau = [c["ratio"] for c in curve if 1e-3 <= c["q"] <= 3e-3]
    ref = float(np.median(plateau)) if plateau else float("nan")
    okA = [c["q"] for c in curve if c["ratio"] <= 1.5 * ref]
    return dict(case=key, seed=s, sigma2=sigma2, curve=curve, per=per,
                q_bias20=float(max(ok20)) if ok20 else float("nan"),
                q_adapt=float(max(okA)) if okA else float("nan"),
                plateau_ref=ref)


def run_qmax(a) -> None:
    cases = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
    jobs = [(k, s) for k in cases for s in range(a.seeds)]
    t0 = time.perf_counter()
    if a.workers <= 1:
        recs = [qmax_job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            recs = list(ex.map(qmax_job, jobs))
    print(f"\n=== q_max 敏感性（{a.seeds} seeds，{time.perf_counter()-t0:.0f}s）===")
    print(f"\n（1）σ̂²/σ² 随 q_max 的变化")
    hdr = f"{'case':<12}{'π₀':>8}" + "".join(f"{qm:>9}" for qm in QMAXES) + f"{'q_safe':>9}"
    print(hdr)
    print("-" * len(hdr))
    out = []
    for k in cases:
        sub = [r for r in recs if r["case"] == k]
        pi0 = (1 - float(D.SOURCE_CONFIGS[k]["param"])) ** N_SRC if k != "tf_gauss" else 0.0
        row = f"{k:<12}{pi0:>8.3f}"
        for qm in QMAXES:
            row += f"{np.mean([r['per'][str(qm)]['s2_ratio'] for r in sub]):>9.3f}"
        row += f"{np.nanmean([r['q_bias20'] for r in sub]):>9.4f}"
        print(row)
        out.append(dict(case=k, pi0=pi0,
                        ratios={str(qm): float(np.mean([r['per'][str(qm)]['s2_ratio'] for r in sub]))
                                for qm in QMAXES},
                        spreads={str(qm): float(np.mean([r['per'][str(qm)]['spread'] for r in sub]))
                                 for qm in QMAXES},
                        q_bias20=float(np.nanmean([r["q_bias20"] for r in sub])),
                        q_adapt=float(np.nanmean([r["q_adapt"] for r in sub])),
                        gate_rate={str(qm): float(np.mean([r['per'][str(qm)]['gate_on'] for r in sub]))
                                   for qm in QMAXES}))

    print(f"\n（2）下游角度误差（°）随 q_max 的变化")
    hdr = f"{'case':<12}" + "".join(f"{qm:>9}" for qm in QMAXES)
    print(hdr)
    print("-" * len(hdr))
    for k in cases:
        sub = [r for r in recs if r["case"] == k]
        print(f"{k:<12}" + "".join(
            f"{np.mean([r['per'][str(qm)]['A'] for r in sub]):>9.3f}" for qm in QMAXES))

    print(f"\n（3）门限开启率随 q_max 的变化")
    hdr = f"{'case':<12}" + "".join(f"{qm:>9}" for qm in QMAXES)
    print(hdr)
    print("-" * len(hdr))
    for k in cases:
        sub = [r for r in recs if r["case"] == k]
        print(f"{k:<12}" + "".join(
            f"{np.mean([r['per'][str(qm)]['gate_on'] for r in sub]):>9.2f}" for qm in QMAXES))

    print("\n（4）安全上界与自适应 q")
    print(f"{'case':<12}{'π₀':>8}{'q_bias20':>11}{'q_adapt':>10}")
    for r in out:
        print(f"  {r['case']:<12}{r['pi0']:>8.3f}{r['q_bias20']:>11.4f}{r['q_adapt']:>10.4f}")
    print("  q_bias20：σ̂²/σ² ≤ 1.20 的最大 q（分析口径，用了真值）")
    print("  q_adapt ：以 q∈[1e-3,3e-3] 的保守平台为基准、偏差 ≤50% 的最大 q（不使用真值）")
    print("\n（5）下游对 q_max 的敏感性：q_max ∈ [0.01,0.05] 内角度误差的相对变化")
    for r in out:
        vs = [np.mean([x['per'][str(qm)]['A'] for x in recs if x['case'] == r['case']])
              for qm in QMAXES if qm >= 0.01]
        spread = (max(vs) - min(vs)) / max(min(vs), 1e-12)
        print(f"  {r['case']:<12} {min(vs):.3f}–{max(vs):.3f}  相对变化 {spread*100:>6.2f}%")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": vars(a), "qmaxes": QMAXES, "summary": out,
                   "records": recs}, open(a.out, "w"))
        print(f"已写 {a.out}")


# ==========================================================================
# Part C — 最小列夹角：Prop. 6 正交近似的定量影响
# ==========================================================================
def mixed_median_ratio(A: np.ndarray, p: float, snr_db: float,
                       rng: np.random.Generator, n_pts: int = 300_000,
                       orthogonal: bool = False) -> float:
    """按生成器约定直接 MC：返回 median(e)/ν。

    orthogonal=True 时把 A_J^H A_J 的特征值全部换成其均值（即 Prop. 6 的假设）。
    """
    N, M = A.shape[1], A.shape[0]
    sigma2 = 1.0
    nu = M * sigma2
    p_act = 1.0 - (1.0 - p) ** N
    e1 = M * sigma2 * 10.0 ** (snr_db / 10.0) / (N * p)   # 单源点平均信号能量

    J = rng.binomial(N, p, size=n_pts)
    # 噪声分量（每点都有）
    e = 0.5 * sigma2 * rng.chisquare(2 * M, size=n_pts)
    for j in range(1, N + 1):
        idx = np.flatnonzero(J == j)
        if idx.size == 0:
            continue
        subs = list(combinations(range(N), j))
        for sub, part in zip(subs, np.array_split(idx, len(subs))):
            if part.size == 0:
                continue
            Aj = A[:, list(sub)]
            G = Aj.T @ Aj
            lam = np.linalg.eigvalsh(G)
            lam = np.maximum(lam, 1e-12)
            if orthogonal:
                lam = np.full(j, lam.mean())
            Ej = j * e1
            # 各模式独立 Exp(1)，尺度 Ej/j·λ_k
            u = rng.exponential(1.0, size=(j, part.size))
            e_sig = (Ej / j) * (lam[:, None] * u).sum(axis=0)
            e[part] += e_sig
    return float(np.median(e) / nu)


def angle_job(job: tuple[float, float, int]) -> dict:
    ang, p_ratio, s = job
    rng = np.random.default_rng(5000 + s)
    A = D.gen_mixing_matrix(M_OBS, N_SRC, rng, ang)
    lam12 = None
    if N_SRC >= 2:
        G = A[:, :2].T @ A[:, :2]
        lam12 = [float(x) for x in np.linalg.eigvalsh(G)]
    r_true = mixed_median_ratio(A, p_ratio, 20.0, rng, orthogonal=False)
    r_orth = mixed_median_ratio(A, p_ratio, 20.0, rng, orthogonal=True)
    out = dict(angle_deg=ang, p=p_ratio, seed=s, r_true=r_true, r_orth=r_orth,
               rel_dev=float((r_orth - r_true) / r_true), lam_pair=lam12)
    # 下游：固定 p 档做一次完整 pipeline
    key = {0.05: "tf_p05", 0.10: "tf_p10", 0.20: "tf_p20", 0.40: "tf_p40"}.get(p_ratio)
    if key is not None:
        pr = D.make_problem(M_OBS, N_SRC, NF_FREQ, NF_FRAME, key, 20.0,
                            seed=1000 + s, min_angle_deg=ang)
        X = pr["X_tf"]
        E = point_energies(X)
        m_nf, dg = nfr_mask(X, M_OBS, alpha=ALPHA, use_self_check=True)
        med = float(np.median(E))
        m_cl = base_mask(X) & (E.reshape(X.shape[1], X.shape[2]) > 0.02 * med)
        m_t5 = base_mask(X) & (E.reshape(X.shape[1], X.shape[2]) > 5.0 * med)
        out.update(A_nf=kmeans_err(pr, m_nf, s), A_classical=kmeans_err(pr, m_cl, s),
                   A_te5=kmeans_err(pr, m_t5, s), gate_on=bool(dg["gate_on"]))
    return out


def run_angle(a) -> None:
    jobs = [(ang, pr, s) for ang in ANGLES for pr in (0.05, 0.10, 0.20, 0.40)
            for s in range(a.seeds)]
    t0 = time.perf_counter()
    if a.workers <= 1:
        recs = [angle_job(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            recs = list(ex.map(angle_job, jobs))
    print(f"\n=== 最小列夹角敏感性（{a.seeds} seeds，{time.perf_counter()-t0:.0f}s）===")
    print("\n（1）Proposition 6 的『活跃列正交』近似误差")
    print(f"{'angle':>7}{'eig(A_2^H A_2)':>18}{'p=0.05':>10}{'p=0.10':>10}{'p=0.20':>10}{'p=0.40':>10}")
    print("-" * 65)
    summ = []
    for ang in ANGLES:
        row = f"{ang:>7.0f}"
        lam = [r["lam_pair"] for r in recs if r["angle_deg"] == ang and r["lam_pair"]]
        row += f"{np.mean([l[0] for l in lam]):>8.3f},{np.mean([l[1] for l in lam]):>8.3f}"
        for pr in (0.05, 0.10, 0.20, 0.40):
            v = [r["rel_dev"] for r in recs if r["angle_deg"] == ang and r["p"] == pr]
            row += f"{np.mean(v)*100:>9.2f}%"
            summ.append(dict(angle_deg=ang, p=pr, rel_dev=float(np.mean(v)),
                             r_true=float(np.mean([r["r_true"] for r in recs
                                                   if r["angle_deg"] == ang and r["p"] == pr])),
                             r_orth=float(np.mean([r["r_orth"] for r in recs
                                                   if r["angle_deg"] == ang and r["p"] == pr]))))
        print(row)
    print("  （数值为 (r_orth − r_true)/r_true，即把特征值换成均值后 r(p) 的相对偏差）")

    print("\n（2）下游角度误差（°）随夹角的分布")
    print(f"{'angle':>7}{'p':>7}{'NF':>9}{'classical':>11}{'te5':>9}{'gate开':>8}")
    print("-" * 52)
    for ang in ANGLES:
        for pr in (0.05, 0.10, 0.20, 0.40):
            sub = [r for r in recs if r["angle_deg"] == ang and r["p"] == pr
                   and "A_nf" in r]
            if not sub:
                continue
            print(f"{ang:>7.0f}{pr:>7.2f}{np.mean([r['A_nf'] for r in sub]):>9.3f}"
                  f"{np.mean([r['A_classical'] for r in sub]):>11.3f}"
                  f"{np.mean([r['A_te5'] for r in sub]):>9.3f}"
                  f"{np.mean([r['gate_on'] for r in sub]):>8.2f}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": vars(a), "angles": ANGLES, "summary": summ,
                   "records": recs}, open(a.out, "w"))
        print(f"已写 {a.out}")


# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["eta", "qmax", "angle"], required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gain-thr", type=float, default=0.15)
    a = ap.parse_args()
    {"eta": run_eta, "qmax": run_qmax, "angle": run_angle}[a.part](a)


if __name__ == "__main__":
    main()
