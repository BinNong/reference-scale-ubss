"""真实语音（LibriSpeech dev-clean）上的实验矩阵。

与合成实验共用同一套指标与恢复器（去偏 ℓ1），唯一区别是源信号来自真实录音。
模型完全同构：X(f,t) = A·S(f,t) + N(f,t)，A 为实瞬时混合（STFT 域精确成立）。

对比的参考尺度（四种"无调参"候选 + 两种逐档 oracle）
---------------------------------------------------
  · SCA-median(默认)   e > 0.02 · median(e)      —— 文献中最常见的写法
  · SCA-max            e > 0.05 · max(e)         —— 另一种常见写法（"最大能量的若干比例"）
  · top-K              保留能量最强的 5% 点       —— 能量排名类做法
  · NF-SSP             e > τ·ν̂，τ=Q_{1−α}(χ²_2M)/(2M) —— 本文方法（单一 α）
  · SCA-median-opt / SCA-max-opt   逐档把系数调到最优（oracle 上界）

用法:
    ../.venv/bin/python experiments_real.py --seeds 10 --workers 6 \
        --out ../results/real_results.json --grid-out ../results/real_grid.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats

import metrics as MT
import realdata as RD
from baselines import fcm_sphere, kmeans_sphere, l1_recover, run_method, ssp_directions
from baselines import ssp_mask_from_complex
from nfr import nfr_mask, threshold_ratio
from experiments_nfr import _record

ALPHA = 1e-4
MED_GRID = [0.02, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
MAX_GRID = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3]
TOPK = 0.05
DUR = 3.0
M_OBS = 2

CONFIGS = [
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


def _corpus(root):
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = RD.scan_corpus(root)
    return _CORPUS


def build(root, kw, seed, n_files=40):
    need = kw["n_sources"] - kw.get("n_dense", 0)
    pool = RD.load_pool(_corpus(root), n_files=n_files, dur_s=DUR, seed=seed,
                        min_speakers=need)
    return RD.make_real_problem(pool, m_obs=M_OBS, dur_s=DUR, seed=1000 + seed,
                                hop=kw.get('win', 1024) // 2, **kw)


# --------------------------------------------------------------------------
# 掩码构造
# --------------------------------------------------------------------------

def mask_median(prob, te, c0=0.98):
    return ssp_mask_from_complex(prob["X_tf"], thr_cos=c0, thr_energy_ratio=te)


def mask_max(prob, c, c0=0.98):
    base = ssp_mask_from_complex(prob["X_tf"], thr_cos=c0, thr_energy_ratio=0.0)
    e = np.sum(np.abs(prob["X_tf"]) ** 2, axis=0)
    return base & (e >= c * e.max())


def mask_topk(prob, frac, c0=0.98):
    base = ssp_mask_from_complex(prob["X_tf"], thr_cos=c0, thr_energy_ratio=0.0)
    e = np.sum(np.abs(prob["X_tf"]) ** 2, axis=0)
    return base & (e > np.quantile(e, 1.0 - frac))



def _record_A_only(exp, method, prob, A_hat, dt, seed, extra):
    """只记录混合矩阵误差的记录（不做源恢复，故 SDR/SIR 为 NaN）。

    逐档 oracle 行的用途是给出"该档把系数调到最优"时 A 误差能达到的上界；
    对它跑恢复只是浪费算力，且若用不同恢复器会让 SDR 列失去可比性，
    因此这里显式地记为 NaN 而不是塞入一个不同协议的数值。
    """
    import metrics as _MT
    rec = {"experiment": exp, "method": method, "cfg_key": prob["cfg_key"],
           "seed": int(seed),
           "snr_db": float(prob["snr_db"]) if prob["snr_db"] is not None else None,
           "m_obs": int(prob["m"]), "n_true": int(prob["n"]),
           "A_angle_deg": float(_MT.mixing_matrix_angle_error_deg(prob["A"], A_hat)),
           "SDR": float("nan"), "SIR": float("nan"), "time_s": float(dt)}
    rec.update(extra)
    return rec


def _A(prob, mask, seed, clusterer="kmeans", n_init=10):
    U = ssp_directions(prob["X_tf"], mask)
    if U.shape[1] < prob["n"]:
        return None
    rng = np.random.default_rng(seed)
    return (kmeans_sphere(U, prob["n"], rng, n_init=n_init)
            if clusterer == "kmeans" else fcm_sphere(U, prob["n"], rng))


def A_err(prob, mask, seed=0, clusterer="kmeans", n_init=3):
    A = _A(prob, mask, seed, clusterer, n_init)
    if A is None:
        return float("nan")
    return MT.mixing_matrix_angle_error_deg(prob["A"], A)


# --------------------------------------------------------------------------

def run_case(job):
    """单个配置的全部工作（在一个工作进程里执行）。"""
    root, exp, tag, kw, seeds, tune_seeds = job
    case = f"{exp}:{tag}"
    recs = []
    diag_keys = ("pi0", "pi1", "pi2p", "pi0_supp", "single_dom_frac", "overlap",
                 "gini", "r_true", "nu_true", "equiv_err", "win", "n_dense",
                 "snr_true_db")
    t0 = time.perf_counter()

    # ---- 1) 逐档 oracle 调参：用与评测集**不重叠**的独立种子（held-out），
    #         只用 A 误差，不跑恢复。
    # 原实现是 `range(tune_seeds)`，即调参种子 {0..tune_seeds-1} 是评测种子
    # {0..seeds-1} 的**子集**——系数是在 2/8 的评测实例上选的，属于部分泄漏。
    # 改为 `range(seeds, seeds+tune_seeds)`（种子 8,9），成为真正的验证集调参，
    # 与评测集不相交。实测该改动对 best-observed 均值的影响为 0.703° -> 0.686°。
    med_tbl = {te: [] for te in MED_GRID}
    max_tbl = {c: [] for c in MAX_GRID}
    topk_tbl = {f: [] for f in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]}
    for s in range(seeds, seeds + tune_seeds):
        p = build(root, kw, s)
        for te in MED_GRID:
            med_tbl[te].append(A_err(p, mask_median(p, te), s))
        for c in MAX_GRID:
            max_tbl[c].append(A_err(p, mask_max(p, c), s))
        for f in topk_tbl:
            topk_tbl[f].append(A_err(p, mask_topk(p, f), s))
    te_star = min(MED_GRID, key=lambda k: np.nanmean(med_tbl[k]))
    c_star = min(MAX_GRID, key=lambda k: np.nanmean(max_tbl[k]))
    f_star = min(topk_tbl, key=lambda k: np.nanmean(topk_tbl[k]))
    grid = {f"{case}|median": {str(k): float(np.nanmean(v)) for k, v in med_tbl.items()},
            f"{case}|max": {str(k): float(np.nanmean(v)) for k, v in max_tbl.items()},
            f"{case}|topk": {str(k): float(np.nanmean(v)) for k, v in topk_tbl.items()},
            f"{case}|star": {"te": te_star, "c": c_star, "frac": f_star}}

    # ---- 2) 各方法在全部种子上评测 ----
    for s in range(seeds):
        p = build(root, kw, s)
        dg = {k: p[k] for k in diag_keys}
        dg["noise_kind"] = p["noise_kind"] or "none"

        # (a) 无调参候选
        cands = [("NF-SSP", None), ("SCA-median", mask_median(p, 0.02)),
                 ("SCA-max", mask_max(p, 0.05)), ("top-K", mask_topk(p, TOPK))]
        for name, mk in cands:
            rng = np.random.default_rng(s)
            t1 = time.perf_counter()
            if name == "NF-SSP":
                mk, d = nfr_mask(p["X_tf"], p["m"], alpha=ALPHA)
            A = _A(p, mk, s)
            if A is None:
                continue
            S = l1_recover(A, p["X_all"])
            extra = dict(case=case, **dg)
            if name == "NF-SSP":
                med_e = float(np.median(np.sum(np.abs(p["X_tf"]) ** 2, axis=0).ravel()))
                extra.update(gate_on=bool(d["gate_on"]), keep_frac=d["keep_frac"],
                             t_median=float(threshold_ratio(p["m"], ALPHA) * d["nu_hat"]
                                            / max(med_e, 1e-30)),
                             nu_hat=d["nu_hat"], spread=d["spread"],
                             n_keep=d["n_keep"], n_base=d["n_base"])
            recs.append(_record(exp, name, p, A, S, time.perf_counter() - t1,
                                seed=s, extra=extra))

        # (b) 逐档 oracle
        for name, mk, par in [("SCA-median-opt", mask_median(p, te_star), dict(te=te_star)),
                              ("SCA-max-opt", mask_max(p, c_star), dict(c=c_star)),
                              ("top-K-opt", mask_topk(p, f_star), dict(frac=f_star))]:
            t1 = time.perf_counter()
            A = _A(p, mk, s)
            if A is None:
                continue
            # 逐档 oracle 只用于给出 A 误差的调参上界，不跑恢复
            recs.append(_record_A_only(exp, name, p, A, time.perf_counter() - t1,
                                       s, dict(case=case, **dg, **par)))

        # (d) 真值 A：SDR 上界
        r = run_method("oracle_a_l1", p, seed=s)
        recs.append(_record(exp, "Oracle-A", p, r["A_hat"], r["S_hat"], r["time"],
                            seed=s, extra=dict(case=case, **dg)))
    return recs, grid, case, time.perf_counter() - t0


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/ls_corpus")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--tune-seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="../results/real_results.json")
    ap.add_argument("--grid-out", default="../results/real_grid.json")
    a = ap.parse_args()

    corpus = RD.scan_corpus(a.corpus)
    spk = sorted({c["speaker"] for c in corpus})
    print(f"语料 {a.corpus}: {len(corpus)} 文件 / {len(spk)} 说话人 {spk}")
    print(f"时长 {DUR}s，M={M_OBS}，α={ALPHA:g}，seeds={a.seeds}（调参 {a.tune_seeds}），"
          f"workers={a.workers}")

    todo = set(a.only.split(",")) if a.only else None
    jobs = [(a.corpus, e, t, kw, a.seeds, a.tune_seeds)
            for e, t, kw in CONFIGS if not todo or t in todo]

    recs, grids, t_all = [], {}, time.perf_counter()
    if a.workers <= 1:
        for j in jobs:
            r, g, case, dt = run_case(j)
            recs += r; grids.update(g)
            print(f"  {case:<22s} {dt:6.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_case, j): j for j in jobs}
            for fu in as_completed(futs):
                r, g, case, dt = fu.result()
                recs += r; grids.update(g)
                print(f"  {case:<22s} {dt:6.1f}s", flush=True)
    print(f"\n共 {len(recs)} 条记录，用时 {time.perf_counter()-t_all:.1f}s")

    # ================= 汇总 =================
    cases = sorted({r["case"] for r in recs},
                   key=lambda c: [f"{e}:{t}" for e, t, _ in CONFIGS].index(c))
    METHODS = ["NF-SSP", "SCA-median", "SCA-max", "top-K",
               "SCA-median-opt", "SCA-max-opt", "top-K-opt", "Oracle-A"]

    def g(c, m, metric="A_angle_deg"):
        v = [r[metric] for r in recs if r.get("case") == c and r["method"] == m]
        return float(np.mean(v)) if v else float("nan")

    def d(c, key):
        v = [r[key] for r in recs if r.get("case") == c and r["method"] == "NF-SSP"
             and r.get(key) is not None]
        return float(np.mean(v)) if v else float("nan")

    print("\n" + "=" * 118)
    print("R0  真实数据统计 + 各档最优点数（用于判断阈值是否落在正确量级）")
    print("=" * 118)
    h = (f"{'case':<20s}{'π₀':>7s}{'π₁':>7s}{'WDO.9':>7s}{'r_true':>8s}{'ν_true':>10s} │"
         f"{'te*':>8s}{'c*':>7s}{'f*':>6s}{'t_e(NF)':>9s}{'gate':>6s} │"
         f"{'NF':>8s}{'med':>8s}{'max':>8s}{'topK':>8s}{'med*':>8s}{'max*':>8s}{'topK*':>8s}")
    print(h); print("-" * len(h))
    for c in cases:
        st = grids.get(f"{c}|star", {})
        print(f"{c:<20s}{d(c,'pi0'):>7.3f}{d(c,'pi1'):>7.3f}{d(c,'single_dom_frac'):>7.3f}"
              f"{d(c,'r_true'):>8.2f}{d(c,'nu_true'):>10.2e} │"
              f"{st.get('te',float('nan')):>8g}{st.get('c',float('nan')):>7g}"
              f"{st.get('frac',float('nan')):>6g}{d(c,'t_median'):>9.3f}"
              f"{d(c,'gate_on'):>6.2f} │"
              + "".join(f"{g(c,m):>8.2f}" for m in
                        ["NF-SSP", "SCA-median", "SCA-max", "top-K",
                         "SCA-median-opt", "SCA-max-opt", "top-K-opt"]))

    print("\n" + "=" * 118)
    print("R1  A 误差(°)")
    print("=" * 118)
    h2 = f"{'case':<20s}" + "".join(f"{m:>15s}" for m in METHODS)
    print(h2); print("-" * len(h2))
    for c in cases:
        print(f"{c:<20s}" + "".join(f"{g(c,m):>15.2f}" for m in METHODS))

    print("\n--- SDR(dB) ---")
    print(h2); print("-" * len(h2))
    for c in cases:
        print(f"{c:<20s}" + "".join(f"{g(c,m,'SDR'):>15.2f}" for m in METHODS))

    print("\n" + "=" * 118)
    print("R2  配对 t 检验（各无调参候选 vs 经典中位数默认）")
    print("=" * 118)
    for c in cases:
        a0 = {r["seed"]: r["A_angle_deg"] for r in recs
              if r.get("case") == c and r["method"] == "SCA-median"}
        if len(a0) < 3:
            continue
        line = f"  {c:<20s} 默认={np.mean(list(a0.values())):7.2f} │"
        for m in ["NF-SSP", "SCA-max", "top-K"]:
            a1 = {r["seed"]: r["A_angle_deg"] for r in recs
                  if r.get("case") == c and r["method"] == m}
            ks = sorted(set(a0) & set(a1))
            t, pv = stats.ttest_rel([a0[k] for k in ks], [a1[k] for k in ks])
            line += f" {m}:{np.mean([a1[k] for k in ks]):6.2f}(p={pv:7.1e})"
        print(line)

    print("\n--- 运行时间 (s/问题) ---")
    for m in METHODS:
        v = [r["time_s"] for r in recs if r["method"] == m]
        if v:
            print(f"  {m:<16s}{np.mean(v):>9.4f}")

    print("\n--- NF-SSP 自检触发 ---")
    for c in cases:
        rows = [r for r in recs if r.get("case") == c and r["method"] == "NF-SSP"]
        if rows:
            off = sum(1 for r in rows if not r.get("gate_on", True))
            print(f"  {c:<20s} 关闭 {off}/{len(rows)}  平均保留 {d(c,'keep_frac'):.3f}")

    err = max(d(c, "equiv_err") for c in cases)
    print(f"\n时域混合 ⇔ 频域按列混合最大偏差 {err:.3e}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"n_seeds": a.seeds, "tune_seeds": a.tune_seeds, "alpha": ALPHA,
                   "dur_s": DUR, "m_obs": M_OBS, "topk": TOPK,
                   "n_speakers": len(spk), "speakers": spk,
                   "median_grid": MED_GRID, "max_grid": MAX_GRID,
                   "records": recs}, f)
    with open(a.grid_out, "w") as f:
        json.dump(grids, f, indent=2)
    print(f"已写出 {a.out} 与 {a.grid_out}")


if __name__ == "__main__":
    main()
