#!/usr/bin/env python3
"""R4 / Major 5: 密度聚类前端上的迁移检验。

审稿人要求：取一个密度聚类前端（DBSCAN 类，文献 [9] 为 DBSCAN-R），
**只替换其能量门限**（经典 0.02×median  vs  NF 准则），看增益是否转移。
Table 11 已在 spherical k-means 与 fuzzy c-means 上做过同样的事，这里补第三个前端。

实现说明（为何这样选）：
  DBSCAN 本身不产生"恰好 k 个簇中心"，无法直接给出 A 估计。文献里 DBSCAN 在 UBSS 中的
  典型用法是**密度预筛 + 聚类**（[9] 的 DBSCAN-R 即此类）：先用 DBSCAN 剔除低密度离群点，
  再对保留点聚类。因此这里采用 `DBSCAN 预筛 + spherical k-means`，
  它以 eps / min_samples 为前端参数，**两个门限共用同一组参数**，只有门限不同。

用法:
  ../.venv/bin/python r4_density_frontend.py --seeds 10 --out ../results/r4_density_frontend.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import metrics as MT
from baselines import kmeans_sphere, l1_recover, ssp_directions, ssp_mask_from_complex
from nfr import nfr_mask

ALPHA = 1e-4
LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
EPS_GRID = [0.02, 0.05, 0.10]      # 单位向量上的欧氏距离，≈ 夹角(rad)
MIN_SAMPLES = 10


def dbscan_filter(U: np.ndarray, eps: float, min_samples: int):
    """DBSCAN 预筛：返回保留列的布尔掩码（簇标签 >= 0）。"""
    from sklearn.cluster import DBSCAN
    if U.shape[1] == 0:
        return np.zeros(0, dtype=bool), 0
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(U.T)
    keep = lab >= 0
    return keep, int(len(set(lab[keep].tolist())) if keep.any() else 0)


def _mk(m_obs, n_src, F, T, key, snr, seed):
    return D.make_problem(m_obs, n_src, F, T, key, snr, seed=seed)


def run(seeds: int, eps: float, out_path: str | None):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    jobs = [(key, s) for key in LADDER for s in range(seeds)]
    ctx = mp.get_context("spawn")     # 父进程已 import sklearn/BLAS，fork 会死锁（本项目踩过）
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 4), mp_context=ctx) as ex:
        recs = [r for part in ex.map(_one_regime, [(j[0], j[1], eps) for j in jobs])
                for r in part]

    print(f"\n=== 密度预筛（DBSCAN eps={eps}, min_samples={MIN_SAMPLES}）+ spherical k-means ===")
    print(f"{'regime':<10}{'经典':>9}{'NF':>9}{'改善':>9}{'密度筛后保留':>13}{'NF 门限':>9}")
    print(f"{'':<10}{'A误差°':>9}{'A误差°':>9}{'':>9}")
    TOT = {"classical": [], "nf": [], "db_keep": []}
    per = {}
    for key in LADDER:
        sub = [r for r in recs if r["regime"] == key]
        a = float(np.nanmean([r["A_classical"] for r in sub]))
        b = float(np.nanmean([r["A_nf"] for r in sub]))
        k = float(np.nanmean([r["db_kept_frac"] for r in sub]))
        per[key] = dict(A_classical=a, A_nf=b, db_kept_frac=k,
                        n_clust=float(np.nanmean([r["n_clusters"] for r in sub])))
        TOT["classical"].append(a); TOT["nf"].append(b); TOT["db_keep"].append(k)
        print(f"{key:<10}{a:>9.3f}{b:>9.3f}{(a - b) / a:>8.0%}{k:>13.1%}{'':>9}")
    ma, mb = float(np.mean(TOT["classical"])), float(np.mean(TOT["nf"]))
    print(f"{'mean':<10}{ma:>9.3f}{mb:>9.3f}{(ma - mb) / ma:>8.0%}"
          f"{float(np.mean(TOT['db_keep'])):>13.1%}")
    # 逐格赢家
    wins = sum(1 for key in LADDER if per[key]["A_nf"] < per[key]["A_classical"])
    print(f"NF 更优的档位数: {wins}/{len(LADDER)}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(dict(eps=eps, min_samples=MIN_SAMPLES, n_seeds=seeds,
                           per_regime=per, mean_classical=ma, mean_nf=mb,
                           records=recs), f)
    return per, ma, mb, recs


def _one_regime(args):
    key, s, eps = args
    p = _mk(2, 4, 33, 64, key, 20.0, 1000 + s)
    rng = np.random.default_rng(s)
    out = []
    for tag, mask in (("classical", ssp_mask_from_complex(p["X_tf"], thr_cos=0.98,
                                                          thr_energy_ratio=0.02)),
                      ("nf", nfr_mask(p["X_tf"], 2, alpha=ALPHA)[0])):
        U = ssp_directions(p["X_tf"], mask)
        keep, ncl = dbscan_filter(U, eps, MIN_SAMPLES)
        U2 = U[:, keep] if keep.size else U
        A = kmeans_sphere(U2, p["n"], rng) if U2.shape[1] >= p["n"] else \
            kmeans_sphere(U, p["n"], rng)
        out.append(dict(
            regime=key, seed=s, gate=tag, eps=eps,
            n_admitted=int(mask.sum()), db_kept=int(keep.sum()),
            db_kept_frac=float(keep.mean()) if keep.size else float("nan"),
            n_clusters=ncl,
            A=float(MT.mixing_matrix_angle_error_deg(p["A"], A)),
        ))
    rec = dict(regime=key, seed=s, eps=eps,
               A_classical=out[0]["A"], A_nf=out[1]["A"],
               db_kept_frac=out[1]["db_kept_frac"], n_clusters=out[1]["n_clusters"],
               n_adm_classical=out[0]["n_admitted"], n_adm_nf=out[1]["n_admitted"])
    return [rec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results = {}
    for eps in ([a.eps] if a.eps else EPS_GRID):
        t0 = time.perf_counter()
        per, ma, mb, recs = run(a.seeds, eps, a.out if a.eps else None)
        results[str(eps)] = dict(per=per, mean_classical=ma, mean_nf=mb)
        print(f"  （eps={eps} 用时 {time.perf_counter()-t0:.0f}s）")
    if a.out and not a.eps:
        with open(a.out, "w") as f:
            json.dump(dict(eps_grid=EPS_GRID, min_samples=MIN_SAMPLES,
                           n_seeds=a.seeds, by_eps=results), f)
    print("\n=== 各 eps 下 NF 相对经典的改善（逐档均值平均）===")
    for eps, r in results.items():
        imp = (r["mean_classical"] - r["mean_nf"]) / r["mean_classical"]
        print(f"  eps={eps:<6} 经典 {r['mean_classical']:.3f}  NF {r['mean_nf']:.3f}  {imp:+.0%}")


if __name__ == "__main__":
    main()
