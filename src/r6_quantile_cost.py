"""R6-1  §A.17 / Table 37 的复测：**交货实现到底在做哪种选择**。

动机（改稿自查时发现，非审稿人指出）：

`r5_complexity.py` 把 `nfr.estimate_noise_power` 的计时列标为 "partition"，docstring 也写
"每个用 np.partition（选择算法，期望 O(n)）而非全排序"。但 `nfr.estimate_noise_power`
第 164 行是 `es = np.sort(E)`——**全排序**。归档那两列（"selection" 与 "full sort"）实际
都在排序，所以 ratio 才随 n 单调收敛到 1.00：3.99 / 1.64 / 1.16 / 1.04 / 1.01 / 1.00。
原脚本另有两个缺陷：REPS 循环**无预热**（首轮含 numpy 首次调用开销），且未固定线程数
（本机不固定时会超订，见 MEMORY.md 的环境记录）。

本脚本把三类成本**分开计时**，否则不可解释：

  sort_gather       全排序 + 取 30 个名义水平对应的次序统计量   —— 交货实现的数据路径
  partition_gather  一次 np.partition(kth=idx-1) 取同一批        —— 真正 O(n) 期望的那条
  ppf                scipy 的 chi2.ppf(水平向量)                  —— 与 n 无关的固定开销
  shipped            estimate_noise_power（= ppf + sort_gather + 中位数）

协议：预热 WARM 次、REPS 次取中位数、单线程（调用方设 OMP/MKL/OPENBLAS_NUM_THREADS=1）。
斜率在全部 6 档与最大的 3 档上分别拟合，因为小 n 档被 ppf 常数项压平。

用法：
    OMP_NUM_THREADS=1 python3 r6_quantile_cost.py --out ../results/r6_quantile_cost.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import chi2

from nfr import estimate_noise_power

SIZES = [2_112, 8_192, 32_768, 131_072, 524_288, 1_048_576]
M_OBS, SIG2 = 2, 1.0
REPS, WARM = 40, 5
Q_HI, N_Q = 0.02, 30
SEEDS_ACC = 20          # 精度列必须多 seed 平均：单次抽样的直方图误差在 0.1%–13% 之间跳


def synth(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((M_OBS, n)) + 1j * rng.standard_normal((M_OBS, n))) \
        * np.sqrt(SIG2 / 2.0)
    return np.sum(np.abs(Z) ** 2, axis=0)


def level_indices(n: int) -> np.ndarray:
    """与 `nfr.estimate_noise_power` 完全一致的索引：名义 30 个几何水平，整数截断后去重。"""
    i_hi = max(8, int(Q_HI * n))
    return np.unique(np.geomspace(1.0, i_hi, N_Q).astype(int))


def sort_gather(E: np.ndarray, idx: np.ndarray) -> np.ndarray:
    es = np.sort(E)
    return es[idx - 1]


def partition_gather(E: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.partition(E, idx - 1)[idx - 1]


def hist_quantile(E: np.ndarray, qs: np.ndarray, bins: int = 4096) -> np.ndarray:
    lo = float(E.min())
    hi = float(np.quantile(E, 0.5))
    if hi <= lo:
        hi = lo + 1e-12
    cnt, edges = np.histogram(E, bins=bins, range=(lo, hi))
    cdf = np.cumsum(cnt) / E.size
    return np.interp(qs, cdf, edges[:-1])


def timeit(fn, reps: int = REPS) -> float:
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def slope(ns: np.ndarray, ys: np.ndarray) -> float:
    return float(np.polyfit(np.log(ns), np.log(ys), 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"sizes": SIZES, "reps": REPS, "warm": WARM,
           "protocol": "median of REPS after WARM warm-up, single thread",
           "rows": []}
    print(f"{'n':>10}{'levels':>7}{'sort+gat':>11}{'part+gat':>11}{'ppf':>9}"
          f"{'hist':>10}{'shipped':>10}{'part/sort':>10}{'ship/sort':>10}{'ns/pt sort':>11}")
    for n in SIZES:
        E = synth(n)
        idx = level_indices(n)
        q = idx / (n + 1.0)
        t_sort = timeit(lambda: sort_gather(E, idx))
        t_part = timeit(lambda: partition_gather(E, idx))
        t_ppf = timeit(lambda: chi2.ppf(q, 2 * M_OBS))
        t_hist = timeit(lambda: hist_quantile(E, q))
        t_ship = timeit(lambda: estimate_noise_power(E, M_OBS))

        es = np.sort(E)
        s2_s = float(np.median(2.0 * es[idx - 1] / chi2.ppf(q, 2 * M_OBS)))
        pt = partition_gather(E, idx)
        s2_p = float(np.median(2.0 * pt / chi2.ppf(q, 2 * M_OBS)))
        qh = hist_quantile(E, q)
        s2_h = float(np.median(2.0 * qh / chi2.ppf(q, 2 * M_OBS)))

        # 精度列：多 seed 平均（timing 仍是单次实现，成本表不平均）
        ex_err, hi_err = [], []
        for s in range(SEEDS_ACC):
            Es = synth(n, seed=1000 + s)
            iq = level_indices(n) / (n + 1.0)
            ess = np.sort(Es)
            ex_err.append(abs(float(np.median(2.0 * ess[idx - 1] / chi2.ppf(iq, 2 * M_OBS))) / SIG2 - 1.0))
            qhs = hist_quantile(Es, iq)
            hi_err.append(abs(float(np.median(2.0 * qhs / chi2.ppf(iq, 2 * M_OBS))) / SIG2 - 1.0))

        print(f"{n:>10}{idx.size:>7}{t_sort*1e3:>9.3f} ms{t_part*1e3:>9.3f} ms"
              f"{t_ppf*1e3:>7.3f} ms{t_hist*1e3:>8.3f} ms{t_ship*1e3:>8.3f} ms"
              f"{t_part/t_sort:>10.2f}{t_ship/t_sort:>10.2f}{t_sort/n*1e9:>11.1f}")
        out["rows"].append(dict(
            n=int(n), n_levels=int(idx.size),
            t_sort_s=t_sort, t_partition_s=t_part, t_ppf_s=t_ppf,
            t_hist_s=t_hist, t_shipped_s=t_ship,
            ratio_partition_over_sort=float(t_part / t_sort),
            ratio_shipped_over_sort=float(t_ship / t_sort),
            ns_per_point_sort=float(t_sort / n * 1e9),
            ns_per_point_partition=float(t_part / n * 1e9),
            s2_sort=s2_s, s2_partition=s2_p, s2_histogram=s2_h,
            s2_rel_err_histogram=float(abs(s2_h / SIG2 - 1.0)),
            s2_rel_err_histogram_single=float(abs(s2_h / SIG2 - 1.0)),
            seeds_accuracy=int(SEEDS_ACC),
            s2_rel_err_exact_mean=float(np.mean(ex_err)),
            s2_rel_err_exact_max=float(np.max(ex_err)),
            s2_rel_err_histogram_mean=float(np.mean(hi_err)),
            s2_rel_err_histogram_max=float(np.max(hi_err)),
            s2_identical_sort_partition=bool(abs(s2_s - s2_p) < 1e-12)))

    ns = np.array([r["n"] for r in out["rows"]], float)
    tail = slice(3, None)
    for key in ("t_sort_s", "t_partition_s", "t_hist_s", "t_shipped_s"):
        ys = np.array([r[key] for r in out["rows"]])
        out[f"slope_{key}"] = slope(ns, ys)
        out[f"slope_tail_{key}"] = slope(ns[tail], ys[tail])
    out["ns_per_point_sort_at_1e6"] = out["rows"][-1]["ns_per_point_sort"]
    print("\nlog-log 斜率（全部 6 档 / 最大 3 档）")
    for key in ("t_sort_s", "t_partition_s", "t_hist_s", "t_shipped_s"):
        print(f"  {key:<14} {out['slope_'+key]:.3f}  /  {out['slope_tail_'+key]:.3f}")
    print(f"\n实际次序统计量条数： n=2112 → {out['rows'][0]['n_levels']}，"
          f"n=32768 → {level_indices(32768).size}，n=48000 → {level_indices(48000).size}，"
          f"n=1048576 → {out['rows'][-1]['n_levels']}")
    print(f"sort 与 partition 的估计值逐档相同： "
          f"{all(r['s2_identical_sort_partition'] for r in out['rows'])}")
    print(f"\n相对误差（{SEEDS_ACC} seeds 的均值 / 最坏值）")
    print(f"  {'n':>10}{'exact 均值':>12}{'exact 最坏':>12}{'直方图 均值':>13}{'直方图 最坏':>13}")
    for r in out["rows"]:
        print(f"  {r['n']:>10}{100*r['s2_rel_err_exact_mean']:>11.2f}%"
              f"{100*r['s2_rel_err_exact_max']:>11.2f}%"
              f"{100*r['s2_rel_err_histogram_mean']:>12.2f}%"
              f"{100*r['s2_rel_err_histogram_max']:>12.2f}%")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
