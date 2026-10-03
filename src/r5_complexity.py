"""R5-7  Q3：噪声底估计里分位数计算的复杂度与加速空间。

审稿人问：算法在超长音频或高频谱分辨率（F·T 很大）时，对下尾排序/分位数计算的
复杂度是多少？能否用直方图近似或局部采样进一步加速？

回答分三层：
  1. **实现层面的复杂度**：估计只用 30 个小分位数，每个用 ``np.partition``（选择算法，
     期望 O(n)）而非全排序（O(n log n)）；本脚本测其随 n 的伸缩并与全排序对比。
  2. **加速方案**：直方图近似（一次 O(n) 遍历 + O(B) 插值），实测其精度代价。
  3. **总量**：把它放回整条流水线（A.7 的分项计时）看它占多少。

用法：
    python3 r5_complexity.py --out ../results/r5_complexity.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import chi2

from nfr import estimate_noise_power

SIZES = [2_112, 8_192, 32_768, 131_072, 524_288, 1_048_576]
M_OBS, SIG2, REPS = 2, 1.0, 20


def synth(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((M_OBS, n)) + 1j * rng.standard_normal((M_OBS, n))) \
        * np.sqrt(SIG2 / 2.0)
    return np.sum(np.abs(Z) ** 2, axis=0)


def hist_quantile(E: np.ndarray, qs: np.ndarray, bins: int = 4096) -> np.ndarray:
    """直方图近似分位数：一次 O(n) 直方图 + O(B) 插值（分位点都很小，只需低端区间）。"""
    lo = float(E.min())
    hi = float(np.quantile(E, 0.5))
    if hi <= lo:
        hi = lo + 1e-12
    cnt, edges = np.histogram(E, bins=bins, range=(lo, hi))
    cdf = np.cumsum(cnt) / E.size
    return np.interp(qs, cdf, edges[:-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"sizes": SIZES, "reps": REPS, "rows": []}
    print(f"{'n':>10}{'partition':>13}{'全排序':>12}{'直方图':>12}"
          f"{'ns/点':>9}{'σ̂²(精确)':>11}{'σ̂²(直方)':>11}{'相对误差':>10}")
    for n in SIZES:
        E = synth(n)
        t0 = time.perf_counter()
        for _ in range(REPS):
            r = estimate_noise_power(E, M_OBS)
        t_est = (time.perf_counter() - t0) / REPS

        t0 = time.perf_counter()
        for _ in range(REPS):
            Es = np.sort(E)
            qs = np.arange(1, 31) / (E.size + 1)
            _ = np.interp(qs, np.arange(E.size), Es)
        t_sort = (time.perf_counter() - t0) / REPS

        qs = np.arange(1, 31) / (E.size + 1)
        t0 = time.perf_counter()
        for _ in range(REPS):
            _ = hist_quantile(E, qs)
        t_hist = (time.perf_counter() - t0) / REPS

        qh = hist_quantile(E, qs)
        s2_h = float(np.median(2.0 * qh / chi2.ppf(qs, 2 * M_OBS)))
        err = abs(s2_h / SIG2 - 1.0)
        print(f"{n:>10}{t_est*1e3:>11.3f} ms{t_sort*1e3:>10.3f} ms{t_hist*1e3:>10.3f} ms"
              f"{t_est/n*1e9:>9.1f}{r['s2']:>11.4f}{s2_h:>11.4f}{err*100:>9.2f}%")
        out["rows"].append(dict(n=int(n), t_partition_s=t_est, t_sort_s=t_sort,
                                t_hist_s=t_hist, s2_partition=float(r["s2"]),
                                s2_hist=s2_h, s2_rel_err_hist=float(err),
                                ns_per_point=t_est / n * 1e9))

    ns = np.array([r["n"] for r in out["rows"]], dtype=float)
    ts = np.array([r["t_partition_s"] for r in out["rows"]])
    sl = float(np.polyfit(np.log(ns), np.log(ts), 1)[0])
    sl_s = float(np.polyfit(np.log(ns), np.log([r["t_sort_s"] for r in out["rows"]]), 1)[0])
    sl_h = float(np.polyfit(np.log(ns), np.log([r["t_hist_s"] for r in out["rows"]]), 1)[0])
    print(f"\nlog-log 斜率：partition {sl:.3f}  全排序 {sl_s:.3f}  直方图 {sl_h:.3f}")
    print(f"n=1e6 时每点耗时：partition {ts[-1]/ns[-1]*1e9:.1f} ns"
          f"（全排序 {out['rows'][-1]['t_sort_s']/ns[-1]*1e9:.1f} ns）")
    print(f"外推：A.7 实测真实语音 4.8e4 点用 2.99 ms；按 O(n) 推到 1e6 点"
          f"约 {ts[-1]*1e3:.1f} ms，占同一问题规模下 ℓ1 恢复（≈2.3 s × 21）的"
          f" {ts[-1]*1000/(2337*1e6/4.8e4):.3%}")
    out.update(slope_partition=sl, slope_sort=sl_s, slope_hist=sl_h)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
