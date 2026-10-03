#!/usr/bin/env python3
"""R4 / Major 7 + Question 2：保留自检的**逐种子**错误分布与 spread 条件消融。

回答两件事：
  (M7) 17/270 的逐种子实质错误集中在哪些工况？是否在 (p, SNR) 平面的角上？
  (Q2) Algorithm 1 第 11 行的 `spread <= 0.5` 是否曾经起决定作用？做消融。

与 Table 4 / Table 21 完全同一条件网格与同一记录口径（10 seeds/条件，共 270 个决策），
逐种子把**决策本身**（而不是聚合量）与"该开该关"比较。

用法:
  ../.venv/bin/python r4_guard_errors.py --out ../results/r4_guard_errors.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from nfr import nfr_mask

CASES = [(k, 20.0) for k in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]]
CASES += [(k, s) for k in ["tf_p05", "tf_p20", "tf_gauss"] for s in [0.0, 5.0, 10.0, 30.0, 40.0]]
CASES += [(k, 20.0) for k in ["td_chirp", "td_sinusoid", "td_amfm", "td_impulse"]]
CASES += [(k, 20.0) for k in ["tf_b06", "tf_lap"]]
ALPHA = 1e-4
N_MIN = 10                       # nfr_mask 的 min_keep_abs
GAIN_THR = 0.15                  # 与 Table 4 / Table 21 同一"实质"判据


def plane(key: str, snr: float) -> tuple[str, str]:
    """把条件落到 (p, SNR) 平面上；返回 (p 标签, snr 标签)。"""
    pl = {"tf_p02": "0.02", "tf_p05": "0.05", "tf_p10": "0.10", "tf_p20": "0.20",
          "tf_p40": "0.40", "tf_gauss": "dense"}.get(key, "other")
    return pl, (f"{snr:.0f}" if key in ("tf_p05", "tf_p20", "tf_gauss") else "20")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/r4_guard_errors.json")
    a = ap.parse_args()

    per_seed = []
    for key, snr in CASES:
        for s in range(10):
            p = D.make_problem(2, 4, 33, 64, key, snr,
                               seed=(1000 if snr == 20.0 else 2000) + s)
            mka, dg = nfr_mask(p["X_tf"], 2, alpha=ALPHA, use_self_check=True)
            mko, _ = nfr_mask(p["X_tf"], 2, alpha=ALPHA, use_self_check=False)
            # 语义必须与 Table 4 / Table 21 一致：
            #   mka = 自动决策的掩码        mko = 门限**强制开启**的掩码
            #   base = 门限**关闭**（只有共线+均衡判据）—— 这才是 "A_off"
            # 第一版把 mka 当成了 A_on、把 mko 当成了 A_off（标签写反），
            # 结果 dense 档凭空多出 50 个"假回退"。这与本项目反复踩的坑同类：标签而非数值出错。
            Xr, Xi = p["X_tf"].real, p["X_tf"].imag
            nr = np.linalg.norm(Xr, axis=0); ni = np.linalg.norm(Xi, axis=0)
            dd = np.maximum(nr + ni, 1e-15)
            cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
            base_mask = (np.abs(cos) > 0.98) & (nr / dd > 0.1) & (ni / dd > 0.1)
            errs = {}
            for tag, mk in (("on", mko), ("off", base_mask)):
                rng = np.random.default_rng(s)
                errs[tag] = MT.mixing_matrix_angle_error_deg(
                    p["A"], kmeans_sphere(ssp_directions(p["X_tf"], mk), 4, rng))
            A_on, A_off = errs["on"], errs["off"]
            gain = abs(A_on - A_off) / max(A_off, 1e-12)
            material = bool(gain > GAIN_THR)
            preferred = "ON" if A_on < A_off else "OFF"
            decision = "ON" if dg["gate_on"] else "OFF"
            dec_nospread = "ON" if (dg["retention_ok"] and dg["n_base"] >= N_MIN) else "OFF"
            pl, sl = plane(key, snr)
            per_seed.append(dict(
                case=key, snr_db=snr, p_label=pl, snr_label=sl, seed=s,
                spread=float(dg["spread"]), n_base=int(dg["n_base"]),
                n_keep=int(dg["n_keep"]), need=int(dg["need"]),
                keep_frac=float(dg["keep_frac"]),
                spread_ok=bool(dg["spread_ok"]), retention_ok=bool(dg["retention_ok"]),
                base_ok=bool(dg["n_base"] >= N_MIN),
                A_on=float(A_on), A_off=float(A_off), gain=float(gain),
                material=material, preferred=preferred, decision=decision,
                decision_no_spread=dec_nospread,
                error=bool(decision != preferred and material),
                error_no_spread=bool(dec_nospread != preferred and material),
            ))

    n = len(per_seed)
    e = sum(r["error"] for r in per_seed)
    e2 = sum(r["error_no_spread"] for r in per_seed)
    print(f"=== 逐种子决策（{n} 个，与 Table 21 同网格）===")
    print(f"  实质错误：含 spread 条件 {e}/{n} = {e/n:.1%}   |   去掉 spread 条件 {e2}/{n} = {e2/n:.1%}")
    fa = sum(1 for r in per_seed if r["error"] and r["decision"] == "ON")
    ff = sum(1 for r in per_seed if r["error"] and r["decision"] == "OFF")
    print(f"  方向：假开启 {fa}   假回退 {ff}")

    print("\n=== 错误在 (p, SNR) 平面上的分布 ===")
    pls = ["0.02", "0.05", "0.10", "0.20", "0.40", "dense", "other"]
    sls = ["0", "5", "10", "20", "30", "40"]
    print(f"  {'p \\ SNR':<8}" + "".join(f"{s:>8}" for s in sls) + f"{'合计':>8}{'决策数':>8}")
    for pl in pls:
        row, tot = [], 0
        for sl in sls:
            sub = [r for r in per_seed if r["p_label"] == pl and r["snr_label"] == sl]
            row.append(sum(r["error"] for r in sub)); tot += sum(r["error"] for r in sub)
        cnt = sum(1 for r in per_seed if r["p_label"] == pl)
        if cnt == 0:
            continue
        print(f"  {pl:<8}" + "".join(f"{v if v else '.':>8}" for v in row)
              + f"{tot:>8}{cnt:>8}")
    print(f"  合计 {e}")

    print("\n=== 触发关闭时，哪一条子条件在起决定作用 ===")
    off = [r for r in per_seed if r["decision"] == "OFF"]
    print(f"  关闭次数 {len(off)}/{n}")
    for tag, f in (("仅 spread 不过", lambda r: (not r["spread_ok"]) and r["retention_ok"] and r["base_ok"]),
                   ("仅 retention 不过", lambda r: r["spread_ok"] and (not r["retention_ok"]) and r["base_ok"]),
                   ("仅 n_base 不足", lambda r: r["spread_ok"] and r["retention_ok"] and (not r["base_ok"])),
                   ("多条同时不过", lambda r: sum([not r["spread_ok"], not r["retention_ok"], not r["base_ok"]]) > 1)):
        k = sum(1 for r in off if f(r))
        print(f"    {tag:<18} {k:>4}")
    print("\n=== spread<=0.5 的消融 ===")
    changed = [r for r in per_seed if r["decision"] != r["decision_no_spread"]]
    print(f"  决策被该条件改变：{len(changed)}/{n}")
    for r in changed[:12]:
        print(f"    {r['case']}@{r['snr_db']:.0f}dB seed{r['seed']}  spread={r['spread']:.3f} "
              f"keep={r['keep_frac']:.3f} 偏好={r['preferred']} 原决策={r['decision']} 去该条件={r['decision_no_spread']}")
    # 关闭的情况里，若没有 spread 条件，有多少会变成"开着且更差"
    harmed = [r for r in changed if r["decision"] == "OFF" and r["decision_no_spread"] == "ON"]
    print(f"  其中去掉该条件后改为开启的 {len(harmed)} 例，实质错误数由 {e} 变为 {e2}")

    print("\n=== spread 值的分布（关闭 vs 开启）===")
    so = [r["spread"] for r in off]
    son = [r["spread"] for r in per_seed if r["decision"] == "ON"]
    print(f"  关闭的 spread：{sorted(round(x,3) for x in so)}")
    print(f"  开启的 spread：最大 {max(son):.3f}  中位 {np.median(son):.3f}")

    json.dump(dict(alpha=ALPHA, gain_thr=GAIN_THR, n_decisions=n,
                   n_errors=e, n_errors_no_spread=e2,
                   false_activation=fa, false_fallback=ff,
                   n_changed_by_spread=len(changed),
                   records=per_seed), open(a.out, "w"), indent=1)
    print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
