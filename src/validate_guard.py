"""验证保留约束自检：auto 是否在"该开时开、该关时关"。

对全条件网格比较 auto / 强制ON / 强制OFF 的 A 误差，并报告决策正确性。
结果同时写入 JSON，作为论文 Table 4 的机器可读记录。

用法（服务器 src 目录）：
    ../.venv/bin/python validate_guard.py --out ../results/guard_validation.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from nfr import nfr_mask

CASES = [(k, 20.0) for k in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]]
CASES += [(k, s) for k in ["tf_p05", "tf_p20", "tf_gauss"] for s in [0.0, 5.0, 10.0, 30.0, 40.0]]
CASES += [(k, 20.0) for k in ["td_chirp", "td_sinusoid", "td_amfm", "td_impulse"]]
CASES += [(k, 20.0) for k in ["tf_b06", "tf_lap"]]
ALPHA = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/guard_validation.json")
    ap.add_argument("--gain-thr", type=float, default=0.15,
                    help="判为误判所需的最小相对收益")
    a = ap.parse_args()
    print(f"{'case':<20s}{'keep%':>8s}{'auto':>6s}{'A(auto)':>9s}{'A(ON)':>8s}{'A(OFF)':>8s}"
          f"{'最优':>7s}{'增益':>8s}{'决策':>7s}")
    bad = 0
    rows = []
    for key, snr in CASES:
        ka, ea, eo, ef, onf = [], [], [], [], []
        for s in range(10):
            p = D.make_problem(2, 4, 33, 64, key, snr, seed=(1000 if snr == 20.0 else 2000) + s)
            mka, dg = nfr_mask(p["X_tf"], 2, alpha=ALPHA, use_self_check=True)
            Xr, Xi = p["X_tf"].real, p["X_tf"].imag
            nr = np.linalg.norm(Xr, axis=0); ni = np.linalg.norm(Xi, axis=0)
            d = np.maximum(nr + ni, 1e-15)
            cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, 1e-15)
            base = (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)
            mko, _ = nfr_mask(p["X_tf"], 2, alpha=ALPHA, use_self_check=False)
            for mk, buf in ((mka, ea), (mko, eo), (base, ef)):
                rng = np.random.default_rng(s)
                A = kmeans_sphere(ssp_directions(p["X_tf"], mk), 4, rng)
                buf.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
            ka.append(dg["keep_frac"])
            onf.append(bool(dg["gate_on"]))
        ma, mo, mf = np.mean(ea), np.mean(eo), np.mean(ef)
        best = "ON" if mo < mf else "OFF"
        gain = abs(mo - mf) / max(mf, 1e-9)
        # 逐条件决策必须**数决策本身**，不能用"平均误差是否相等"去推断：
        # 只要 10 个种子里有一个关了门限，均值就会不同而被误判为 OFF（实测踩过）。
        dec = "ON" if sum(onf) > len(onf) / 2 else "OFF"
        wrong = (dec != best) and gain > a.gain_thr
        if wrong:
            bad += 1
        rows.append({"case": key, "snr_db": snr, "keep_frac": float(np.mean(ka)),
                     "A_auto": float(ma), "A_on": float(mo), "A_off": float(mf),
                     "preferred": best, "gain": float(gain), "decision": dec,
                     "n_gate_on": int(sum(onf)), "n_seeds": int(len(onf)),
                     "wrong": bool(wrong)})
        print(f"{key+'@'+str(int(snr))+'dB':<20s}{np.mean(ka):>8.1%}{dec:>6s}{ma:>9.2f}"
              f"{mo:>8.2f}{mf:>8.2f}{best:>7s}{gain:>8.1%}{'✗' if wrong else 'ok':>7s}")
    print(f"\n条件数 = {len(CASES)}   严重误判数 = {bad}")
    json.dump({"alpha": ALPHA, "gain_thr": a.gain_thr, "n_conditions": len(CASES),
               "n_wrong": bad, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"已写 {a.out}")


if __name__ == "__main__":
    main()
