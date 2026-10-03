"""从实验 JSON 生成论文表格与关键结论（纯文本 + CSV + Markdown）。

用法（项目 src 目录下）:
    ../.venv/bin/python analyze.py --in ../results/raw/main_results.json --out ../results/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from baselines import BASELINE_LABELS

LABEL = dict(BASELINE_LABELS)
LABEL["SA-DUN"] = "SA-DUN"
CFG_ORDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
CFG_LABEL = {"tf_p02": "p=.02", "tf_p05": "p=.05", "tf_p10": "p=.10",
             "tf_p20": "p=.20", "tf_p40": "p=.40", "tf_gauss": "dense"}
METHOD_ORDER = ["SA-DUN", "oracle_a_l1", "ssp_kmeans_l1", "ssp_fcm_sp",
                "pf_auto_l1", "sl0_l1", "duet"]


def _match(value, target) -> bool:
    """数值/字符串通用的匹配（1e-9 容差用于浮点，其余直接用 ==）。"""
    if isinstance(value, str) or isinstance(target, str):
        return value == target
    return abs(float(value) - float(target)) < 1e-9


def ms(records, metric, **filt):
    v = [r[metric] for r in records
         if all(_match(r[k], val) for k, val in filt.items())]
    v = np.asarray(v, dtype=float)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0, v.size)


def table(records, by, metric, by_order=None, fmt="{:.2f}"):
    rows = {}
    keys = sorted({r[by] for r in records}, key=(lambda k: by_order.index(k)) if by_order else None)
    for m in METHOD_ORDER:
        if not any(r["method"] == m for r in records):
            continue
        rows[m] = [ms(records, metric, method=m, **{by: k})[0] for k in keys]
    return keys, rows


def render(title, keys, rows, keyfmt, fmt="{:7.2f}", note=""):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if note:
        print(note)
    hdr = f"{'方法':<26s}" + "".join(f"{keyfmt(k):>9s}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for m, vals in rows.items():
        best = max(vals) if any(v == max(vals) for v in vals) else None
        line = f"{LABEL.get(m, m):<26s}"
        for v in vals:
            mark = "*" if (best is not None and abs(v - best) < 1e-9) else " "
            line += f"{fmt.format(v):>8s}{mark}"
        print(line)
    print("（* 表示该列最优）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../results/raw/main_results.json")
    ap.add_argument("--out", default="../results/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.inp) as f:
        res = json.load(f)["results"]

    lines_out = []

    def emit(s=""):
        print(s)
        lines_out.append(s)

    # ---------- E1 ----------
    if "E1_sparsity" in res:
        recs = res["E1_sparsity"]
        emit("\n" + "=" * 78)
        emit("表 1（E1）混合矩阵估计精度 vs 稀疏度（SNR=20dB, N=4, M=2）")
        emit("=" * 78)
        ks, rows = table(recs, "cfg_key", "A_angle_deg", CFG_ORDER)
        emit(f"{'方法':<26s}" + "".join(f"{CFG_LABEL[k]:>9s}" for k in ks))
        for m, vals in rows.items():
            emit(f"{LABEL.get(m, m):<26s}" + "".join(f"{v:>9.2f}" for v in vals))
        emit("（单位为列平均夹角误差 °，越低越好）")

        emit("")
        emit(f"{'方法':<26s}" + "".join(f"{CFG_LABEL[k]:>9s}" for k in ks))
        ks2, rows2 = table(recs, "cfg_key", "SDR", CFG_ORDER)
        for m, vals in rows2.items():
            emit(f"{LABEL.get(m, m):<26s}" + "".join(f"{v:>9.2f}" for v in vals))
        emit("（SDR dB，越高越好）")

    # ---------- E3 ----------
    if "E3_nsources" in res:
        recs = res["E3_nsources"]
        emit("\n" + "=" * 78)
        emit("表 2（E3）源恢复 SDR vs 源数目 N（SNR=20dB, M=2）")
        emit("=" * 78)
        ks, rows = table(recs, "n_true", "SDR", [3, 4, 5, 6])
        emit(f"{'方法':<26s}" + "".join(f"{'N=' + str(int(k)):>9s}" for k in ks))
        for m, vals in rows.items():
            emit(f"{LABEL.get(m, m):<26s}" + "".join(f"{v:>9.2f}" for v in vals))

    # ---------- E4 ----------
    if "E4_ncount" in res:
        recs = res["E4_ncount"]
        emit("\n" + "=" * 78)
        emit("表 3（E4）源数目估计准确率")
        emit("=" * 78)
        emit(f"{'方法':<26s}{'完全正确率':>12s}{'平均绝对误差':>14s}{'平均估计值':>12s}{'真值均值':>10s}")
        for m in METHOD_ORDER:
            sub = [r for r in recs if r["method"] == m]
            if not sub:
                continue
            ex = np.mean([r["n_exact"] for r in sub])
            ae = np.mean([r["n_abs_err"] for r in sub])
            nh = np.mean([r["n_hat"] for r in sub])
            nt = np.mean([r["n_true"] for r in sub])
            emit(f"{LABEL.get(m, m):<26s}{ex:>12.3f}{ae:>14.3f}{nh:>12.2f}{nt:>10.2f}")

    # ---------- E2 汇总 ----------
    if "E2_snr" in res:
        recs = res["E2_snr"]
        emit("\n" + "=" * 78)
        emit("表 4（E2）各稀疏度下的平均性能（跨 SNR=0..40dB）")
        emit("=" * 78)
        emit(f"{'稀疏度':<10s}" + "".join(f"{LABEL.get(m, m)[:11]:>13s}" for m in METHOD_ORDER
                                          if any(r['method'] == m for r in recs)))
        for k in ["tf_p05", "tf_p20", "tf_p40", "tf_gauss"]:
            line = f"{CFG_LABEL[k]:<10s}"
            for m in METHOD_ORDER:
                if not any(r["method"] == m for r in recs):
                    continue
                v = ms([r for r in recs if r["cfg_key"] == k], "SDR", method=m)[0]
                line += f"{v:>13.2f}"
            emit(line)
        emit("（SDR dB，越高越好）")

    # ---------- E6 ----------
    if "E6_time" in res:
        recs = res["E6_time"]
        emit("\n" + "=" * 78)
        emit("表 5（E6）单问题平均运行时间")
        emit("=" * 78)
        for m in METHOD_ORDER:
            v = [r["time_s"] for r in recs if r["method"] == m]
            if v:
                emit(f"{LABEL.get(m, m):<26s}{np.mean(v):>10.3f} s")

    # ---------- 结论要点 ----------
    emit("\n" + "=" * 78)
    emit("关键结论")
    emit("=" * 78)
    if "E1_sparsity" in res:
        recs = res["E1_sparsity"]
        a_sadun = [ms([r for r in recs if r["cfg_key"] == k], "A_angle_deg", method="SA-DUN")[0]
                   for k in CFG_ORDER]
        a_best = [min(ms([r for r in recs if r["cfg_key"] == k], "A_angle_deg", method=m)[0]
                      for m in METHOD_ORDER if m not in ("SA-DUN", "oracle_a_l1")
                      and any(r["method"] == m for r in recs))
                  for k in CFG_ORDER]
        emit(f"混合矩阵夹角误差：SA-DUN {['%.2f' % v for v in a_sadun]}")
        emit(f"                最优基线 {['%.2f' % v for v in a_best]}")
        win = sum(1 for a, b in zip(a_sadun, a_best) if a < b)
        emit(f"SA-DUN 在 {win}/{len(CFG_ORDER)} 个稀疏度上优于所有基线")
        emit(f"跨稀疏度的波动（标准差）：SA-DUN {np.std(a_sadun):.2f}° vs "
             f"最优基线 {np.std(a_best):.2f}°")

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines_out))
    print(f"\n表格已写入 {os.path.abspath(args.out)}/summary.txt")


if __name__ == "__main__":
    main()
