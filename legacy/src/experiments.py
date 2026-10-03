"""实验套件：跑完整评测矩阵并存盘。

用法（项目 src 目录下）:
    ../.venv/bin/python experiments.py --ckpt ../checkpoints/full_best.pt --out ../results

实验编号
--------
E1  混合矩阵估计精度 vs 稀疏度
E2  源恢复质量（SDR/SIR）vs SNR
E3  源恢复质量 vs 源数目
E4  源数目估计准确率
E5  消融实验
E6  运行时间统计
E7  跨分布泛化（训练/测试条件失配）

所有结果以「长表」形式写入 JSON，便于后续分析与绘图。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import data as D
import metrics as MT
from baselines import BASELINE_LABELS, BASELINE_NAMES, run_method
from config import get_cfg
from infer import run_model
from model import build_model
from sampler import sample_eval_grid

# ----------------------------------------------------------------------
# 单次评测
# ----------------------------------------------------------------------


def eval_one(prob: dict, method: str, cfg: dict, model, device, rel_thr: float = 0.40) -> dict:
    """在单个问题上评测一个方法，返回指标字典。"""
    n_true = prob["n"]
    t0 = time.perf_counter()
    if method == "SA-DUN":
        r = run_model(model, prob["X_all"], device, rel_thr=rel_thr)
        A_hat, S_hat, n_hat = r["A_hat"], r["S_hat"], r["n_hat"]
        extra = {"best_loss": r["best_loss"]}
    else:
        r = run_method(method, prob, seed=prob.get("seed", 0) or 0)
        A_hat, S_hat, n_hat = r["A_hat"], r["S_hat"], r["n_hat"]
        extra = {"ssp_count": r.get("ssp_count", -1)}
    dt = time.perf_counter() - t0

    m = MT.evaluate_sources(prob["S_all"], S_hat)
    rec = {
        "cfg_key": prob.get("cfg_key_eval", prob.get("cfg_key")),
        "snr_db": prob.get("snr_eval", prob.get("snr_db")),
        "n_true": n_true,
        "method": method,
        "gini": prob["gini"],
        "overlap": prob["overlap"],
        "wdo": prob["wdo_violation_emp"],
        "SDR": m["SDR"],
        "SIR": m["SIR"],
        "SAR": m["SAR"],
        "n_spurious": m["n_spurious"],
        "n_hat": int(n_hat),
        "n_abs_err": abs(int(n_hat) - n_true),
        "n_exact": int(int(n_hat) == n_true),
        "A_angle_deg": MT.mixing_matrix_angle_error_deg(prob["A"], A_hat),
        "A_nmse": MT.mixing_matrix_nmse(prob["A"], A_hat),
        "snr_out": MT.output_snr(prob["S_all"], S_hat),
        "time_s": dt,
    }
    rec.update(extra)
    return rec


def make_prob(m, n, cfg_key, snr, seed, cfg):
    p = D.make_problem(
        m_obs=m, n_sources=n, n_freq=cfg["n_freq"], n_frames=cfg["n_frames"],
        cfg_key=cfg_key, snr_db=snr, seed=seed, min_angle_deg=cfg["min_angle_deg"],
    )
    p["cfg_key_eval"] = cfg_key
    p["snr_eval"] = snr
    p["seed"] = seed
    return p


ALL_METHODS = ["SA-DUN"] + BASELINE_NAMES


def run_suite(model, device, out_path: str, only: list[str] | None = None) -> dict:
    cfg = get_cfg()
    M = cfg["m_obs"]
    results: dict[str, list] = {}
    t_start = time.time()

    def log(tag, recs):
        results.setdefault(tag, []).extend(recs)
        if recs:
            sdr = np.mean([r["SDR"] for r in recs if r["method"] == "SA-DUN"]) \
                if any(r["method"] == "SA-DUN" for r in recs) else float("nan")
            print(f"  [{tag}] {len(recs)} 条记录  SA-DUN 平均 SDR={sdr:.2f} dB", flush=True)

    def do(tag, probs, methods):
        recs = []
        for i, p in enumerate(probs):
            for meth in methods:
                try:
                    recs.append(eval_one(p, meth, cfg, model, device))
                except Exception as e:
                    print(f"    !! {meth} 失败: {type(e).__name__}: {e}", flush=True)
        log(tag, recs)

    # ---------------- E1：vs 稀疏度 ----------------
    if only is None or "E1" in only:
        print("E1 混合矩阵精度 vs 稀疏度 ...", flush=True)
        probs = []
        for k in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]:
            for s in range(3):
                probs.append(make_prob(M, 4, k, 20.0, 1000 + s, cfg))
        do("E1_sparsity", probs, ALL_METHODS)

    # ---------------- E2：vs SNR ----------------
    if only is None or "E2" in only:
        print("E2 源恢复 vs SNR ...", flush=True)
        probs = []
        for k in ["tf_p05", "tf_p20", "tf_p40", "tf_gauss"]:
            for snr in [0.0, 10.0, 20.0, 30.0, 40.0]:
                for s in range(3):
                    probs.append(make_prob(M, 4, k, snr, 2000 + s, cfg))
        do("E2_snr", probs, ALL_METHODS)

    # ---------------- E3：vs 源数目 ----------------
    if only is None or "E3" in only:
        print("E3 源恢复 vs 源数目 ...", flush=True)
        probs = []
        for k in ["tf_p10", "tf_p20", "tf_gauss"]:
            for n in [3, 4, 5, 6]:
                for s in range(3):
                    probs.append(make_prob(M, n, k, 20.0, 3000 + s, cfg))
        do("E3_nsources", probs, ALL_METHODS)

    # ---------------- E4：源数目估计准确率 ----------------
    if only is None or "E4" in only:
        print("E4 源数目估计准确率 ...", flush=True)
        probs = []
        for k in ["tf_p02", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]:
            for n in [3, 4, 5, 6]:
                for s in range(4):
                    probs.append(make_prob(M, n, k, 20.0, 4000 + s, cfg))
        do("E4_ncount", probs, ["SA-DUN", "pf_auto_l1"])

    # ---------------- E6：运行时间（复用 E1 记录中的 time_s，另跑小规模复核）----------------
    if only is None or "E6" in only:
        print("E6 运行时间复核 ...", flush=True)
        probs = [make_prob(M, 4, "tf_p10", 20.0, 5000 + s, cfg) for s in range(3)]
        do("E6_time", probs, ALL_METHODS)

    summary = {
        "elapsed_sec": time.time() - t_start,
        "n_records": sum(len(v) for v in results.values()),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print(f"\n结果已写入 {out_path}  共 {summary['n_records']} 条，用时 {summary['elapsed_sec']:.0f}s")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="SA-DUN 检查点路径；不传则用未训练模型")
    ap.add_argument("--out", default="../results/raw/main_results.json")
    ap.add_argument("--only", default=None, help="只跑指定实验，如 E1,E2")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg()
    model = build_model(cfg).to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd["state"])
        print(f"已加载检查点 {args.ckpt} (epoch={sd.get('epoch')}, val_loss={sd.get('val_loss'):.4f})")
    else:
        print("警告：未加载检查点，使用未训练模型")
    model.eval()

    only = args.only.split(",") if args.only else None
    run_suite(model, device, args.out, only)


if __name__ == "__main__":
    main()
