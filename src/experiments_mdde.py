"""MDDE 论文实验脚本（自包含、可复现、CPU 即可跑）。

⚠️ 遗留代码（2026-09-16 标注）
----------------------------
**MDDE（深度展开）路线已放弃。** 手稿 `paper/manuscript.md` 现以 NF-SSP
（噪声底标定门限）为主线，本文件与 `model.py` 产出的
`mdde_results.json` / `mdde_ablation.json` **不属于当前论文的任何表**，
保留仅为可追溯，不再更新。

仍被引用的地方只有两处，且都只借 `make_model()` 调 `blind_init_A`：
  · `theory_validate_data.py` —— §6 的 π0/π1/p_c/κ 验证
    （对照用硬阈值 0.98，与本文件的 MDDE 前向无关）；
  · `fairness_check.py` —— 复算"经典流水线@最优阈值"的对照网格。

当前论文的实验脚本是 `experiments_nfr.py`（合成）、`experiments_real.py`（真实语音）、
`scale_invariance.py`、`weighted_variant.py` / `rerun_synth.py`（加权对照）。

原 docstring
-----------
主方法：盲方向密度估计器（多门限 + 加权球面 K-means，给定源数目 N）
        + 去偏 ℓ1 源恢复（与所有基线完全相同的恢复器）。

实验：
    E1  A误差/SDR vs 稀疏度（6 档）+ 与基线的配对 t 检验
    E2  A误差 vs SNR
    E3  A误差/SDR vs 源数目 N
    E4  三门限消融（逐项关闭）
    E5  运行时间

用法（项目 src 目录下）:
    ../.venv/bin/python experiments_mdde.py --out ../results/mdde_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from scipy import stats

import data as D
import metrics as MT
from baselines import BASELINE_NAMES, run_method
from model import SADUN

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
LABEL = {"ssp_kmeans_l1": "SCA-L1", "ssp_fcm_sp": "SCA-SP",
         "pf_auto_l1": "PF-auto", "sl0_l1": "SL0", "duet": "DUET",
         "oracle_a_l1": "Oracle-A"}


def make_model() -> SADUN:
    """构造只用于盲初始化的估计器。

    调用方（本文件的 run_mdde、`theory_validate_data.py`、`fairness_check.py`）
    只使用 `blind_init_A` 与 `_apply_psi`，**不使用展开网络的任何一层**——
    仓库里没有任何代码路径调用 SADUN 的前向。
    原先 n_layers=12 会构造 6 组长度为 12 的逐层参数张量（eta/mu/lam/tau/p/nu），
    这些张量根本不参与计算，故降为 1 以免误导。
    （若将来要复活 MDDE 前向，须把 n_layers 改回 12。）
    """
    return SADUN(m_obs=2, n_max=8, n_layers=1, n_freq=33)


@torch.no_grad()
def run_mdde(prob: dict, model: SADUN, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """主方法：盲方向密度估计 (给定 N) + 去偏 ℓ1 恢复。"""
    Xn = D.normalize_obs(prob["X_all"])
    Xt = torch.as_tensor(Xn[None], dtype=torch.float32, device=device)
    A_hat = model.blind_init_A(model._apply_psi(Xt), n_clusters=prob["n"])[0].cpu().numpy()
    S_hat = MT_l1(A_hat, Xn)
    return A_hat, S_hat


def MT_l1(A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    """去偏 ℓ1 恢复（与基线同一实现，见 baselines.l1_recover）。"""
    from baselines import l1_recover
    return l1_recover(A_hat, X)


def eval_case(prob: dict, method: str, model: SADUN, device: torch.device,
              seed: int = 0, experiment: str = "E1") -> dict:
    t0 = time.perf_counter()
    if method == "MDDE":
        A_hat, S_hat = run_mdde(prob, model, device)
        n_hat = prob["n"]
    else:
        r = run_method(method, prob, seed=seed)
        A_hat, S_hat, n_hat = r["A_hat"], r["S_hat"], r["n_hat"]
    dt = time.perf_counter() - t0
    m = MT.evaluate_sources(prob["S_all"], S_hat)
    return {
        "experiment": experiment,
        "seed": seed,
        "method": method,
        "cfg_key": prob["cfg_key_eval"],
        "snr_db": prob["snr_eval"],
        "n_true": prob["n"],
        "A_angle_deg": MT.mixing_matrix_angle_error_deg(prob["A"], A_hat),
        "SDR": m["SDR"],
        "SIR": m["SIR"],
        "n_hat": int(n_hat),
        "time_s": dt,
    }


def make_prob(n: int, cfg_key: str, snr: float, seed: int) -> dict:
    p = D.make_problem(m_obs=2, n_sources=n, n_freq=33, n_frames=64,
                       cfg_key=cfg_key, snr_db=snr, seed=seed, min_angle_deg=12.0)
    p["cfg_key_eval"] = cfg_key
    p["snr_eval"] = snr
    p["seed"] = seed
    return p


def paired_p(recs, key, metric, m_a="MDDE", m_b="ssp_kmeans_l1") -> float:
    a = [r[metric] for r in recs if r["method"] == m_a and r["cfg_key"] == key]
    b = [r[metric] for r in recs if r["method"] == m_b and r["cfg_key"] == key]
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    return float(stats.ttest_rel(a, b).pvalue)


def run(n_seeds: int, out_path: str) -> dict:
    device = torch.device("cpu")
    model = make_model().to(device)
    model.eval()
    recs: list[dict] = []
    methods = ["MDDE"] + BASELINE_NAMES

    def do(probs, experiment):
        # 组内索引作为随机种子：保证同一 group 内 seed = 0..n-1，
        # 与论文表格所用的种子约定一致（可复现）。
        for i, p in enumerate(probs):
            for mth in methods:
                try:
                    recs.append(eval_case(p, mth, model, device, seed=i, experiment=experiment))
                except Exception as e:  # 单个方法失败不应中断整轮
                    print(f"  !! {mth} 失败: {type(e).__name__}: {e}")

    # E1 / E5
    print("E1: A误差/SDR vs 稀疏度 ...", flush=True)
    for key in LADDER:
        do([make_prob(4, key, 20.0, 1000 + s) for s in range(n_seeds)], "E1_sparsity")

    # E2
    print("E2: A误差 vs SNR ...", flush=True)
    for key in ["tf_p05", "tf_p20", "tf_gauss"]:
        for snr in [0.0, 10.0, 20.0, 30.0, 40.0]:
            do([make_prob(4, key, snr, 2000 + s) for s in range(n_seeds)], "E2_snr")

    # E3
    print("E3: A误差/SDR vs 源数目 ...", flush=True)
    for n in [3, 4, 5, 6]:
        do([make_prob(n, "tf_p10", 20.0, 3000 + s) for s in range(n_seeds)], "E3_nsources")

    summary = {
        "n_seeds": n_seeds,
        "n_records": len(recs),
        # 关键显著性（MDDE vs SCA-L1 的 A 误差，逐稀疏度）
        "p_value_MDDE_vs_SCA_L1_A": {k: paired_p(recs, k, "A_angle_deg") for k in LADDER},
    }

    # ---- 打印 E1 / E3 表格（供与手稿核对）----
    def cell(exp, key, method, metric, n=4, snr=20.0):
        v = [r[metric] for r in recs if r["experiment"] == exp and r["cfg_key"] == key
             and r["method"] == method and r["n_true"] == n and r["snr_db"] == snr]
        return f"{np.mean(v):.2f}" if v else "-"

    print("\n=== E1 表格: A误差(°) ===")
    print(f"{'method':<11s}" + "".join(f"{k.replace('tf_', 'p='):>9s}" for k in LADDER))
    for m in methods:
        print(f"{LABEL.get(m, m):<11s}" + "".join(f"{cell('E1_sparsity', k, m, 'A_angle_deg'):>9s}" for k in LADDER))
    print("=== E1 表格: SDR(dB) ===")
    for m in methods:
        print(f"{LABEL.get(m, m):<11s}" + "".join(f"{cell('E1_sparsity', k, m, 'SDR'):>9s}" for k in LADDER))
    print("=== E3 表格: A误差(°) vs 源数目 N (tf_p10) ===")
    print(f"{'method':<11s}" + "".join(f"{'N=' + str(n):>9s}" for n in [3, 4, 5, 6]))
    for m in methods:
        row = ""
        for N in [3, 4, 5, 6]:
            v = [r["A_angle_deg"] for r in recs
                 if r["experiment"] == "E3_nsources" and r["method"] == m and r["n_true"] == N]
            row += f"{np.mean(v):>9.2f}" if v else f"{'-':>9s}"
        print(f"{LABEL.get(m, m):<11s}{row}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"records": recs, "summary": summary}, f, indent=2)
    print(f"\n结果写入 {out_path}（{len(recs)} 条）")
    return summary


def run_ablation(n_seeds: int) -> dict:
    """E4：三门限消融（同一估计器上逐项关闭）。"""
    device = torch.device("cpu")
    model = make_model().to(device)
    model.eval()
    variants = {"full": {}, "no_ssp": dict(use_ssp=False),
                "no_en": dict(use_en=False), "no_bal": dict(use_bal=False)}
    out = {}
    print("E4: 三门限消融 ...", flush=True)
    for tag, kw in variants.items():
        vals = []
        for key in LADDER:
            for s in range(n_seeds):
                p = make_prob(4, key, 20.0, 1000 + s)
                Xn = D.normalize_obs(p["X_all"])
                with torch.no_grad():
                    Xt = torch.as_tensor(Xn[None], dtype=torch.float32, device=device)
                    A = model.blind_init_A(model._apply_psi(Xt), n_clusters=4, **kw)[0].cpu().numpy()
                vals.append(MT.mixing_matrix_angle_error_deg(p["A"], A))
        out[tag] = float(np.mean(vals))
        print(f"  {tag:<8s} 平均 A误差 = {out[tag]:.2f}°")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="../results/mdde_results.json")
    ap.add_argument("--ablation-out", default="../results/mdde_ablation.json")
    args = ap.parse_args()

    s = run(args.seeds, args.out)
    ab = run_ablation(args.seeds)
    os.makedirs(os.path.dirname(args.ablation_out), exist_ok=True)
    with open(args.ablation_out, "w") as f:
        json.dump(ab, f, indent=2)
    print("\n显著性 p 值 (MDDE vs SCA-L1, A误差):")
    for k, v in s["p_value_MDDE_vs_SCA_L1_A"].items():
        print(f"  {k:<10s} p = {v:.4f}")
    print("消融:", ab)


if __name__ == "__main__":
    main()
