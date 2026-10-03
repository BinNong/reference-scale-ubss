"""消融实验：在已训练模型上逐项关闭组件，量化各创新点的贡献。

变体
----
full            完整模型
no_transform    关闭可学习稀疏化变换（Ψ 固定为单位阵）
no_adaptive     关闭幅值自适应偏置（退化为经典全局软阈值）
no_group        关闭行组收缩（不再自动定阶）
no_debias       关闭支撑集去偏层

注：这是**推理期消融**——即在训练好的参数上关闭各组件，衡量该组件在
测试阶段的实际贡献。它不衡量"如果从训练起就移除该组件会学到什么"，
后者需要为每个变体单独训练；我们在文中明确说明这一点。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import data as D
import metrics as MT
from config import get_cfg
from infer import run_model
from model import build_model

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]


def evaluate_variant(model, device, tag: str, n_seeds: int = 10, rel_thr: float = 0.40) -> list[dict]:
    recs = []
    for key in LADDER:
        for s in range(n_seeds):
            p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=11000 + s)
            r = run_model(model, p["X_all"], device, rel_thr=rel_thr)
            recs.append({
                "variant": tag,
                "cfg_key": key,
                "gini": p["gini"],
                "A_angle_deg": MT.mixing_matrix_angle_error_deg(p["A"], r["A_hat"]),
                "SDR": MT.evaluate_sources(p["S_all"], r["S_hat"])["SDR"],
                "SIR": MT.evaluate_sources(p["S_all"], r["S_hat"])["SIR"],
                "n_hat": r["n_hat"],
                "n_abs_err": abs(r["n_hat"] - p["n"]),
            })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../checkpoints/full_best.pt")
    ap.add_argument("--out", default="../results/raw/ablation.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg()
    base = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    base.load_state_dict(sd["state"])
    base.eval()
    print(f"检查点 {args.ckpt} (epoch={sd.get('epoch')})")

    all_recs = []

    # ---- full ----
    all_recs += evaluate_variant(base, device, "full")

    # ---- no_transform ----
    m = build_model(cfg).to(device)
    m.load_state_dict(sd["state"])
    m.eval()
    with torch.no_grad():
        m.psi.copy_(torch.eye(cfg["n_freq"], device=device))
    m.use_transform = False
    all_recs += evaluate_variant(m, device, "no_transform")

    # ---- no_adaptive ----
    m = build_model(cfg).to(device)
    m.load_state_dict(sd["state"])
    m.use_adaptive_shrink = False
    m.eval()
    all_recs += evaluate_variant(m, device, "no_adaptive")

    # ---- no_group ----
    m = build_model(cfg).to(device)
    m.load_state_dict(sd["state"])
    m.use_group_shrink = False
    m.eval()
    all_recs += evaluate_variant(m, device, "no_group")

    # ---- no_debias ----
    m = build_model(cfg).to(device)
    m.load_state_dict(sd["state"])
    m.use_debias = False
    m.eval()
    all_recs += evaluate_variant(m, device, "no_debias")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_recs, f, indent=2)

    # ---- 汇总打印 ----
    print(f"\n{'=' * 84}")
    print("消融结果（推理期关闭各组件；M=2, N=4, SNR=20dB, 10 seeds）")
    print("=" * 84)
    print(f"{'变体':<14s}" + "".join(f"{k.replace('tf_', ''):>9s}" for k in LADDER)
          + f"{'平均':>9s}{'波动':>8s}")
    for tag in ["full", "no_transform", "no_adaptive", "no_group", "no_debias"]:
        sub = [r for r in all_recs if r["variant"] == tag]
        row = f"{tag:<14s}"
        vals = []
        for k in LADDER:
            v = np.mean([r["A_angle_deg"] for r in sub if r["cfg_key"] == k])
            vals.append(v)
            row += f"{v:>9.2f}"
        row += f"{np.mean(vals):>9.2f}{np.std(vals, ddof=1):>8.2f}"
        print(row)
    print("（数值为混合矩阵夹角误差 °；'波动' 为跨稀疏度的标准差）")

    print()
    print(f"{'变体':<14s}" + "".join(f"{k.replace('tf_', ''):>9s}" for k in LADDER)
          + f"{'平均':>9s}")
    for tag in ["full", "no_transform", "no_adaptive", "no_group", "no_debias"]:
        sub = [r for r in all_recs if r["variant"] == tag]
        row = f"{tag:<14s}"
        vals = []
        for k in LADDER:
            v = np.mean([r["SDR"] for r in sub if r["cfg_key"] == k])
            vals.append(v)
            row += f"{v:>9.2f}"
        row += f"{np.mean(vals):>9.2f}"
        print(row)
    print("（数值为 SDR dB）")


if __name__ == "__main__":
    main()
