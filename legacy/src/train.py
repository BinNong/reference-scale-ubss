"""SA-DUN 训练脚本。

训练完全自监督：损失只用可观测量，不需要干净源标签。
模型选择依据验证集上的无监督损失。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import metrics as MT
from config import get_cfg
from infer import run_model
from model import build_model, unsupervised_loss
from sampler import sample_eval_grid, sample_problems


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * min(prog, 1.0)))


# ----------------------------------------------------------------------
# 验证
# ----------------------------------------------------------------------

@torch.no_grad()
def validate(model, cfg, eval_subset, device, rng, loss_kwargs, n_restarts=2):
    """返回 (无监督验证损失, 有监督诊断指标)。"""
    model.eval()
    # --- 无监督损失 ---
    losses = []
    for _ in range(cfg["val_batches"]):
        b = sample_problems(cfg, cfg["batch_size"], rng, device)
        out = model(b["X"], None)   # 盲初始化
        _, parts = unsupervised_loss(b["X"], out, model, **loss_kwargs)
        losses.append(float(parts["total"]))
    val_loss = float(np.mean(losses))

    # --- 有监督诊断（仅用于报告，不参与模型选择）---
    sdrs, angs, nerr = [], [], []
    for prob in eval_subset:
        res = run_model(model, prob["X_all"], device,
                        n_random=n_restarts - 1 if n_restarts > 1 else 0,
                        rng=rng, loss_kwargs=loss_kwargs)
        n_true = prob["n"]
        m = MT.evaluate_sources(prob["S_all"], res["S_hat"])
        e = MT.mixing_matrix_angle_error_deg(prob["A"][:, :n_true], res["A_hat"])
        sdrs.append(m["SDR"])
        angs.append(e)
        nerr.append(abs(res["n_hat"] - n_true))
    return val_loss, {
        "SDR": float(np.mean(sdrs)),
        "A_angle_deg": float(np.mean(angs)),
        "n_abs_err": float(np.mean(nerr)),
    }


# ----------------------------------------------------------------------
# 主训练循环
# ----------------------------------------------------------------------

def train(cfg: dict, out_dir: str, tag: str = "full", verbose: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["epochs"] * cfg["steps_per_epoch"]
    loss_kwargs = {
        "beta_sp": cfg["beta_sp"],
        "gamma_ind": cfg["gamma_ind"],
        "delta_orth": cfg["delta_orth"],
        "delta_dir": cfg.get("delta_dir", 0.5),
    }

    # 固定验证子集（规模小，训练中反复使用）
    val_grid = sample_eval_grid(cfg, n_per_cell=1, seed=cfg["seed"] + 777)
    val_subset = val_grid[:: max(len(val_grid) // 18, 1)][:18]

    history = []
    best = {"val_loss": float("inf"), "epoch": -1, "state": None, "diag": None}
    step = 0
    t0 = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        ep_losses, ep_parts = [], []
        for _ in range(cfg["steps_per_epoch"]):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, cfg["lr"], cfg["warmup_steps"])

            batch = sample_problems(cfg, cfg["batch_size"], rng, device)
            out = model(batch["X"], None)   # 盲初始化通路
            loss, parts = unsupervised_loss(batch["X"], out, model, **loss_kwargs)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

            ep_losses.append(float(parts["total"]))
            ep_parts.append({k: float(v) for k, v in parts.items() if k != "total"})
            step += 1

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(ep_losses)),
            "lr": lr_at(step, total_steps, cfg["lr"], cfg["warmup_steps"]),
            "sec": round(time.time() - t0, 1),
        }
        for k in ep_parts[0]:
            rec[f"train_{k}"] = float(np.mean([p[k] for p in ep_parts]))

        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"] - 1:
            val_loss, diag = validate(
                model, cfg, val_subset, device, rng, loss_kwargs, n_restarts=2
            )
            rec["val_loss"] = val_loss
            rec.update({f"val_{k}": v for k, v in diag.items()})
            if val_loss < best["val_loss"]:
                best = {
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "diag": diag,
                }
                torch.save(
                    {"cfg": cfg, "state": best["state"], "epoch": epoch, "val_loss": val_loss},
                    os.path.join(out_dir, f"{tag}_best.pt"),
                )
            if verbose:
                print(
                    f"[{tag}] ep{epoch:3d} train={rec['train_loss']:.4f} "
                    f"val={val_loss:.4f} SDR={diag['SDR']:6.2f}dB "
                    f"A_err={diag['A_angle_deg']:5.2f}° n_err={diag['n_abs_err']:.2f} "
                    f"({rec['sec']}s)",
                    flush=True,
                )
        history.append(rec)

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    summary = {
        "tag": tag,
        "n_params": n_params,
        "best_val_loss": best["val_loss"],
        "best_epoch": best["epoch"],
        "best_diag": best["diag"],
        "wall_sec": round(time.time() - t0, 1),
        "device": str(device),
        "history": history,
    }
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return model, summary


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/experiment/paper5_ubss_sadun/checkpoints")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--n_inner", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_transform", action="store_true")
    ap.add_argument("--no_adaptive", action="store_true")
    ap.add_argument("--no_group", action="store_true")
    args = ap.parse_args()

    over = {}
    if args.epochs is not None:
        over["epochs"] = args.epochs
    if args.n_layers is not None:
        over["n_layers"] = args.n_layers
    if args.batch_size is not None:
        over["batch_size"] = args.batch_size
    if args.n_inner is not None:
        over["n_inner"] = args.n_inner
    if args.lr is not None:
        over["lr"] = args.lr
    if args.seed is not None:
        over["seed"] = args.seed
    if args.no_transform:
        over["use_transform"] = False
    if args.no_adaptive:
        over["use_adaptive_shrink"] = False
    if args.no_group:
        over["use_group_shrink"] = False

    cfg = get_cfg(**over)
    print(f"device = {get_device()}", flush=True)
    _, summary = train(cfg, args.out, args.tag)
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
