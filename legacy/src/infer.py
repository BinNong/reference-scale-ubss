"""推理封装：盲初始化 + 可选随机重启，按无监督损失择优，再做源数目剪枝。"""

from __future__ import annotations

import numpy as np
import torch

import data as D
from baselines import kmeans_sphere, ssp_directions, ssp_mask_from_complex
from model import SADUN, unsupervised_loss


def normalize_obs(X: np.ndarray) -> np.ndarray:
    s = float(np.sqrt(np.mean(X ** 2))) + 1e-12
    return X / s


def _loss_of(Xt: torch.Tensor, A: torch.Tensor, S: torch.Tensor, model: SADUN, lk: dict) -> float:
    o = {"A_hat": A, "S_hat": S}
    _, parts = unsupervised_loss(Xt, o, model, **lk)
    return float(parts["total"])


@torch.no_grad()
def run_model(
    model: SADUN,
    X_all: np.ndarray,
    device: torch.device | str = "cpu",
    rel_thr: float = 0.40,
    n_random: int = 0,
    use_classical_init: bool = False,
    rng: np.random.Generator | None = None,
    loss_kwargs: dict | None = None,
    transform_output: bool = True,
    return_all: bool = False,
) -> dict:
    """在单个问题上运行网络。

    Parameters
    ----------
    X_all : (M, L) 实数化观测
    n_random : int
        额外加入的随机初始化重启次数。主候选始终是模型的**盲初始化**通路。
    transform_output : bool
        是否把源估计从变换域映射回原时频域。

    Returns
    -------
    dict: A_hat (M,N̂), S_hat (N̂,L), n_hat, best_loss, all_losses
    """
    model.eval()
    device = torch.device(device)
    Xn = normalize_obs(X_all)
    Xt = torch.as_tensor(Xn[None], dtype=torch.float32, device=device)
    lk = loss_kwargs or {}
    if rng is None:
        rng = np.random.default_rng(0)

    cands: list[tuple[float, torch.Tensor, torch.Tensor]] = []

    # 候选 1：模型内置的可微盲初始化（在（可学习）变换域中搜索方向密度峰）
    out = model(Xt, None)
    cands.append((_loss_of(Xt, out["A_hat"], out["S_hat"], model, lk), out["A_hat"], out["S_hat"]))

    # 可选候选：经典 SSP + K-means（在变换域上进行）。
    # 注意：无监督损失对混合矩阵质量的排序并不可靠（诊断已验证），
    # 故默认不启用该候选，使推理路径唯一且确定。
    try:
        if not use_classical_init:
            raise RuntimeError("skip")
        F, L2 = model.F, Xt.shape[2]
        T = (L2 // 2) // F
        with torch.no_grad():
            Xw_np = model._apply_psi(Xt).detach().cpu().numpy()[0]
        Xw_tf = D.real_to_tf(Xw_np, F, T)
        U = ssp_directions(Xw_tf, ssp_mask_from_complex(Xw_tf))
        if U.shape[1] >= model.Nmax:
            A_cls = kmeans_sphere(U, model.Nmax, rng)
            At = torch.as_tensor(A_cls[None], dtype=torch.float32, device=device)
            o2 = model(Xt, At)
            cands.append(
                (_loss_of(Xt, o2["A_hat"], o2["S_hat"], model, lk), o2["A_hat"], o2["S_hat"])
            )
    except Exception:
        pass

    # 候选 3+：随机初始化重启
    if n_random > 0:
        inits = [
            D.gen_mixing_matrix(model.M, model.Nmax, rng, min_angle_deg=12.0)
            for _ in range(n_random)
        ]
        At0 = torch.as_tensor(np.stack(inits), dtype=torch.float32, device=device)
        out3 = model(Xt.expand(len(inits), -1, -1), At0)
        for b in range(len(inits)):
            A_b = out3["A_hat"][b:b + 1]
            S_b = out3["S_hat"][b:b + 1]
            cands.append((_loss_of(Xt, A_b, S_b, model, lk), A_b, S_b))

    losses = [c[0] for c in cands]
    best = int(np.argmin(losses))
    _, A_best, S_best = cands[best]

    if transform_output:
        S_best = model.inverse_transform(S_best)

    A_sel, S_sel, n_list = model.prune(A_best, S_best, rel_thr=rel_thr)

    res = {
        "A_hat": A_sel[0].cpu().numpy(),
        "S_hat": S_sel[0].cpu().numpy(),
        "n_hat": int(n_list[0]),
        "best_loss": float(losses[best]),
        "all_losses": losses,
        "best_candidate": best,
    }
    if return_all:
        res["candidates"] = [
            {"loss": l, "A": a[0].cpu().numpy(), "S": s[0].cpu().numpy()} for l, a, s in cands
        ]
    return res
