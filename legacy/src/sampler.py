"""
数据采样器：为训练/验证在线生成 UBSS 问题（保持无监督设定，仅产出观测）。

同时产出初始混合矩阵 A_init（随机、列单位化、带最小夹角约束），
使网络在训练时就见到随机初始化分布，从而对初始化鲁棒。
"""

from __future__ import annotations

import numpy as np
import torch

import data as D


def _rand_min_angle_matrix(M: int, N: int, rng: np.random.Generator, min_deg: float = 12.0):
    return D.gen_mixing_matrix(M, N, rng, min_angle_deg=min_deg)


def sample_problems(
    cfg: dict,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device | str = "cpu",
    eval_mode: bool = False,
) -> dict:
    """生成一批 UBSS 问题。

    Returns
    -------
    dict of torch tensors:
        X        (B, M, L)           归一化后的实数化观测（网络输入）
        S_true   (B, Nmax, L)        真值源（仅评测用；Nmax 之外补零）
        A_true   (B, M, Nmax)        真值混合矩阵（补零列）
        A_init   (B, M, Nmax)        随机初始混合矩阵
        meta     dict                逐样本的 n_true / snr / cfg_key 等
    """
    M = cfg["m_obs"]
    Nmax = cfg["n_max"]
    F, T = cfg["n_freq"], cfg["n_frames"]
    xs, s_true, a_true, a_init = [], [], [], []
    n_list, snr_list, cfg_list = [], [], []

    for _ in range(batch_size):
        n_true = int(rng.integers(cfg["n_min"], cfg["n_max_true"] + 1))
        key = cfg["train_cfgs"][int(rng.integers(0, len(cfg["train_cfgs"])))]
        snr = cfg["snr_choices"][int(rng.integers(0, len(cfg["snr_choices"])))]

        prob = D.make_problem(
            m_obs=M, n_sources=n_true, n_freq=F, n_frames=T,
            cfg_key=key, snr_db=snr, seed=int(rng.integers(1 << 31)),
            min_angle_deg=cfg.get("min_angle_deg", 12.0),
        )

        X = prob["X_all"]                                    # (M, L)
        # 归一化到单位 RMS，稳定训练；SDR 对尺度不敏感，故不影响评测
        scale = float(np.sqrt(np.mean(X ** 2))) + 1e-12
        xs.append((X / scale).astype(np.float32))

        St = np.zeros((Nmax, prob["S_all"].shape[1]), dtype=np.float32)
        St[:n_true] = prob["S_all"].astype(np.float32)
        s_true.append(St)

        At = np.zeros((M, Nmax), dtype=np.float32)
        At[:, :n_true] = prob["A"].astype(np.float32)
        a_true.append(At)

        A0 = _rand_min_angle_matrix(M, Nmax, rng, cfg.get("min_angle_deg", 12.0))
        a_init.append(A0.astype(np.float32))

        n_list.append(n_true)
        snr_list.append(-1.0 if snr is None else float(snr))
        cfg_list.append(key)

    to = lambda arr, dt=torch.float32: torch.as_tensor(np.stack(arr), dtype=dt, device=device)
    return {
        "X": to(xs),
        "S_true": to(s_true),
        "A_true": to(a_true),
        "A_init": to(a_init),
        "meta": {"n_true": n_list, "snr_db": snr_list, "cfg_key": cfg_list},
    }


def sample_eval_grid(cfg: dict, n_per_cell: int = 4, seed: int = 20260910) -> list[dict]:
    """构造固定评测集：稀疏度阶梯 × SNR 网格，每个格子 n_per_cell 个样本。"""
    rng = np.random.default_rng(seed)
    M, Nmax = cfg["m_obs"], cfg["n_max"]
    F, T = cfg["n_freq"], cfg["n_frames"]
    items = []
    for key in cfg["eval_cfgs"]:
        for snr in cfg["eval_snrs"]:
            for _ in range(n_per_cell):
                n_true = int(rng.integers(cfg["n_min"], cfg["n_max_true"] + 1))
                prob = D.make_problem(
                    m_obs=M, n_sources=n_true, n_freq=F, n_frames=T,
                    cfg_key=key, snr_db=snr, seed=int(rng.integers(1 << 31)),
                    min_angle_deg=cfg.get("min_angle_deg", 12.0),
                )
                A0 = _rand_min_angle_matrix(M, Nmax, rng, cfg.get("min_angle_deg", 12.0))
                prob["A_init"] = A0
                prob["cfg_key_eval"] = key
                prob["snr_eval"] = snr
                items.append(prob)
    return items
