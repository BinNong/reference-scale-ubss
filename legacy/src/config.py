"""SA-DUN 实验配置。"""

from __future__ import annotations

import copy

# 主实验设置：M=2（欠定，也是 DUET 等经典方法的标准场景）
BASE_CFG: dict = {
    # ---- 问题规模 ----
    "m_obs": 2,
    "n_max": 8,          # 网络维护的最大源数目
    "n_min": 3,          # 训练时真实源数目下界
    "n_max_true": 6,     # 训练时真实源数目上界
    "n_freq": 33,
    "n_frames": 64,
    "min_angle_deg": 12.0,

    # ---- 网络 ----
    "n_layers": 12,
    "n_inner": 8,          # 每层内层近端梯度步数
    "use_transform": True,
    "use_adaptive_shrink": True,
    "use_group_shrink": True,

    # ---- 训练分布 ----
    "train_cfgs": ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"],
    "snr_choices": [10.0, 20.0, 30.0, 40.0, None],

    # ---- 优化 ----
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "steps_per_epoch": 100,
    "epochs": 120,
    "grad_clip": 1.0,
    "warmup_steps": 200,

    # ---- 损失权重 ----
    "beta_sp": 0.05,
    "gamma_ind": 0.5,     # 独立性项：诊断显示这是唯一有区分力的项
    "delta_orth": 0.1,
    "delta_dir": 0.05,     # 方向覆盖损失：把混合矩阵锚定到观测方向结构

    # ---- 验证 ----
    "val_batches": 6,
    "val_every": 5,
    "eval_cfgs": ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"],
    "eval_snrs": [10.0, 20.0, 30.0],
    "eval_per_cell": 4,
    "eval_restarts": 0,   # 推理时额外随机重启次数（0 = 仅用盲初始化通路）

    "seed": 42,
}


def get_cfg(**overrides) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    cfg.update(overrides)
    return cfg


# 消融配置：逐个关闭创新点
def ablation_cfgs() -> dict[str, dict]:
    return {
        "full":            get_cfg(),
        "no_transform":     get_cfg(use_transform=False),
        "no_adaptive":      get_cfg(use_adaptive_shrink=False),
        "no_group":         get_cfg(use_group_shrink=False),
        "no_transform_no_adaptive": get_cfg(use_transform=False, use_adaptive_shrink=False),
    }
