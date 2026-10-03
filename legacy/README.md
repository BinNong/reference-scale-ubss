# legacy/ — 旧版（SA-DUN 深度展开路线）归档

这些文件属于本课题的**第一版论文定位**：SA-DUN（Sparsity-Adaptive Deep Unfolding
Network），即"深度展开网络 + 可学习稀疏化变换 + 行组收缩定阶"。

## 为什么归档

该路线经实验证伪（详见 `../paper/review_report.md`）：

1. **训练从未真正学到分离**：三个训练 run（full/full2/full3）的验证 SDR 全程为负
   （≈ −0.6 dB），验证 A 误差在训练中反而恶化（7° → 9°）。
2. **端到端结果不可复现**：`full_best.pt` 训练完成后代码又改动（单源门限
   `|cos|^q → 0.90 → 0.97` 等），导致"旧 checkpoint + 当前代码"在 dense 端
   从论文声称的 5.04° 崩到实测 16.64°。仓库里没有任何 checkpoint 能复现论文表格。
3. **消融显示展开网络本身无净收益**：唯一有强支持的组件是"自适应近端算子"，
   而可学习变换与行组收缩**反而有害**（去掉它们指标更好）。

据此论文改为 **MDDE（Multi-gate Directional Density Estimation）**：纯模型驱动、
可复现、不依赖训练。

## 归档内容

- `src/` — 旧的训练 / 实验 / 推理代码
  （train / experiments / ablation / infer / sampler / diagnose / smoke_test /
   config / make_figures / analyze）
- `results/` — 旧版实验数据与图表
  （main_results.json / ablation.json / summary.txt / fig_*.pdf,png）
- `docs/` — 过时的选题论证（01_research_proposal.md，其创新点列表已不成立）

## 现役代码（新论文，位于 `../src/`）

| 文件 | 作用 |
|---|---|
| `model.py` | 盲方向密度估计器 `blind_init_A`（多门限 + 加权球面 K-means） |
| `data.py` | TF 域 UBSS 仿真数据生成 + `normalize_obs` |
| `metrics.py` | 平均夹角误差（真 mean angle）、SDR/SIR/SAR |
| `baselines.py` | SCA-ℓ1 / SCA-SP / 势函数 / SL0 / DUET / Oracle + 去偏 ℓ1 恢复 |
| `experiments_mdde.py` | **自包含实验脚本**（E1–E5，10 seeds，含配对 t 检验） |
| `make_figures_v2.py` | 论文图表（内嵌实测数据，英文标签） |
| `make_architecture_fig.py` | 方法架构图 |

本目录仅供对照查阅，**不应再用于论文**。
