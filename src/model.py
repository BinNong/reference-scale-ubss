"""
SA-DUN: Sparsity-Adaptive Deep Unfolding Network for Underdetermined BSS.

模型
----
求解如下联合优化问题（W = X 为实数化后的观测）：

    min_{A∈𝒜, S}  0.5·||W - A S||_F²  +  Σ_{n,l} φ_{θ_k}(S_{n,l})  +  Σ_n ν_k·||S_n||_2

其中 𝒜 = {A : ||a_n||_2 = 1}（球面约束，消除尺度模糊）。

用近端交替迭代求解，并把 K 次迭代展开为 K 层网络：

    Ŝᵏ = 𝒫_{θ_k}( S^{k-1} - η_k·Aᵀ(A S^{k-1} - W) )      ← 学习型自适应近端算子
    Ŝᵏ ← Ŝᵏ · relu(1 - ν_k / ||Ŝᵏ_n||₂)                    ← 行组收缩（源数目自适应）
    Aᵏ = Π_𝒜( A^{k-1} - μ_k·(A^{k-1} Ŝᵏ - W) Ŝᵏᵀ )        ← 球面投影梯度

四个创新点对应实现：
  (1) 可学习稀疏化变换  —— self.psi（频域正交线性变换，作用于频率轴，与 A 可交换）
  (2) 学习型自适应近端算子 —— 阈值随系数幅值自适应衰减，γ_k 可学习（消解 ℓ1 幅值收缩偏差）
  (3) 行组收缩 + 列范数剪枝 —— 源数目在优化中自然涌现
  (4) 自监督目标（见 unsupervised_loss）—— 仅依赖可观测量
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y: float) -> float:
    """softplus(x) = y 的逆，用于把参数初始化到期望的有效值。"""
    return math.log(math.expm1(y)) if y > 1e-8 else -20.0


def softplus_pos(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(x) + eps


class SADUN(nn.Module):
    """展开式欠定盲分离网络。

    Parameters
    ----------
    m_obs : int
        观测通道数 M。
    n_max : int
        网络维护的最大源数目 N_max（≥ 真实源数目）。多余列会被行组收缩自动剔除。
    n_layers : int
        展开层数 K。
    n_freq : int
        频率维长度 F，用于构造可学习频域变换（use_transform=True 时必需）。
    use_transform : bool
        是否启用创新点 (1) 的可学习稀疏化变换。
    use_adaptive_shrink : bool
        是否启用创新点 (2) 的幅值自适应阈值（False 时退化为经典全局软阈值，用于消融）。
    use_group_shrink : bool
        是否启用创新点 (3) 的行组收缩（False 时无法自动定阶，用于消融）。
    """

    def __init__(
        self,
        m_obs: int,
        n_max: int = 8,
        n_layers: int = 12,
        n_freq: int = 33,
        n_inner: int = 3,
        n_init_iter: int = 12,
        use_debias: bool = True,
        debias_rel: float = 0.1,
        use_transform: bool = True,
        use_adaptive_shrink: bool = True,
        use_group_shrink: bool = True,
    ) -> None:
        super().__init__()
        self.M = m_obs
        self.Nmax = n_max
        self.K = n_layers
        self.F = n_freq
        self.n_inner = n_inner
        self.n_init_iter = n_init_iter
        self.use_debias = use_debias
        self.debias_rel = debias_rel
        self.use_transform = use_transform
        self.use_adaptive_shrink = use_adaptive_shrink
        self.use_group_shrink = use_group_shrink

        k = n_layers

        # ---- 每层的可学习超参（经软正映射保证合法）----
        # S 步长 η_k（无量纲，前向中按 ‖AᵀA‖₂ 归一化）
        self.eta_raw = nn.Parameter(torch.full((k,), inv_softplus(1.0)))
        # A 更新中「最小二乘字典更新」的混合比例 μ_k ∈ (0,1)
        self.mu_raw = nn.Parameter(torch.full((k,), -3.9))   # sigmoid≈0.02，精修须保守
        # 字典更新阻尼 λ_k（保证 SSᵀ+λI 可逆）
        self.lam_raw = nn.Parameter(torch.full((k,), inv_softplus(1e-3)))
        # 近端算子阈值尺度 τ_k 与形状指数 p_k
        self.tau_raw = nn.Parameter(torch.full((k,), inv_softplus(0.25)))
        self.p_raw = nn.Parameter(torch.zeros(k))
        # 行组收缩阈值 ν_k（作用于行 RMS）
        self.nu_raw = nn.Parameter(torch.full((k,), inv_softplus(0.05)))

        # ---- 逐源阈值尺度 ρ_n ----
        self.rho_raw = nn.Parameter(torch.zeros(n_max))

        # ---- 盲初始化：方向密度峰的核带宽 σ_init 与软单源权重锐化指数 ----
        self.sigma_init_raw = nn.Parameter(torch.full((), inv_softplus(0.12)))
        self.sigma_def_raw = nn.Parameter(torch.full((), inv_softplus(0.04)))
        self.ssp_pow_raw = nn.Parameter(torch.full((), inv_softplus(4.0)))
        # 单源判据软阈值位置。与手稿的经典硬阈值统一取 c0 = 0.98。
        # 原先取 0.97（属已放弃的 MDDE 路线）会造成"代码与论文声明的 c0 不一致"，
        # 现对齐。改动只影响本类的软单源门限；theory_validate_data.py 的
        # p_c / kappa 对照用的是硬阈值 0.98，不受影响。
        self.cos_thr = float(0.98)
        self.cos_soft = float(0.03)   # 软阈值过渡宽度
        self.rel_gate = float(0.25)   # 实/虚部均衡门限的相对尺度
        self.egate_tau = float(10.0)  # 能量门限：相对中位数能量的倍数
        self.egate_sigma = float(3.0) # 能量门限的软化宽度

        # ---- 创新点 (1)：可学习频域线性变换 Ψ，初始化为恒等 ----
        if use_transform:
            self.psi = nn.Parameter(torch.eye(n_freq))
        else:
            self.register_buffer("psi", torch.eye(n_freq))

    # ------------------------------------------------------------------
    # 可学习稀疏化变换
    # ------------------------------------------------------------------

    def _apply_psi(self, X: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        """沿频率轴施加 Ψ。

        输入 X : (B, C, L)，其中 L = 2·F·T，前 F·T 为实部、后 F·T 为虚部。
        由 A 与频率无关可知 Ψ 与 A 可交换，故变换不破坏混合模型结构。
        """
        if not self.use_transform:
            return X
        B, C, L = X.shape
        FT = L // 2
        T = FT // self.F
        P = self.psi.T if transpose else self.psi          # (F, F)
        re = X[:, :, :FT].reshape(B, C, self.F, T)
        im = X[:, :, FT:].reshape(B, C, self.F, T)
        re_t = torch.einsum("qf,bcft->bcqt", P, re)
        im_t = torch.einsum("qf,bcft->bcqt", P, im)
        return torch.cat([re_t.reshape(B, C, FT), im_t.reshape(B, C, FT)], dim=2)

    def transform_penalty(self) -> torch.Tensor:
        """约束 Ψ 接近正交，保证可逆（正交变换保持能量，且逆即转置）。"""
        if not self.use_transform:
            return torch.zeros((), device=self.eta_raw.device)
        eye = torch.eye(self.F, device=self.psi.device, dtype=self.psi.dtype)
        return ((self.psi @ self.psi.T - eye) ** 2).mean()

    def inverse_transform(self, S: torch.Tensor) -> torch.Tensor:
        """把变换域中的源估计映射回原时频域（Ψ 正交时逆即转置）。"""
        return self._apply_psi(S, transpose=True)

    # ------------------------------------------------------------------
    # 近端算子
    # ------------------------------------------------------------------

    def _prox(self, Z: torch.Tensor, k: int) -> torch.Tensor:
        """学习型自适应近端算子（**相对阈值**）。

        关键设计：阈值以**逐时频点的最大系数幅值**为基准，而不是绝对常数。

            s_l      = max_n |Z_{n,l}|                     逐样本尺度
            θ_{n,l}  = softplus(τ_k) · softplus(ρ_n) · s_l  逐源、逐点阈值
            bias(z)  = θ · (1 + |z|/θ)^(-p),  p = sigmoid(p_k) ∈ (0,1)
            S        = sign(z) · relu(|z| - bias)

        这样做有两个好处：
        * **尺度无关**：阈值自动跟随数据的量级，不会因为绝对阈值设得过小而形同虚设
          （这正是早期版本失败的原因——阈值 ≈0.055 而系数 ≈0.85，实际没有稀疏作用）。
        * **语义正确**：只保留与当前时频点主导源可比的分量，等价于在每一点上自适应地
          判定"哪几个源活跃"，从而在非稀疏条件下也能给出稀疏解。

        偏置项随幅值衰减：|z| ≫ θ 时 bias → 0，大系数几乎不被压缩，
        从而消除经典 ℓ1 的幅值收缩偏差。
        """
        scale = Z.abs().amax(dim=1, keepdim=True)               # (B,1,L) 逐样本最大幅值
        # 注意广播形状：rho 必须写成 (1,N,1)，否则会与 (B,1,L) 广播出 (N,N,L)
        rho = softplus_pos(self.rho_raw).view(1, -1, 1)          # (1,N,1)
        theta = softplus_pos(self.tau_raw[k]) * rho * scale      # (B,N,L)

        if not self.use_adaptive_shrink:
            # 消融：退化为经典软阈值（偏置恒为 θ）
            return torch.sign(Z) * F.relu(Z.abs() - theta)

        p = torch.sigmoid(self.p_raw[k])
        bias = theta * (1.0 + Z.abs() / theta.clamp_min(1e-12)).pow(-p)
        return torch.sign(Z) * F.relu(Z.abs() - bias)

    @staticmethod
    def _normalize_cols(A: torch.Tensor) -> torch.Tensor:
        return A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def support_debias(
        self, A: torch.Tensor, Xw: torch.Tensor, m: torch.Tensor, lam: float = 1e-6
    ) -> torch.Tensor:
        """按支撑集逐样本重解最小二乘（去偏操作）。

        给定支撑掩码 m ∈ {0,1}^{B×N×L}，对每个样本 l 求
            z_l = argmin_{supp(z)⊆supp(m_l)} ‖A z - x_l‖₂
        闭式解为  z_l = P Aᵀ (A P Aᵀ)⁻¹ x_l，其中 P = diag(m_l)。
        A P Aᵀ 仅为 M×M（M=2 时是 2×2），可批量求逆，故开销极小。

        该层消除了 ℓ1/近端迭代固有的幅值收缩偏差——这是本方法相对经典
        两步法的关键改进之一，且完全可微（掩码部分不参与梯度）。
        """
        B, M, N = A.shape
        L = Xw.shape[2]
        eye = torch.eye(M, device=A.device, dtype=A.dtype)
        G = torch.einsum("bin,bnl,bjn->blij", A, m, A)          # (B,L,M,M)
        # 相对岭项，保证 (A P Aᵀ) 始终可逆
        dmean = G.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True) \
                 .clamp_min(0.0).unsqueeze(-1)                  # (B,L,1,1)
        G = G + (lam + 1e-5 * dmean + 1e-10) * eye              # 与 (B,L,M,M) 广播
        Ginv = torch.linalg.inv(G)
        tmp = torch.einsum("blij,bjl->bli", Ginv, Xw)            # (B,L,M) = G⁻¹x
        z = m * torch.einsum("bin,bli->bnl", A, tmp)             # (B,N,L)
        return z

    # ------------------------------------------------------------------
    # 盲初始化：方向密度的峰值（势函数法的可微实现）
    # ------------------------------------------------------------------

    @staticmethod
    def _top_direction(Gw: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        """加权散点矩阵的主方向，带岭项与退化保护。

        直接用 `eigh` 在退化情形（簇为空 → 零矩阵，或近似重根）会抛出
        "algorithm failed to converge"。故：
          * 加岭项 1e-6·I 改善条件数；
          * 用 SVD 取左奇异向量（比 eigh 更稳健）；
          * 结果非有限或簇为空时回退到原方向。
        """
        B, M, _ = Gw.shape
        eye = torch.eye(M, device=Gw.device, dtype=Gw.dtype)
        G = Gw + 1e-6 * eye + 1e-12
        try:
            v = torch.linalg.svd(G, full_matrices=False)[0][:, :, 0]     # (B,M)
        except Exception:
            return fallback
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        bad = (~torch.isfinite(v).all(dim=1, keepdim=True))
        return torch.where(bad, fallback, v)

    def blind_init_A(
        self, Xw: torch.Tensor, n_clusters: int | None = None,
        use_ssp: bool = True, use_en: bool = True, use_bal: bool = True,
    ) -> torch.Tensor:
        """从观测的散点方向结构盲估计初始混合矩阵。

        n_clusters : 指定输出列数（= 源数目 N）；为 None 时输出 N_max 列。
        给定 N 时贪心只取 N 个主方向、K-means 聚 N 类，用于「已知源数目」的公平对比。

        原理：稀疏源下，观测向量的方向会在真实混合方向附近聚集。故在单位球面上
        做核密度峰值搜索（势函数法），并逐列软去相关。

        关键设计：用**软单源权重**加权。由 A 为实矩阵可知，单源点上
        Re(X) = a_n·Re(s)、Im(X) = a_n·Im(s)，二者严格共线，故
        |cos(Re(X), Im(X))| → 1。以该量的幂作为权重，等效于经典 SSP 筛选，
        但完全可微。注意该准则在可学习变换域内同样成立（Ψ 与 A 可交换）。

        返回 (B, M, Nmax)，列为单位向量。
        """
        B, M, L = Xw.shape
        K = n_clusters if n_clusters is not None else self.Nmax
        FT = L // 2
        Xr, Xi = Xw[:, :, :FT], Xw[:, :, FT:]                  # (B,M,FT)

        nr = Xr.norm(dim=1)                                    # (B,FT)
        ni = Xi.norm(dim=1)
        cos = (Xr * Xi).sum(dim=1) / (nr * ni).clamp_min(1e-12)
        q = softplus_pos(self.ssp_pow_raw)                     # 备用锐化指数

        # 软单源权重（三重门限）：
        #   ① 单源判据软阈值：经典方法用硬阈值 |cos(Re,Im)| > 0.98。幂次形式
        #      (|cos|^q) 在中等密度下不够锐利，多源点会渗入并污染方向统计。
        #      改用可学习的 sigmoid 软阈值，逼近硬判据但保持可微。
        #   ② 能量门限：稀疏数据中大量时频点是纯噪声，方向随机，必须剔除。
        #   ③ 实/虚部均衡门限：Re(s) 很小时 Re(X)≈噪声、方向不可靠。
        ssp_gate = torch.sigmoid((cos.abs() - self.cos_thr) / self.cos_soft) if use_ssp else torch.ones_like(cos.abs())

        e_tf = nr.pow(2) + ni.pow(2)                           # (B,FT)
        eref = e_tf.median(dim=1, keepdim=True).values.clamp_min(1e-12)
        egate = torch.sigmoid((e_tf / eref - self.egate_tau) / self.egate_sigma) if use_en else torch.ones_like(e_tf)

        mm = torch.minimum(nr, ni)                             # (B,FT)
        mref = mm.median(dim=1, keepdim=True).values.clamp_min(1e-12)
        pgate = mm / (mm + self.rel_gate * mref) if use_bal else torch.ones_like(mm)

        w = ssp_gate * egate * pgate                           # (B,FT) 软单源权重

        # 方向取自 Re(X)（单源点上 ∝ ±a_n）
        U = Xr / nr.unsqueeze(1).clamp_min(1e-8)               # (B,M,FT)
        k = U.abs().argmax(dim=1, keepdim=True)                # 符号规范化
        sgn = torch.gather(U, 1, k).sign()
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        U = U * sgn

        sigma_def = softplus_pos(self.sigma_def_raw)           # 去相关带宽（窄）
        cols = []
        for _ in range(K):
            Gw = torch.einsum("bl,bml,bnl->bmn", w, U, U)      # (B,M,M) 加权散点矩阵
            v = self._top_direction(Gw, cols[-1] if cols else U[:, :, 0])
            cols.append(v)
            # 抑制与 v 共线的散点；带宽须明显小于源间角距，否则相邻源会被一并压掉
            d2 = (U - v.unsqueeze(2)).pow(2).sum(dim=1)        # (B,FT)
            w = w * (1.0 - torch.exp(-d2 / (2 * sigma_def ** 2)))

        C = torch.stack(cols, dim=2)                           # (B,M,Nmax) 初始中心

        # ---- 加权球面 K-means 精修 ----
        # 纯贪心去相关会因"先选到的列把相邻源一并吸收"而漏源；经典两步法正是靠
        # 全体单源点参与的聚类避免该问题。这里做同样的补救：以上述峰值作初始化，
        # 再用全部（加权）散点做若干轮 K-means 迭代。
        # K-means 复原权重用较软的幂次门限 |cos|^q（而非锐利的 ssp_gate）：
        # 贪心阶段需锐利门限精确定位峰值；K-means 阶段需软门限让足够多的点参与精修，
        # 否则密集/低 SNR 下 w0 几乎处处为零、聚类退化为随机。此为有意的设计差异。
        w0 = (cos.abs().clamp(0.0, 1.0).pow(q) if use_ssp else torch.ones_like(cos.abs())) * egate * pgate
        for _ in range(self.n_init_iter):
            sim = torch.einsum("bmn,bml->bnl", C, U)           # (B,Nmax,FT)
            assign = (1.0 - sim.abs()).argmin(dim=1)           # (B,FT) 最近中心
            new_cols = []
            for n in range(K):
                msk = (assign == n).to(U.dtype) * w0           # (B,FT) 加权归属
                Gw = torch.einsum("bl,bml,bnl->bmn", msk, U, U)
                v = self._top_direction(Gw, C[:, :, n])
                # 若该簇为空（权重全零），保留原中心
                empty = (msk.sum(dim=1) < 1e-8).unsqueeze(1)
                new_cols.append(torch.where(empty, C[:, :, n], v))
            C = torch.stack(new_cols, dim=2)

        return self._normalize_cols(C)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        A_init: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> dict:
        """前向展开。

        Parameters
        ----------
        X : (B, M, L)   实数化观测
        A_init : (B, M, Nmax) 或 None
            初始混合矩阵。为 None 时使用 `blind_init_A` 的盲估计（推荐）。
            传入随机初始化可用于多次重启。

        Returns
        -------
        dict: A_hat (B,M,Nmax), S_hat (B,Nmax,L)，可选逐层轨迹
        """
        Xw = self._apply_psi(X)                   # 变换域观测
        B, M, L = X.shape
        sqrtL = math.sqrt(L)
        eyeN = torch.eye(self.Nmax, device=X.device, dtype=X.dtype)

        # ---- 初始化 ----
        A_blind = self.blind_init_A(Xw)
        if A_init is None:
            A = A_blind
        else:
            A = self._normalize_cols(A_init)
        S = self._prox(A.transpose(1, 2) @ Xw, 0)              # 由 LS 码收缩起步

        A_seq, S_seq = [], []
        for k in range(self.K):
            eta = softplus_pos(self.eta_raw[k])
            mu = torch.sigmoid(self.mu_raw[k])
            lam = softplus_pos(self.lam_raw[k])

            # --- 源更新：内层多步近端梯度（步长按 Lipschitz 常数归一化）---
            for _ in range(self.n_inner):
                G = A.transpose(1, 2) @ A
                L_A = torch.linalg.eigvalsh(G).amax(dim=-1).clamp_min(1e-6).view(B, 1, 1)
                residual = A @ S - Xw
                Z = S - (eta / L_A) * (A.transpose(1, 2) @ residual)
                S = self._prox(Z, k)

            # --- 行组收缩：整行归零 ≡ 该源不存在 → 源数目自适应 ---
            if self.use_group_shrink:
                nu = softplus_pos(self.nu_raw[k])
                g = S.norm(dim=2, keepdim=True) / sqrtL        # 行 RMS (B,N,1)
                # 相对阈值：以最大行 RMS 为基准，自动适配整体幅度尺度
                g_rel = g / g.amax(dim=1, keepdim=True).clamp_min(1e-12)
                S = S * F.relu(1.0 - nu / g_rel.clamp_min(1e-12))

            # --- 混合矩阵更新：最小二乘字典更新 + 梯度步的凸组合 ---
            # A_ls = X Sᵀ (S Sᵀ + λI)^{-1}，比单纯梯度步收敛快得多
            SS = S @ S.transpose(1, 2)                          # (B,N,N)
            XSt = Xw @ S.transpose(1, 2)                        # (B,M,N)
            # 相对岭项：λ_k 可能被训练压到过小而使 SS 奇异，故叠加一个与尺度
            # 成正比的绝对下限，保证 solve 始终可行
            diag_mean = SS.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)   # (B,1)
            reg = (lam + 1e-6 * diag_mean.clamp_min(0.0) + 1e-8).unsqueeze(-1)     # (B,1,1)
            A_ls = torch.linalg.solve(SS + reg * eyeN, XSt.transpose(1, 2)).transpose(1, 2)
            nrm_ls = A_ls.norm(dim=1, keepdim=True)
            valid = (nrm_ls > 1e-6).to(A.dtype)                 # 该源已被收缩掉则保留原列
            A_ls_u = A_ls / nrm_ls.clamp_min(1e-8)
            A = (1.0 - mu) * A + mu * (valid * A_ls_u + (1.0 - valid) * A)
            A = self._normalize_cols(A)

            if return_trace:
                A_seq.append(A)
                S_seq.append(S)

        # ---- 末端去偏：按网络给出的支撑集重解最小二乘，消除幅值收缩偏差 ----
        if self.use_debias:
            with torch.no_grad():
                athr = self.debias_rel * S.abs().amax(dim=2, keepdim=True)
                m = (S.abs() > athr).to(S.dtype)
            S = self.support_debias(A, Xw, m)

        out = {"A_hat": A, "S_hat": S, "A_blind": A_blind}
        if return_trace:
            out["A_seq"] = A_seq
            out["S_seq"] = S_seq
        return out

    # ------------------------------------------------------------------
    # 源数目判定与剪枝
    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_source_number(self, S: torch.Tensor, rel_thr: float = 0.35) -> torch.Tensor:
        """按行能量相对阈值（相对最大行能量）判定源数目。S : (B, Nmax, L) → (B,)

        注：曾尝试「最大相对间隙（肘部）」准则，但在极稀疏条件下不稳定
        （会把真实源一并截掉，实测夹角误差恶化到 49°）。相对阈值虽然偏保守
        （略高估源数目），但性质稳定，且高估的代价远小于漏源。
        """
        g = S.norm(dim=2)                                      # (B,Nmax)
        thr = rel_thr * g.max(dim=1, keepdim=True).values
        return (g > thr).sum(dim=1).clamp_min(1)

    @torch.no_grad()
    def prune(self, A: torch.Tensor, S: torch.Tensor, n_keep: int | None = None, rel_thr: float = 0.05):
        """剪掉不活跃的源。

        返回 (A_sel, S_sel, n_hat)，为逐样本列表（各样本源数目可不同）。
        """
        B = S.shape[0]
        if n_keep is None:
            n_hat = self.estimate_source_number(S, rel_thr)
        else:
            n_hat = torch.full((B,), int(n_keep), device=S.device)
        A_list, S_list, n_list = [], [], []
        for b in range(B):
            g = S[b].norm(dim=1)
            order = torch.argsort(g, descending=True)
            k = int(n_hat[b].item())
            k = max(k, 1)
            idx = order[:k]
            A_list.append(A[b][:, idx])
            S_list.append(S[b][idx])
            n_list.append(k)
        return A_list, S_list, n_list


# ======================================================================
# 自监督损失
# ======================================================================

def directional_coverage_loss(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """方向覆盖损失（自监督）。

    动机：稀疏源下观测向量在真实混合方向附近高度聚集。若把每个观测方向的
    "被解释程度"定义为它对 A 各列的最大 |cos|，则好的混合矩阵应能**覆盖**这些
    聚集方向：

        cover_l = max_n |cos(a_n, u_l)|
        L_dir   = -Σ_l w_l·cover_l / Σ_l w_l

    取 max（而非 sum）可避免所有列塌缩到同一方向：每个观测方向只需被某一列
    解释即可，要覆盖全部方向就必须让各列分散开。

    该项起到"锚定"作用——防止联合优化过程中混合矩阵偏离观测的方向结构
    （诊断显示，缺少该项时展开精修会把已很好的初始化 A 显著恶化）。
    权重 w_l 复用单源判据（|cos(Re,Im)| 的幂 × 能量门限）。
    """
    B, M, L = X.shape
    FT = L // 2
    Xr, Xi = X[:, :, :FT], X[:, :, FT:]
    nr = Xr.norm(dim=1)                                        # (B,FT)
    ni = Xi.norm(dim=1)
    e_tf = nr.pow(2) + ni.pow(2)
    eref = e_tf.median(dim=1, keepdim=True).values.clamp_min(1e-12)
    egate = torch.sigmoid((e_tf / eref - 10.0) / 3.0)
    cos_ri = (Xr * Xi).sum(dim=1) / (nr * ni).clamp_min(1e-12)
    w = cos_ri.abs().clamp(0.0, 1.0).pow(4.0) * egate          # (B,FT)

    U = Xr / nr.unsqueeze(1).clamp_min(1e-8)                   # (B,M,FT)
    sim = torch.einsum("bmn,bml->bnl", A, U)                   # (B,Nmax,FT)
    cover = sim.abs().amax(dim=1)                              # (B,FT)
    return -((w * cover).sum(dim=1) / (w.sum(dim=1) + 1e-8)).mean()


def unsupervised_loss(
    X: torch.Tensor,
    out: dict,
    model: SADUN,
    beta_sp: float = 0.05,
    gamma_ind: float = 0.5,
    delta_orth: float = 0.1,
    delta_dir: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    """自监督目标：仅依赖观测 X，不需要干净源标签。

    L = ‖W - ÂŜ‖²/‖W‖²           观测一致性
      + β·mean(|Ŝ|)                稀疏性（防止阈值塌缩为 0）
      + γ·源间相关惩罚             独立性（BSS 核心准则的稀疏域替代）
      + δ·Ψ 正交约束               保证变换可逆
      + ζ·方向覆盖损失             把混合矩阵锚定到观测的方向结构
    """
    A, S = out["A_hat"], out["S_hat"]

    rec = ((A @ S - X) ** 2).mean() / (X ** 2).mean().clamp_min(1e-12)

    sp = S.abs().mean()

    # 源间相关：稀疏域下不同源应几乎不重叠
    Sc = S - S.mean(dim=2, keepdim=True)
    Sc = Sc / Sc.norm(dim=2, keepdim=True).clamp_min(1e-8)
    G = torch.einsum("bnl,bml->bnm", Sc, Sc)
    Nm = S.shape[1]
    off = (G ** 2).sum(dim=(1, 2)) - (G.diagonal(dim1=1, dim2=2) ** 2).sum(dim=1)
    ind = (off / max(Nm * (Nm - 1), 1)).mean()

    orth = model.transform_penalty()
    dir_loss = directional_coverage_loss(X, A)

    loss = rec + beta_sp * sp + gamma_ind * ind + delta_orth * orth + delta_dir * dir_loss
    parts = {
        "rec": rec.detach(),
        "sp": sp.detach(),
        "ind": ind.detach(),
        "orth": orth.detach(),
        "dir": dir_loss.detach(),
        "total": loss.detach(),
    }
    return loss, parts


# ======================================================================
# 便捷构造
# ======================================================================

def build_model(cfg: dict) -> SADUN:
    return SADUN(
        m_obs=cfg["m_obs"],
        n_max=cfg["n_max"],
        n_layers=cfg["n_layers"],
        n_freq=cfg["n_freq"],
        n_inner=cfg.get("n_inner", 3),
        use_transform=cfg.get("use_transform", True),
        use_adaptive_shrink=cfg.get("use_adaptive_shrink", True),
        use_group_shrink=cfg.get("use_group_shrink", True),
    )
