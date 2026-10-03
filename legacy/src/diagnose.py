"""诊断：验证未训练的前向展开是否已是有效求解器，并定位剩余问题。

用法（项目 src 目录下）:
    ../.venv/bin/python diagnose.py [checkpoint路径]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

import data as D
import metrics as MT
from baselines import kmeans_sphere, l1_recover, run_method, ssp_directions, ssp_mask_from_complex
from config import get_cfg
from infer import run_model
from model import build_model, unsupervised_loss


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg()
    M, Nmax = cfg["m_obs"], cfg["n_max"]
    model = build_model(cfg).to(device)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd["state"])
        print(f"[已加载 {ckpt}] epoch={sd.get('epoch')} val_loss={sd.get('val_loss'):.4f}")
    model.eval()

    rng = np.random.default_rng(7)

    # ------------------------------------------------------------------
    hr("1. 未训练/当前模型的前向是否有效（各稀疏度，N=4，SNR=30）")
    print(f"  {'cfg':<10s} {'gini':>6s} │ {'SA-DUN 夹角':>12s} {'SA-DUN SDR':>11s} {'N̂':>3s} "
          f"│ {'SCA-L1 夹角':>12s} {'SCA-L1 SDR':>11s}")
    for key in ["tf_p02", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]:
        a_m, s_m, a_b, s_b, nh, g = [], [], [], [], [], []
        for seed in range(3):
            p = D.make_problem(M, 4, cfg["n_freq"], cfg["n_frames"], key, 30.0, seed=200 + seed)
            g.append(p["gini"])
            r = run_model(model, p["X_all"], device, rel_thr=0.05)
            a_m.append(MT.mixing_matrix_angle_error_deg(p["A"], r["A_hat"]))
            s_m.append(MT.evaluate_sources(p["S_all"], r["S_hat"])["SDR"])
            nh.append(r["n_hat"])
            rb = run_method("ssp_kmeans_l1", p, seed=seed)
            a_b.append(MT.mixing_matrix_angle_error_deg(p["A"], rb["A_hat"]))
            s_b.append(MT.evaluate_sources(p["S_all"], rb["S_hat"])["SDR"])
        print(f"  {key:<10s} {np.mean(g):>6.3f} │ {np.mean(a_m):>12.2f} {np.mean(s_m):>11.2f} "
              f"{np.mean(nh):>3.1f} │ {np.mean(a_b):>12.2f} {np.mean(s_b):>11.2f}")

    # ------------------------------------------------------------------
    hr("2. 盲初始化的质量 vs 随机初始化（未训练通路）")
    Xt = None
    for key in ["tf_p10", "tf_gauss"]:
        p = D.make_problem(M, 4, cfg["n_freq"], cfg["n_frames"], key, 30.0, seed=11)
        Xn = p["X_all"] / (np.sqrt(np.mean(p["X_all"] ** 2)) + 1e-12)
        with torch.no_grad():
            Xt = torch.as_tensor(Xn[None], dtype=torch.float32, device=device)
            Ab = model.blind_init_A(model._apply_psi(Xt))[0].cpu().numpy()
        A_rand = D.gen_mixing_matrix(M, Nmax, rng)
        print(f"  [{key}] 盲初始化夹角={MT.mixing_matrix_angle_error_deg(p['A'], Ab):>6.2f}°  "
              f"随机初始化夹角={MT.mixing_matrix_angle_error_deg(p['A'], A_rand):>6.2f}°")
        U = ssp_directions(p["X_tf"], ssp_mask_from_complex(p["X_tf"]))
        A_ssp = kmeans_sphere(U, 4, rng)
        print(f"        经典 SSP+K-means 夹角={MT.mixing_matrix_angle_error_deg(p['A'], A_ssp):>6.2f}°")

    # ------------------------------------------------------------------
    hr("3. 展开过程中 A 夹角误差的变化（看 A 更新是否真的在收敛）")
    p = D.make_problem(M, 4, cfg["n_freq"], cfg["n_frames"], "tf_p10", 30.0, seed=11)
    Xn = p["X_all"] / (np.sqrt(np.mean(p["X_all"] ** 2)) + 1e-12)
    with torch.no_grad():
        Xt = torch.as_tensor(Xn[None], dtype=torch.float32, device=device)
        out = model(Xt, None, return_trace=True)
    print(f"  {'init(blind)':>12s} {MT.mixing_matrix_angle_error_deg(p['A'], out['A_blind'][0].cpu().numpy()):>8.2f}°")
    for k, A in enumerate(out["A_seq"]):
        print(f"  {'layer ' + str(k + 1):>12s} {MT.mixing_matrix_angle_error_deg(p['A'], A[0].cpu().numpy()):>8.2f}°")

    # ------------------------------------------------------------------
    hr("4. 无监督损失的区分度（真值解 vs 退化解）")
    L = Xn.shape[1]
    n_true = p["n"]
    lk = {"beta_sp": cfg["beta_sp"], "gamma_ind": cfg["gamma_ind"], "delta_orth": cfg["delta_orth"]}

    def pad(A):
        o = np.zeros((M, Nmax), np.float32); o[:, :A.shape[1]] = A; return o

    def padS(Ss):
        o = np.zeros((Nmax, L), np.float32); o[:Ss.shape[0]] = Ss; return o

    scale = np.sqrt(np.mean(p["X_all"] ** 2))
    S_true_n = padS(p["S_all"] / scale)
    S_ls_true = padS(np.linalg.lstsq(p["A"], Xn, rcond=None)[0])
    A_rand = D.gen_mixing_matrix(M, Nmax, rng)
    S_ls_rand = np.linalg.lstsq(A_rand, Xn, rcond=None)[0]
    S_l1_rand = l1_recover(A_rand, Xn)

    with torch.no_grad():
        print(f"  {'场景':<30s} {'总损失':>10s} {'rec':>9s} {'sp':>9s} {'ind':>9s}")
        for nm, AA, SS in [
            ("真值A + 真值S", pad(p["A"]), S_true_n),
            ("真值A + LS稠密解", pad(p["A"]), S_ls_true),
            ("随机A + LS稠密解", A_rand, S_ls_rand),
            ("随机A + ℓ1稀疏码", A_rand, S_l1_rand),
        ]:
            o = {"A_hat": torch.as_tensor(AA[None], device=device),
                 "S_hat": torch.as_tensor(SS[None], device=device)}
            l, parts = unsupervised_loss(Xt, o, model, **lk)
            print(f"  {nm:<30s} {float(l):>10.4f} {float(parts['rec']):>9.4f} "
                  f"{float(parts['sp']):>9.4f} {float(parts['ind']):>9.4f}")

    # ------------------------------------------------------------------
    hr("5. 学到的参数")
    print(f"  eta = {np.round(Fn.softplus(model.eta_raw).detach().cpu().numpy(), 3)}")
    print(f"  mu  = {np.round(torch.sigmoid(model.mu_raw).detach().cpu().numpy(), 3)}")
    print(f"  tau = {np.round(Fn.softplus(model.tau_raw).detach().cpu().numpy(), 4)}")
    print(f"  nu  = {np.round(Fn.softplus(model.nu_raw).detach().cpu().numpy(), 4)}")
    print(f"  p   = {np.round(torch.sigmoid(model.p_raw).detach().cpu().numpy(), 3)}")
    print(f"  rho = {np.round(Fn.softplus(model.rho_raw).detach().cpu().numpy(), 3)}")
    print(f"  sigma_init = {float(Fn.softplus(model.sigma_init_raw)):.4f}")
    if model.use_transform:
        psi = model.psi.detach().cpu().numpy()
        print(f"  Ψ 与单位阵最大偏差 = {np.abs(psi - np.eye(cfg['n_freq'])).max():.4f}")


if __name__ == "__main__":
    main()
