"""冒烟测试：在服务器上快速验证各模块可跑通、量级合理。

用法（在项目 src 目录下）:
    ../.venv/bin/python smoke_test.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

import data as D
import metrics as MT
from baselines import BASELINE_LABELS, BASELINE_NAMES, run_method
from config import get_cfg
from infer import random_inits, run_model
from model import build_model, unsupervised_loss


def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def test_data():
    hr("1. 数据模块")
    for key in ["tf_p02", "tf_p10", "tf_p40", "tf_gauss"]:
        p = D.make_problem(m_obs=2, n_sources=4, n_freq=33, n_frames=64,
                           cfg_key=key, snr_db=30.0, seed=0)
        print(f"  {key:9s} gini={p['gini']:.3f} overlap={p['overlap']:.2f} "
              f"wdo_emp={p['wdo_violation_emp']:.3f} X_all={p['X_all'].shape}")

    # 实值化等价性
    p = D.make_problem(2, 4, 33, 64, "tf_p10", 30.0, seed=0)
    Xtf = np.einsum("mn,nft->mft", p["A"], p["S_tf"])
    err = np.max(np.abs(D.tf_to_real(Xtf) - p["A"] @ p["S_all"]))
    assert err < 1e-10, f"实值化不等价: {err}"
    print(f"  实值化等价性误差 = {err:.2e}  OK")

    # 往返变换
    Z = p["S_tf"]
    back = D.real_to_tf(D.tf_to_real(Z), Z.shape[1], Z.shape[2])
    print(f"  TF↔实数往返误差 = {np.max(np.abs(Z - back)):.2e}  OK")


def test_metrics():
    hr("2. 评测指标")
    p = D.make_problem(2, 4, 33, 64, "tf_p10", 30.0, seed=1)
    # 用真值本身评测，应得到很高的 SDR
    m = MT.evaluate_sources(p["S_all"], p["S_all"])
    print(f"  真值自评 SDR={m['SDR']:.2f} dB  SIR={m['SIR']:.2f} dB  (应显著为正)")
    assert m["SDR"] > 60, "指标异常"

    ang = MT.mixing_matrix_angle_error_deg(p["A"], p["A"])
    print(f"  真值自评混合矩阵夹角误差 = {ang:.4f}°  (应 ≈0)")
    assert ang < 1e-6, "夹角指标异常"

    # 随机矩阵应得到较大的角度误差
    rng = np.random.default_rng(0)
    Ar = D.gen_mixing_matrix(2, 4, rng)
    print(f"  随机矩阵夹角误差 = {MT.mixing_matrix_angle_error_deg(p['A'], Ar):.2f}°")
    print(f"  源数目指标 = {MT.source_number_metrics(4, 5)}")


def test_baselines(key="tf_p05", snr=30.0, n=4):
    hr(f"3. 基线方法（{key}, SNR={snr}, N={n}, M=2）")
    p = D.make_problem(2, n, 33, 64, key, snr, seed=2)
    print(f"  真值参考: gini={p['gini']:.3f} overlap={p['overlap']:.2f}")
    rows = []
    for name in BASELINE_NAMES:
        try:
            t0 = time.perf_counter()
            r = run_method(name, p, seed=0)
            dt = time.perf_counter() - t0
            m = MT.evaluate_sources(p["S_all"], r["S_hat"])
            ang = MT.mixing_matrix_angle_error_deg(p["A"], r["A_hat"])
            rows.append((name, r["n_hat"], m["SDR"], m["SIR"], ang, dt, r["ssp_count"]))
        except Exception as e:
            rows.append((name, -1, float("nan"), float("nan"), float("nan"), 0.0, 0))
            print(f"  !! {name} 失败: {type(e).__name__}: {e}")
    print(f"  {'方法':<28s} {'N̂':>3s} {'SDR':>8s} {'SIR':>8s} {'A_err°':>8s} {'秒':>7s} {'SSP':>6s}")
    for name, nh, sdr, sir, ang, dt, ssp in rows:
        print(f"  {BASELINE_LABELS[name]:<28s} {nh:>3d} {sdr:>8.2f} {sir:>8.2f} "
              f"{ang:>8.2f} {dt:>7.3f} {ssp:>6d}")


def test_recovery():
    """ℓ1 恢复的合法性检验：用真值 A 做恢复，SDR 应显著为正。"""
    hr("4. ℓ1 恢复合法性检验（Oracle-A，λ 扫描）")
    from baselines import l1_recover, omp_recover

    for key in ["tf_p05", "tf_p20"]:
        p = D.make_problem(2, 4, 33, 64, key, 30.0, seed=3)
        print(f"\n  [{key}] gini={p['gini']:.3f} overlap={p['overlap']:.2f}")
        print(f"    {'lam_ratio':>10s} {'no-debias':>10s} {'debias':>10s}")
        for lr_ in [0.001, 0.005, 0.02, 0.05, 0.1, 0.2]:
            s1 = l1_recover(p["A"], p["X_all"], lam_ratio=lr_, debias=False)
            s2 = l1_recover(p["A"], p["X_all"], lam_ratio=lr_, debias=True)
            d1 = MT.evaluate_sources(p["S_all"], s1)["SDR"]
            d2 = MT.evaluate_sources(p["S_all"], s2)["SDR"]
            print(f"    {lr_:>10.3f} {d1:>10.2f} {d2:>10.2f}")
        so = omp_recover(p["A"], p["X_all"])
        print(f"    OMP(max_active=M)            SDR = {MT.evaluate_sources(p['S_all'], so)['SDR']:.2f} dB")


def test_model():
    hr("4. 模型前向 / 反向")
    cfg = get_cfg(epochs=1, steps_per_epoch=2, n_layers=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}")
    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数量 = {n_par:,}")

    from sampler import sample_problems
    rng = np.random.default_rng(0)
    b = sample_problems(cfg, 8, rng, device)
    print(f"  批次: X={tuple(b['X'].shape)} A_init={tuple(b['A_init'].shape)}")

    t0 = time.perf_counter()
    out = model(b["X"], b["A_init"], return_trace=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  前向耗时 = {time.perf_counter() - t0:.4f}s  层数={len(out['A_seq'])}")
    print(f"  输出: A_hat={tuple(out['A_hat'].shape)} S_hat={tuple(out['S_hat'].shape)}")

    loss, parts = unsupervised_loss(b["X"], out, model)
    print(f"  损失 = {float(loss):.4f}  明细 = "
          + "  ".join(f"{k}={float(v):.4f}" for k, v in parts.items() if k != 'total'))
    loss.backward()
    gnorm = sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  梯度范数 = {gnorm:.4f}  (应 > 0)")
    assert gnorm > 0, "梯度为 0，反向传播有问题"

    # 梯度数值检验：对 tau_raw 做有限差分
    hr("5. 梯度数值检验（有限差分）")
    model.zero_grad()
    p0 = model.tau_raw.detach().clone()
    eps = 1e-4

    def loss_at(perturb):
        with torch.no_grad():
            model.tau_raw.copy_(p0 + perturb)
            o = model(b["X"][:2], b["A_init"][:2])
            l, _ = unsupervised_loss(b["X"][:2], o, model)
            return float(l)

    model.zero_grad()
    with torch.no_grad():
        model.tau_raw.copy_(p0)
    o = model(b["X"][:2], b["A_init"][:2])
    l, _ = unsupervised_loss(b["X"][:2], o, model)
    l.backward()
    ana = model.tau_raw.grad.detach().clone()

    idxs = sorted(set([0, min(3, p0.numel() - 1), p0.numel() - 1]))
    print(f"  {'idx':>4s} {'解析梯度':>14s} {'数值梯度':>14s} {'相对误差':>12s}")
    ok = True
    for i in idxs:
        e = torch.zeros_like(p0)
        e[i] = eps
        num = (loss_at(e) - loss_at(-e)) / (2 * eps)
        a = float(ana[i])
        rel = abs(a - num) / max(abs(a) + abs(num), 1e-8)
        print(f"  {i:>4d} {a:>14.6e} {num:>14.6e} {rel:>12.2%}")
        if rel > 0.05:
            ok = False
    print(f"  梯度检验: {'通过' if ok else '未通过（需排查）'}")
    return ok


def test_infer():
    hr("6. 推理接口")
    cfg = get_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)          # 未训练，仅验证接口
    p = D.make_problem(2, 4, 33, 64, "tf_p10", 30.0, seed=5)
    rng = np.random.default_rng(0)
    inits = random_inits(cfg["m_obs"], cfg["n_max"], 3, rng)
    t0 = time.perf_counter()
    r = run_model(model, p["X_all"], inits, device)
    print(f"  用时 {time.perf_counter()-t0:.3f}s  N̂={r['n_hat']}  "
          f"A_hat={r['A_hat'].shape} S_hat={r['S_hat'].shape}  loss={r['best_loss']:.4f}")
    m = MT.evaluate_sources(p["S_all"], r["S_hat"])
    print(f"  (未训练) SDR={m['SDR']:.2f} dB")


if __name__ == "__main__":
    test_data()
    test_metrics()
    test_baselines()
    test_recovery()
    ok = test_model()
    test_infer()
    print("\n" + "=" * 70)
    print("冒烟测试结束")
    print("=" * 70)
    sys.exit(0 if ok else 1)
