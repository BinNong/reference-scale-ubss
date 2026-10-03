"""在合成数据上验证 Section 5 的闭式预测（需要 numpy / torch）。

与主实验共用 data.py / model.py；本脚本只做验证，不改变任何主结果。

验证项：
    D1  单源 / 噪声点的实际占比  vs  二项式预测 pi_s, pi_0
    D2  噪声点上硬阈值的通过率    vs  Corollary 1 的 p_c
    D3  实际污染比 kappa          vs  Proposition 4
    D4  能量门限在噪声/信号点上的平均权重  vs  Corollary 2
    D5  Davis-Kahan 量级预测中的 O(1) 常数 c 是否稳定（是否真的 O(1)）

跑法（项目 src 目录）:
    ../.venv/bin/python theory_validate_data.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import data as D
import metrics as MT
from experiments_mdde import make_model
from theory_checks import p_abs_cos_exceed

P_OF_KEY = {"tf_p02": 0.02, "tf_p05": 0.05, "tf_p10": 0.10,
            "tf_p20": 0.20, "tf_p40": 0.40}
LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40"]


def gate_values(model, X_all, use_ssp=True, use_en=True, use_bal=True):
    """逐 TF 点计算三个门限。

    与 model.blind_init_A 中的定义逐字一致（同一组超参、同一归一化），
    区别只是这里把中间量取出来做统计。
    """
    Xn = D.normalize_obs(X_all)
    Xt = torch.as_tensor(Xn[None], dtype=torch.float32)
    with torch.no_grad():
        Xw = model._apply_psi(Xt)                       # (1, M, L)
        L = Xw.shape[2]
        FT = L // 2
        Xr, Xi = Xw[:, :, :FT], Xw[:, :, FT:]           # (1,M,FT)
        nr = Xr.norm(dim=1)                             # (1,FT)
        ni = Xi.norm(dim=1)
        cos = (Xr * Xi).sum(dim=1) / (nr * ni).clamp_min(1e-12)

        ssp = (torch.sigmoid((cos.abs() - model.cos_thr) / model.cos_soft)
               if use_ssp else torch.ones_like(cos))

        e = nr.pow(2) + ni.pow(2)
        eref = e.median(dim=1, keepdim=True).values.clamp_min(1e-12)
        en = (torch.sigmoid((e / eref - model.egate_tau) / model.egate_sigma)
              if use_en else torch.ones_like(e))

        mm = torch.minimum(nr, ni)
        mref = mm.median(dim=1, keepdim=True).values.clamp_min(1e-12)
        bal = (mm / (mm + model.rel_gate * mref)
               if use_bal else torch.ones_like(mm))

    f = lambda t: np.asarray(t[0], dtype=np.float64)
    return f(cos), f(ssp), f(en), f(bal), f(e)


def activity(prob, ft_expected: int) -> np.ndarray:
    """每个 TF 点的活跃源数（由真值 S_tf 得到）。"""
    S = np.abs(np.asarray(prob["S_tf"], dtype=np.float64))
    if S.ndim == 3:
        S = S.reshape(S.shape[0], -1)
    assert S.shape[1] == ft_expected, f"S_tf 的 FT={S.shape[1]} 与观测的 FT={ft_expected} 不一致"
    return (S > 1e-12).sum(axis=0)


def main() -> None:
    model = make_model().eval()
    print(f"门限超参: cos_thr={model.cos_thr} cos_soft={model.cos_soft} "
          f"egate_tau={model.egate_tau} egate_sigma={model.egate_sigma} "
          f"rel_gate={model.rel_gate}")
    print()

    hdr = (f"  {'cfg':<9s} {'p':>5s} | {'pi_s 实测/理论':>18s} {'pi_0 实测/理论':>18s} | "
           f"{'p_c 实测/理论':>17s} | {'kappa 实测/理论':>17s}")
    print("=" * len(hdr))
    print("[D1-D3] 占比、噪声通过率、污染比")
    print(hdr)
    print("-" * len(hdr))

    kappa_rows = []
    for key in LADDER:
        p = P_OF_KEY[key]
        seed_list = [1000, 1001, 1002]
        acc = dict(fs=[], f0=[], pc=[], ks=[], k0=[], gen_n=[], gen_s=[], gsp_n=[], gsp_s=[], gbal=[])
        for sd in seed_list:
            prob = D.make_problem(2, 4, 33, 64, key, 20.0, seed=sd)
            cos, ssp, en, bal, e = gate_values(model, prob["X_all"])
            FT = cos.size
            cnt = activity(prob, FT)
            m_sig, m_noise = (cnt == 1), (cnt == 0)

            acc["fs"].append(m_sig.mean())
            acc["f0"].append(m_noise.mean())
            if m_noise.sum() > 0:
                acc["pc"].append(float((np.abs(cos[m_noise]) > 0.98).mean()))
                acc["gen_n"].append(float(en[m_noise].mean()))
                acc["gsp_n"].append(float(ssp[m_noise].mean()))
                acc["gbal"].append(float(bal[m_noise].mean()))
            if m_sig.sum() > 0:
                acc["gen_s"].append(float(en[m_sig].mean()))
                acc["gsp_s"].append(float(ssp[m_sig].mean()))

            # 硬阈值（理想化：只看 |cos|>0.98）下的点数
            thr = np.abs(cos) > 0.98
            acc["k0"].append(float((thr & m_noise).sum()))
            acc["ks"].append(float((thr & m_sig).sum()))

        mean = lambda k: float(np.mean(acc[k])) if acc[k] else float("nan")
        pi_s_th = 4 * p * (1 - p) ** 3
        pi_0_th = (1 - p) ** 4
        pc_th = p_abs_cos_exceed(2, 0.98)
        kappa_th = (1 - p) * pc_th / (4 * p)
        ks, k0 = np.sum(acc["ks"]), np.sum(acc["k0"])
        kappa_meas = k0 / ks if ks > 0 else float("nan")

        print(f"  {key:<9s} {p:>5.2f} | {mean('fs'):>8.4f}/{pi_s_th:<8.4f} "
              f"{mean('f0'):>8.4f}/{pi_0_th:<8.4f} | {mean('pc'):>8.4f}/{pc_th:<8.4f} | "
              f"{kappa_meas:>8.3f}/{kappa_th:<8.3f}")

        kappa_rows.append(dict(key=key, p=p, kappa_th=kappa_th, kappa_meas=kappa_meas,
                               gen_n=mean("gen_n"), gsp_n=mean("gsp_n"),
                               gen_s=mean("gen_s"), gsp_s=mean("gsp_s"),
                               gbal=mean("gbal"), ks=ks, k0=k0, pc_meas=mean("pc")))

    print()
    print("=" * 78)
    print("[D4] 门限在噪声点 / 信号点上的平均取值")
    print("=" * 78)
    print(f"  {'cfg':<9s} {'g_en(noise)':>12s} {'g_en(signal)':>13s} "
          f"{'g_ssp(noise)':>13s} {'g_ssp(signal)':>14s} {'g_bal(noise)':>13s}")
    for r in kappa_rows:
        print(f"  {r['key']:<9s} {r['gen_n']:>12.5f} {r['gen_s']:>13.5f} "
              f"{r['gsp_n']:>13.5f} {r['gsp_s']:>14.5f} {r['gbal']:>13.5f}")
    print()
    print("  理论: g_en(noise) -> sigma((1 - tau_e)/sigma_e) = "
          f"{1 / (1 + np.exp(3.0)):.5f}   (tau_e=10, sigma_e=3)")

    print()
    print("=" * 78)
    print("[D5] Davis-Kahan 量级预测中的 O(1) 常数")
    print("     sin(angle) <= c * sqrt(K0) / Ks   =>   c ~ sin(angle)*Ks/sqrt(K0)")
    print("=" * 78)
    print(f"  {'cfg':<9s} {'Ks':>7s} {'K0':>7s} {'sqrt(K0)/Ks':>13s} "
          f"{'实测角度':>9s} {'实测sin':>9s} {'=> c':>8s} {'c(能量门限后)':>14s}")
    dk_rows = []
    for r in kappa_rows:
        ang, gen_n = [], r["gen_n"]
        for sd in [1000, 1001, 1002]:
            prob = D.make_problem(2, 4, 33, 64, r["key"], 20.0, seed=sd)
            with torch.no_grad():
                A = model.blind_init_A(model._apply_psi(
                    torch.as_tensor(D.normalize_obs(prob["X_all"])[None],
                                    dtype=torch.float32)), n_clusters=4)[0].numpy()
            ang.append(MT.mixing_matrix_angle_error_deg(prob["A"], A))
        a = float(np.mean(ang))
        sa = np.sin(np.radians(a))
        ratio = np.sqrt(r["k0"]) / r["ks"]
        c_eff = sa / ratio
        # 能量门限把噪声点有效数降到 K0 * g_en
        ratio_g = np.sqrt(r["k0"] * gen_n) / r["ks"]
        c_eff_g = sa / ratio_g
        print(f"  {r['key']:<9s} {r['ks']:>7.0f} {r['k0']:>7.0f} {ratio:>13.4f} "
              f"{a:>8.2f}° {sa:>9.4f} {c_eff:>8.2f} {c_eff_g:>14.2f}")
        dk_rows.append(dict(key=r["key"], p=r["p"], Ks=r["ks"], K0=r["k0"],
                            ratio=ratio, angle_deg=a, sin_angle=sa,
                            c_no_gate=c_eff, c_gated=c_eff_g))

    print()
    print("  预期：若推导正确，c 应跨稀疏度保持 O(1)，不随 p 系统性漂移；")
    print("        若 c 随 p 剧烈变化，说明闭式预测漏掉了主导机制。")

    return {
        "gates": {"cos_thr": model.cos_thr, "cos_soft": model.cos_soft,
                  "egate_tau": model.egate_tau, "egate_sigma": model.egate_sigma,
                  "rel_gate": model.rel_gate},
        "ladder": kappa_rows,
        "davis_kahan": dk_rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/theory_validation.json")
    a = ap.parse_args()
    payload = main()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n已写出 {a.out}")
