"""真实语音的时频统计量：设计实验 + 检验闭式理论的可迁移性。

先量后设
--------
合成实验里"稀疏度"由一个可调的 p 控制；真实语音没有这个旋钮，π₀/π₁ 由语音
自身的停顿结构决定。因此必须先量清楚：真实语音落在什么工况上？
若 π₀ 很小（稠密），则中位数根本不是噪声分位数，闭式的噪声区间不成立；
若 π₀ 较大，则理论可以照搬。这一步决定后续实验怎么设计，也决定论文的
真实数据一节能声称什么。

本脚本输出每个配置的：
  π₀, π₁, π₂₊        由真值源的活跃模式统计（相对门限 −30 dB）
  gini, overlap      稀疏度与同时活跃源数
  r_true             中位数 / 噪声底（噪声底由注入噪声的已知功率给出）
  ŝ²/σ²              盲估计的噪声功率与真值之比（估计器在真实数据上的准确度）
  p_fit, r_closed    由 π₀=(1−p)^N 反解等效 p，再代入闭式 mixture_median_ratio
  gate               自检是否启用能量门限

用法:
    ../.venv/bin/python real_stats.py --corpus ../data/ls_partial [--pretty]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import realdata as RD
from nfr import estimate_noise_power, mixture_median_ratio, nfr_mask


def fmt(x, w=8, p=3):
    return f"{'—':>{w}}" if (x is None or not np.isfinite(x)) else f"{x:>{w}.{p}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/ls_partial")
    ap.add_argument("--n-files", type=int, default=40)
    ap.add_argument("--dur", type=float, default=6.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="../results/real_stats.json")
    a = ap.parse_args()

    corpus = RD.scan_corpus(a.corpus)
    spk = sorted({c["speaker"] for c in corpus})
    print(f"语料：{len(corpus)} 个文件 / {len(spk)} 个说话人  {spk[:12]}")

    cfgs = []
    for win in [256, 512, 1024, 2048]:
        cfgs.append(dict(win=win, n_sources=4, snr_db=20.0))
    for N in [2, 3, 5, 6]:
        cfgs.append(dict(win=1024, n_sources=N, snr_db=20.0))
    for snr in [0.0, 10.0, 30.0, 40.0]:
        cfgs.append(dict(win=1024, n_sources=4, snr_db=snr))
    for nd in [1, 2]:
        cfgs.append(dict(win=1024, n_sources=4, snr_db=20.0, n_dense=nd))
    cfgs.append(dict(win=1024, n_sources=4, snr_db=20.0, noise="babble"))
    cfgs.append(dict(win=1024, n_sources=4, snr_db=None))

    print()
    hdr = (f"{'win':>5s}{'N':>3s}{'SNR':>5s}{'dense':>6s}{'noise':>7s} │"
           f"{'π₀':>7s}{'π₁':>7s}{'π₂₊':>7s}{'overlap':>8s}{'gini':>6s} │"
           f"{'ν_true':>10s}{'r_true':>8s} │{'ŝ²/σ²':>7s}{'p_fit':>7s}{'r_closed':>9s} │"
           f"{'gate':>5s}{'keep':>6s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    pool = None
    for cfg in cfgs:
        acc = []
        for s in range(a.seeds):
            need_spk = cfg["n_sources"] - cfg.get("n_dense", 0)
            pool = RD.load_pool(corpus, n_files=a.n_files, dur_s=a.dur, seed=s,
                                min_speakers=need_spk)
            p = RD.make_real_problem(pool, m_obs=2, dur_s=a.dur, seed=1000 + s, **cfg)
            est = estimate_noise_power(np.sum(np.abs(p["X_tf"]) ** 2, axis=0).ravel(), p["m"])
            mask, diag = nfr_mask(p["X_tf"], p["m"])
            need_spk = cfg["n_sources"] - cfg.get("n_dense", 0)
            p_fit = 1.0 - p["pi0"] ** (1.0 / cfg["n_sources"])
            r_closed = mixture_median_ratio(p_fit, cfg["n_sources"], p["m"],
                                            cfg["snr_db"] if cfg["snr_db"] else 40.0)["ratio"]
            acc.append(dict(
                win=cfg["win"], N=cfg["n_sources"], snr=cfg["snr_db"],
                n_dense=cfg.get("n_dense", 0), noise=cfg.get("noise", "gauss") or "none",
                pi0=p["pi0"], pi1=p["pi1"], pi2p=p["pi2p"], overlap=p["overlap"],
                gini=p["gini"], nu_true=p["nu_true"], r_true=p["r_true"],
                s2_ratio=est["s2"] / max(p["nu_true"] / p["m"], 1e-30),
                p_fit=p_fit, r_closed=r_closed,
                gate_on=bool(diag["gate_on"]), keep_frac=diag["keep_frac"],
                n_levels=est.get("n_levels", 0), equiv_err=p["equiv_err"],
                snr_true_db=p["snr_true_db"]))
        rows += acc
        m = {k: float(np.mean([r[k] for r in acc])) for k in
             ["pi0", "pi1", "pi2p", "overlap", "gini", "nu_true", "r_true",
              "s2_ratio", "p_fit", "r_closed", "keep_frac"]}
        go = sum(1 for r in acc if r["gate_on"])
        print(f"{cfg['win']:>5d}{cfg['n_sources']:>3d}"
              f"{(cfg['snr_db'] if cfg['snr_db'] is not None else -1):>5.0f}"
              f"{cfg.get('n_dense', 0):>6d}{cfg.get('noise', 'gauss') or 'none':>7s} │"
              f"{m['pi0']:>7.3f}{m['pi1']:>7.3f}{m['pi2p']:>7.3f}{m['overlap']:>8.2f}{m['gini']:>6.3f} │"
              f"{m['nu_true']:>10.3e}{m['r_true']:>8.3f} │"
              f"{m['s2_ratio']:>7.3f}{m['p_fit']:>7.3f}{fmt(m['r_closed'], 9)} │"
              f"{go:>3d}/{len(acc):<1d}{m['keep_frac']:>6.3f}")

    err = max(r["equiv_err"] for r in rows)
    print(f"\n时域混合 ⇔ 频域按列混合 的最大偏差：{err:.3e}（应为机器精度量级）")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"corpus": a.corpus, "n_speakers": len(spk), "rows": rows}, f, indent=2)
    print(f"已写出 {a.out}")


if __name__ == "__main__":
    main()
