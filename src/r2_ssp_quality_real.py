"""R2-1b  真实语音上的单源点掩码质量（审稿意见 MC6，真实数据部分）。

与 r2_ssp_quality.py 同一套检出质量指标，但问题实例来自 LibriSpeech，
逐点标注按论文 §9.2 的口径：源功率 > 1.0 × 每分量噪声功率 记为活跃。

用法：
    python3 r2_ssp_quality_real.py --seeds 4 --workers 10 \
        --out ../results/r2_ssp_quality_real.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import realdata as RD
from nfr import nfr_mask, point_energies
from baselines import ssp_mask_from_complex
from r2_ssp_quality import score

M_OBS, DUR = 2, 3.0
CONFIGS = [
    ("R1_res", "w256", dict(win=256, n_sources=4, snr_db=20.0)),
    ("R1_res", "w512", dict(win=512, n_sources=4, snr_db=20.0)),
    ("R1_res", "w1024", dict(win=1024, n_sources=4, snr_db=20.0)),
    ("R1_res", "w2048", dict(win=2048, n_sources=4, snr_db=20.0)),
    ("R2_n", "N3", dict(win=1024, n_sources=3, snr_db=20.0)),
    ("R2_n", "N5", dict(win=1024, n_sources=5, snr_db=20.0)),
    ("R2_n", "N6", dict(win=1024, n_sources=6, snr_db=20.0)),
    ("R3_snr", "snr00", dict(win=1024, n_sources=4, snr_db=0.0)),
    ("R3_snr", "snr10", dict(win=1024, n_sources=4, snr_db=10.0)),
    ("R3_snr", "snr30", dict(win=1024, n_sources=4, snr_db=30.0)),
    ("R3_snr", "snr40", dict(win=1024, n_sources=4, snr_db=40.0)),
    ("R4_dense", "d1", dict(win=1024, n_sources=4, snr_db=20.0, n_dense=1)),
    ("R4_dense", "d2", dict(win=1024, n_sources=4, snr_db=20.0, n_dense=2)),
    ("R5_noise", "babble", dict(win=1024, n_sources=4, snr_db=20.0, noise="babble")),
    ("R5_noise", "clean", dict(win=1024, n_sources=4, snr_db=None, noise=None)),
]

_CORPUS = None


def corpus(root: str):
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = RD.scan_corpus(root)
    return _CORPUS


def build(root: str, kw: dict, seed: int, n_files: int = 40) -> dict:
    need = kw["n_sources"] - kw.get("n_dense", 0)
    pool = RD.load_pool(corpus(root), n_files=n_files, dur_s=DUR, seed=seed,
                        min_speakers=need)
    return RD.make_real_problem(pool, m_obs=M_OBS, dur_s=DUR, seed=1000 + seed,
                                hop=kw.get("win", 1024) // 2, **kw)


def real_labels(prob: dict) -> np.ndarray:
    """按 §9.2 口径标注：源功率 > 1.0 × 每分量噪声功率。"""
    v = np.abs(prob["S_tf"]) ** 2
    nu = float(prob["nu_true"])
    if nu > 0:
        s2_ref = nu / M_OBS
    else:
        s2_ref = float(np.median(v.sum(axis=0)) / 1e3)
    return (v > 1.0 * s2_ref).sum(axis=0)


def eval_job(job: tuple[str, str, str, dict, int, str]) -> list[dict]:
    exp, tag, root, kw, seed, _ = job
    prob = build(root, kw, seed)
    X = prob["X_tf"]
    J = real_labels(prob)
    A = prob["A"]

    Xr, Xi = X.real, X.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    eps = 1e-15
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
    d = np.maximum(nr + ni, eps)
    base = (np.abs(cos) > 0.98) & (nr / d > 0.1) & (ni / d > 0.1)

    E = point_energies(X).reshape(X.shape[1], X.shape[2])
    med_e, mx_e = float(np.median(E)), float(np.max(E))
    masks = {
        "single-source criterion only": base,
        "classical $t_e{=}0.02$": base & (E > 0.02 * med_e),
        "median $t_e{=}5$": base & (E > 5.0 * med_e),
        "max-referenced $c{=}0.05$": base & (E > 0.05 * mx_e),
    }
    nf, dg = nfr_mask(X, M_OBS, alpha=1e-4, use_self_check=True)
    masks["NF gate $\\alpha{=}10^{-4}$"] = nf

    out = []
    for name, mk in masks.items():
        r = score(mk, J, X, A, base)
        r.update(dom="real", case=f"{exp}:{tag}", seed=seed, method=name,
                 gate_on=bool(dg.get("gate_on", True)) if name.startswith("NF") else True,
                 frac_noise=float((J == 0).mean()), frac_ss=float((J == 1).mean()),
                 frac_multi=float((J >= 2).mean()))
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/ls_corpus")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="../results/r2_ssp_quality_real.json")
    a = ap.parse_args()

    jobs = [(e, t, a.root, kw, s, "") for (e, t, kw) in CONFIGS for s in range(a.seeds)]
    t0 = time.perf_counter()
    recs: list[dict] = []
    if a.workers <= 1:
        for j in jobs:
            recs += eval_job(j)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            for r in ex.map(eval_job, jobs):
                recs += r

    methods = sorted({r["method"] for r in recs})
    print(f"\n=== 真实语音 SSP 检出质量（15 配置 × {a.seeds} seeds，"
          f"{time.perf_counter()-t0:.0f}s）===")
    hdr = (f"{'method':<30}{'prec':>8}{'rec':>8}{'F1':>8}"
           f"{'noiseRej':>10}{'multiRej':>10}{'dirErr':>8}")
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        sub = [r for r in recs if r["method"] == m]
        f = lambda k: float(np.nanmean([r[k] for r in sub]))
        print(f"{m:<30}{f('precision'):>8.3f}{f('recall'):>8.3f}{f('f1'):>8.3f}"
              f"{f('noise_reject')*100:>9.2f}%{f('multi_reject')*100:>9.2f}%"
              f"{f('dir_err_med_deg'):>8.2f}")

    print("\n=== 逐配置 precision ===")
    cases = [f"{e}:{t}" for (e, t, _) in CONFIGS]
    print(f"{'case':<18}{'base':>9}" + "".join(f"{m.split()[0][:9]:>11}" for m in methods))
    for c in cases:
        sub = [r for r in recs if r["case"] == c]
        row = f"{c:<18}{np.nanmean([r['base_precision'] for r in sub]):>9.3f}"
        for m in methods:
            v = [r["precision"] for r in recs if r["method"] == m and r["case"] == c]
            row += f"{np.nanmean(v):>11.3f}"
        print(row)

    nf = [r for r in recs if r["method"].startswith("NF")]
    print(f"\n=== NF 门限自检：整体开启 {sum(r['gate_on'] for r in nf)}/{len(nf)} ===")
    for c in cases:
        v = [r["gate_on"] for r in nf if r["case"] == c]
        if not all(v):
            print(f"  {c:<18} 开启 {sum(v)}/{len(v)}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"config": {k: v for k, v in vars(a).items()},
                   "records": recs}, open(a.out, "w"))
        print(f"\n已写 {a.out}（{len(recs)} 条）")


if __name__ == "__main__":
    main()
