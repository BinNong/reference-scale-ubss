"""R2-1  单源点掩码本身的质量（审稿意见 MC6）。

审稿人的质疑是：论文只用下游的混合矩阵角度误差来论证门限有效，
无法区分「NF-SSP 真的在正确选点」与「只是碰巧让聚类结果更好」。

本脚本直接评价掩码：把每个 TF 点按真值标注为
    noise-only (J=0) / single-source (J=1) / multi-source (J>=2)
然后报告 precision / recall / F1、noise-only 拒收率、多源拒收率，
以及被接纳单源点的方向误差中位数。所有方法共用同一套单源判据
（实虚部夹角 + 均衡），因此差异完全来自能量判据本身。

用法：
    python3 r2_ssp_quality.py --seeds 8 --workers 6 --out ../results/r2_ssp_quality.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import data as D
import metrics as MT
from nfr import nfr_mask, point_energies, threshold_ratio

LADDER = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
N_SRC, M_OBS, NF_FREQ, NF_FRAME = 4, 2, 33, 64
SNR_DB = 20.0
THR_COS, THR_PART = 0.98, 0.1


# --------------------------------------------------------------------------
# 掩码构造（共用单源判据，只换能量判据）
# --------------------------------------------------------------------------
def base_mask(X_tf: np.ndarray) -> np.ndarray:
    """单源判据 + 均衡判据，不含任何能量判据。"""
    Xr, Xi = X_tf.real, X_tf.imag
    nr = np.linalg.norm(Xr, axis=0)
    ni = np.linalg.norm(Xi, axis=0)
    eps = 1e-15
    cos = np.sum(Xr * Xi, axis=0) / np.maximum(nr * ni, eps)
    denom = np.maximum(nr + ni, eps)
    both = (nr / denom > THR_PART) & (ni / denom > THR_PART)
    return (np.abs(cos) > THR_COS) & both


def truth_labels(S_tf: np.ndarray) -> np.ndarray:
    """逐 TF 点的活跃源数 J。稀疏模型下非零即活跃（精确真值）。"""
    mag2 = np.abs(S_tf) ** 2
    act = mag2 > (1e-10 * max(mag2.max(), 1e-30))
    return act.sum(axis=0)


def score(mask: np.ndarray, J: np.ndarray, X_tf: np.ndarray,
          A: np.ndarray, base: np.ndarray) -> dict:
    """给定掩码与真值标注，算检出质量。"""
    adm = np.asarray(mask, dtype=bool)
    n_pts = J.size
    is_ss, is_no, is_mu = (J == 1), (J == 0), (J >= 2)
    tp = int(np.sum(adm & is_ss))
    fp = int(np.sum(adm & ~is_ss))
    pre = tp / max(tp + fp, 1)
    rec = tp / max(int(is_ss.sum()), 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-30)

    d = dict(
        n_points=int(n_pts),
        n_admitted=int(adm.sum()),
        keep_frac=float(adm.mean()),
        precision=float(pre),
        recall=float(rec),
        f1=float(f1),
        noise_reject=float(1.0 - np.sum(adm & is_no) / max(int(is_no.sum()), 1)),
        multi_reject=float(1.0 - np.sum(adm & is_mu) / max(int(is_mu.sum()), 1)),
        n_noise_admitted=int(np.sum(adm & is_no)),
        n_multi_admitted=int(np.sum(adm & is_mu)),
        n_ss_admitted=tp,
        # 相对"只用单源判据"的净收益：精确率提升
        base_precision=float(np.sum(base & is_ss) / max(int(base.sum()), 1)),
        base_keep_frac=float(base.mean()),
    )
    # 被接纳单源点的方向误差中位数（度量"选进来的点好不好用"）
    pts = np.flatnonzero((adm & is_ss).ravel())
    if pts.size:
        Xr = X_tf.real.reshape(X_tf.shape[0], -1)[:, pts]
        nrm = np.linalg.norm(Xr, axis=0)
        keep = nrm > 1e-12
        u = Xr[:, keep] / nrm[keep]
        u = u * np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
        act = np.abs(np.abs(A.T @ u) - 1.0)
        best = np.degrees(np.arccos(np.clip(1.0 - act.min(axis=0), 0, 1)))
        d["dir_err_med_deg"] = float(np.median(best))
    else:
        d["dir_err_med_deg"] = float("nan")
    return d


# --------------------------------------------------------------------------
# 逐问题评测
# --------------------------------------------------------------------------
def eval_job(job: tuple[str, int]) -> list[dict]:
    key, seed = job
    p = D.make_problem(M_OBS, N_SRC, NF_FREQ, NF_FRAME, key, SNR_DB, seed=1000 + seed)
    X_tf, A = p["X_tf"], p["A"]
    J = truth_labels(p["S_tf"])
    base = base_mask(X_tf)
    E = point_energies(X_tf).reshape(X_tf.shape[1], X_tf.shape[2])
    med_e = float(np.median(E))
    mx_e = float(np.max(E))

    masks: dict[str, np.ndarray] = {
        "single-source criterion only": base,
        "classical $t_e{=}0.02$": base & (E > 0.02 * med_e),
        "median $t_e{=}5$": base & (E > 5.0 * med_e),
        "max-referenced $c{=}0.05$": base & (E > 0.05 * mx_e),
    }
    nf_mask, diag = nfr_mask(X_tf, M_OBS, alpha=1e-4, thr_cos=THR_COS,
                             thr_part_ratio=THR_PART, use_self_check=True)
    masks["NF gate $\\alpha{=}10^{-4}$"] = nf_mask

    out = []
    for name, mk in masks.items():
        r = score(mk, J, X_tf, A, base)
        r.update(dom="synth", case=key, seed=seed, method=name,
                 gate_on=bool(diag.get("gate_on", True)) if name.startswith("NF") else True,
                 frac_noise=float((J == 0).mean()), frac_ss=float((J == 1).mean()),
                 frac_multi=float((J >= 2).mean()))
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="../results/r2_ssp_quality.json")
    a = ap.parse_args()

    jobs = [(k, s) for k in LADDER for s in range(a.seeds)]
    t0 = time.perf_counter()
    recs: list[dict] = []
    if a.workers <= 1:
        for j in jobs:
            recs += eval_job(j)
    else:
        # spawn：父进程若已初始化 BLAS/sklearn 线程池，fork 会死锁
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
            for r in ex.map(eval_job, jobs):
                recs += r

    methods = sorted({r["method"] for r in recs})
    print(f"\n=== SSP 检测质量（{len(LADDER)} 档 × {a.seeds} seeds，"
          f"用时 {time.perf_counter()-t0:.1f}s）===")
    hdr = f"{'method':<30}{'keep%':>7}{'prec':>7}{'rec':>7}{'F1':>7}{'noiseRej':>10}{'multiRej':>10}{'dirErr':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        sub = [r for r in recs if r["method"] == m]
        f = lambda k: float(np.nanmean([r[k] for r in sub]))
        print(f"{m:<30}{f('keep_frac')*100:>7.2f}{f('precision'):>7.3f}{f('recall'):>7.3f}"
              f"{f('f1'):>7.3f}{f('noise_reject')*100:>9.2f}%{f('multi_reject')*100:>9.2f}%"
              f"{f('dir_err_med_deg'):>8.2f}")

    print("\n注：tf_gauss（稠密）按构造不存在单源点，J>=1 恒成立，"
          "故 precision/recall 无定义（表格中记为 0），仅保留以显示退化。")
    print("\n=== 逐档 F1 ===")
    print(f"{'case':<12}" + "".join(f"{m.split()[0][:10]:>12}" for m in methods))
    for k in LADDER:
        row = f"{k:<12}"
        for m in methods:
            v = [r["f1"] for r in recs if r["method"] == m and r["case"] == k]
            row += f"{np.nanmean(v):>12.3f}"
        print(row)

    print("\n=== 逐档 precision（相对『只用单源判据』的提升）===")
    print(f"{'case':<12}{'base':>10}" + "".join(f"{m.split()[0][:9]:>11}" for m in methods))
    for k in LADDER:
        sub = [r for r in recs if r["case"] == k]
        row = f"{k:<12}{np.mean([r['base_precision'] for r in sub]):>10.3f}"
        for m in methods:
            v = [r["precision"] for r in recs if r["method"] == m and r["case"] == k]
            row += f"{np.nanmean(v):>11.3f}"
        print(row)

    print("\n=== 逐档 recall ===")
    print(f"{'case':<12}" + "".join(f"{m.split()[0][:10]:>12}" for m in methods))
    for k in LADDER:
        row = f"{k:<12}"
        for m in methods:
            v = [r["recall"] for r in recs if r["method"] == m and r["case"] == k]
            row += f"{np.nanmean(v):>12.3f}"
        print(row)

    print(f"\n=== NF 门限自检触发（{a.seeds} seeds/档）===")
    nf = [r for r in recs if r["method"].startswith("NF")]
    for k in LADDER:
        v = [r["gate_on"] for r in nf if r["case"] == k]
        print(f"  {k:<12} 开启 {sum(v)}/{len(v)}")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"config": vars(a), "ladder": LADDER,
                       "thr_cos": THR_COS, "thr_part": THR_PART,
                       "records": recs}, f)
        print(f"\n已写出 {a.out}（{len(recs)} 条）")


if __name__ == "__main__":
    main()
