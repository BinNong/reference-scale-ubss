"""R8-4  真实录音房间：REVERB2014 **实测 RIR** + **实录各向同性噪声** + 真实语音。

响应 CSSP 意见 Major 5 / Minor 8：稿件此前的"真实数据"是 LibriSpeech 语音 +
**合成**瞬时混合 + **合成**高斯/babble 噪声；评审要求至少补一种真实录音场景。

数据来自 OpenSLR **SLR28「Room Impulse Response and Noise Database」**（Apache-2.0）：
REVERB 2014 在**真实房间**里实测的 8 元环形阵列 RIR，以及同一批房间**实录**的
各向同性噪声（8 通道、30 s、16 kHz）。这是 REVERB 挑战赛的标准素材。

## 先说这一步定下来的口径（踩过一次的坑）

逐通道**必须去直流**（全稿 `realdata.py` 就是 `seg = seg - seg.mean()`）。
更关键的是：**实录房间噪声在频域上跨约 6 个数量级**（f=0 的功率是中位频点的
3×10⁴ 倍，低频 HVAC/建筑振动主导，呈 1/f 型）。因此：
  · **跨频点合并的单一地板**（全稿 §9 真实语音用的就是这种）在这类噪声上**必然失效**——
    σ̂² 的下尾拟合会被低频巨尾压垮，实测 σ̂²/σ² 掉到 10⁻³ 量级；
  · 正确的做法是**逐频带地板**（§A.14 已研究过 banded floor，此处把它用到真实录音上）。
本脚本因此**两种口径都报**：全局那列是"失效的幅度"，逐带那列才是可用的量。

## 两个子研究

**A 实录噪声上的下尾假设**（逐房间）
  8 通道实录噪声 → 取 M 个通道 → STFT → 逐 TF 点能量 e=Σ_m|X_m|²。
  模型律：噪声点 e ~ (σ²/2)·χ²_{2M}。报告
    · 全局 σ̂²/σ² 与全局下尾比（z=2e/σ²_true 的经验分位数 ÷ χ²_{2M} 分位数）；
    · **逐频带**下尾比（在每个频点内算，再取跨频点的中位数与 10%/90% 分位）；
    · 逐频带动态范围（dB）——它是全局口径失效的原因。

**B 真实房间混响下的门限**（逐房间 × seed）
  4 段 LibriSpeech（N=4）各自与**同房间实测的 4 个声源位置 RIR**卷积，取 M=2 个阵列
  通道求和（欠定 M<N），叠加同房间**实录**噪声到 SNR=20 dB。逐点真值来自**已知的
  各个源像**：源 n 在 (f,t) 活跃 ⟺ 其参考通道能量超过自身整幅 TF 均值（κ=1，声明）。
  比较（其余判据一律不施加，只比**能量参考尺度**）：
    · conventional median  e_f > t_e·median(e_f)，t_e=0.02
    · NF-SSP               e_f > τ·ν̂_f，ν̂_f = M·σ̂²_f，τ=Q_{1−α}(χ²_{2M})/(2M)
  两种地板都报：**逐带**（可用）与**全局**（失效）。报告假开率、单源召回，以及
  median 规则要达到与 NF 相同假开率所需的系数 t_e\*——它跨房间的漂移即"NF 更稳定"的证据。

用法：
    python3 r8_real_room.py --rir-root <dir> --corpus <root> --out ../results/r8_real_room.json
    python3 r8_real_room.py ... --rooms smallroom1 --seeds 1 --workers 1   # 冒烟测试
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import stft as scipy_stft
from scipy.stats import chi2

import realdata as RD
from nfr import estimate_noise_power, point_energies, threshold_ratio

FS = 16000
NPERSEG, NOVERLAP = 256, 128
ROOMS = ["smallroom1", "smallroom2", "mediumroom1", "mediumroom2", "largeroom1", "largeroom2"]
POSS = ["far_angla", "far_anglb", "near_angla", "near_anglb"]
MICS = [0, 4]
M_OBS = len(MICS)
N_SRC = len(POSS)
SNR_DB = 20.0
TAIL_Q = [0.02, 0.05, 0.10, 0.20]
KAPPA = 1.0
ALPHA = 1e-4
TE_CONV = 0.02

_CORPUS: list[dict] = []


def _init_corpus(root: str) -> None:
    """每个 worker 自己 set 一次。

    必须用 initializer：本模块走 **spawn**，子进程会重新 import，`main()` 里给全局
    赋的值**不会**被继承（实测会让 worker 看到空语料）。`scan_corpus` 有
    `_valid_files.json` 缓存，重复调用只读 JSON。
    """
    global _CORPUS
    _CORPUS = RD.scan_corpus(root)


def _detrend(x: np.ndarray) -> np.ndarray:
    """逐通道去直流——与全稿 `realdata.py` 的 `seg = seg - seg.mean()` 同口径。"""
    return x - x.mean(axis=-1, keepdims=True)


def _stft(x: np.ndarray) -> np.ndarray:
    _, _, Z = scipy_stft(x, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, axis=-1,
                         boundary=None, padded=False)
    return Z


def _band_stats(X: np.ndarray) -> tuple[float, dict]:
    """逐频带的（下尾比中位数, 分位）与动态范围（dB）。"""
    r = []
    for f in range(X.shape[1]):
        E = np.abs(X[:, f, :]) ** 2
        s2f = float(np.mean(E))
        if s2f <= 0:
            continue
        zf = 2.0 * E.sum(axis=0) / s2f
        r.append(float(np.quantile(zf, 0.05) / chi2.ppf(0.05, 2 * M_OBS)))
    r = np.asarray(r)
    pf = np.mean(np.abs(X) ** 2, axis=(0, 2))
    dr = 10.0 * np.log10(float(pf.max()) / max(float(pf.min()), 1e-300))
    return dr, dict(median=float(np.median(r)), q10=float(np.quantile(r, 0.10)),
                    q90=float(np.quantile(r, 0.90)), nbins=int(r.size))


def study_a_room(room: str, rir_root: str) -> dict:
    f = sorted(glob.glob(os.path.join(rir_root, f"RVB2014_type1_noise_{room}_*.wav")))[0]
    x, sr = sf.read(f, dtype="float64", always_2d=True)
    assert sr == FS, (f, sr)
    X = _stft(_detrend(x[:, MICS].T))
    e = point_energies(X)
    s2t = float(np.mean(np.abs(X) ** 2))
    est = estimate_noise_power(e, M_OBS)
    z = 2.0 * e / s2t
    dr, band = _band_stats(X)
    return dict(study="noise", room=room, mics=MICS, n_pts=int(e.size),
                s2_hat_over_true=float(est["s2"] / s2t), spread=float(est["spread"]),
                tail_ratio={float(q): float(np.quantile(z, q) / chi2.ppf(q, 2 * M_OBS))
                            for q in TAIL_Q},
                band_dynamic_range_db=float(dr), band_tail=band)


def _rir(room: str, pos: str, rir_root: str) -> np.ndarray:
    f = os.path.join(rir_root, f"RVB2014_type1_rir_{room}_{pos}.wav")
    h, sr = sf.read(f, dtype="float64", always_2d=True)
    assert sr == FS, (f, sr)
    return h[:, MICS].T


def _band_energies(X: np.ndarray) -> list[np.ndarray]:
    """逐频点的逐时刻能量 e_f(t) = Σ_m|X_m(f,t)|²。"""
    return [np.sum(np.abs(X[:, f, :]) ** 2, axis=0) for f in range(X.shape[1])]


def study_b_case(arg) -> dict:
    room, seed, rir_root = arg
    rng = np.random.default_rng(4000 + seed)
    pool = _CORPUS
    idx = rng.choice(len(pool), size=N_SRC, replace=False)
    segs = [RD.load_segment(pool[i]["path"], 3.0, fs=FS, mode="energy", rng=rng) for i in idx]
    L = min(len(s) for s in segs)
    segs = [s[:L] for s in segs]

    images, clean = [], np.zeros((M_OBS, L))
    for n, s in enumerate(segs):
        h = _rir(room, POSS[n], rir_root)
        img = np.stack([np.convolve(s, h[m])[:L] for m in range(M_OBS)])
        images.append(img)
        clean += img

    nf = sorted(glob.glob(os.path.join(rir_root, f"RVB2014_type1_noise_{room}_*.wav")))[
        int(rng.integers(0, 9))]
    nz, sr = sf.read(nf, dtype="float64", always_2d=True)
    nz = nz[:, MICS].T
    off = int(rng.integers(0, nz.shape[1] - L))
    nz = nz[:, off:off + L]
    nz = nz * np.sqrt(np.mean(clean ** 2) / max(np.mean(nz ** 2), 1e-30)
                      / 10 ** (SNR_DB / 10.0))

    X = _stft(_detrend(clean + nz))

    # 逐点真值：J(f,t) = 活跃源数（κ=1，以各源参考通道整幅均值归一）
    J = np.zeros(X.shape[1:], dtype=int)
    for img in images:
        Z = np.abs(_stft(_detrend(img))[0]) ** 2
        J += (Z > KAPPA * Z.mean()).astype(int)

    tau = threshold_ratio(M_OBS, ALPHA)
    ef = _band_energies(X)                       # 逐频点
    z0 = (J == 0).ravel()
    z1 = (J == 1).ravel()
    n0, n1 = int(z0.sum()), int(z1.sum())
    if n0 == 0 or n1 == 0:
        return dict(study="gate", room=room, seed=seed, frac_J0=float((J == 0).mean()),
                    frac_J1=float((J == 1).mean()), mean_J=float(J.mean()),
                    degenerate=True)

    # 展平顺序是 f*T+t（point_energies 把 (F,T) ravel），故逐频点是**连续块**
    Tn = X.shape[2]
    a_nf = np.zeros(J.size, bool)
    a_med = np.zeros(J.size, bool)
    med_map: dict[int, float] = {}
    for f in range(X.shape[1]):
        sl = slice(f * Tn, (f + 1) * Tn)
        e = ef[f]
        med_map[f] = float(np.median(e))
        nu_f = M_OBS * estimate_noise_power(e, M_OBS)["s2"]
        a_nf[sl] = e > tau * nu_f
        a_med[sl] = e > TE_CONV * med_map[f]
    fa_nf, fa_med = float(a_nf[z0].mean()), float(a_med[z0].mean())
    sr_nf, sr_med = float(a_nf[z1].mean()), float(a_med[z1].mean())

    # median 要达到与 NF 相同假开率所需的系数：逐频点求后再取中位
    te = []
    for f in range(X.shape[1]):
        sl = slice(f * Tn, (f + 1) * Tn)
        sel0 = z0[sl]
        tgt = a_nf[sl][sel0]
        if sel0.sum() and tgt.any():
            e = ef[f][sel0]
            te.append(float(np.quantile(e, 1 - float(tgt.mean())) / med_map[f]))
    te_star = float(np.median(te)) if te else float("inf")

    # ---- 全局（对照）：把整幅 TF 面当一个尺度
    e_all = point_energies(X)
    g_nu = M_OBS * estimate_noise_power(e_all, M_OBS)["s2"]
    g_med = float(np.median(e_all))
    a_nf_g = e_all > tau * g_nu
    a_med_g = e_all > TE_CONV * g_med

    return dict(study="gate", room=room, seed=seed,
                frac_J0=float((J == 0).mean()), frac_J1=float((J == 1).mean()),
                mean_J=float(J.mean()), degenerate=False,
                s2_hat_over_true=float(estimate_noise_power(e_all, M_OBS)["s2"]
                                       / float(np.mean(np.abs(_stft(_detrend(nz))) ** 2))),
                fa_nf=fa_nf, fa_med=fa_med, sr_nf=sr_nf, sr_med=sr_med,
                te_star=te_star, tau=float(tau),
                fa_nf_global=float(a_nf_g[z0].mean()),
                fa_med_global=float(a_med_g[z0].mean()),
                sr_nf_global=float(a_nf_g[z1].mean()),
                sr_med_global=float(a_med_g[z1].mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rir-root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rooms", nargs="*", default=ROOMS)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    t0 = time.perf_counter()
    A = [study_a_room(r, a.rir_root) for r in a.rooms]
    print(f"=== A. 实录噪声（{len(A)} 房间，M={M_OBS}）===")
    print(f"{'room':<13}{'全局σ̂²/σ²':>11}{'动态范围dB':>11}{'全局尾比@.05':>13}"
          f"{'逐带尾比中位':>13}{'逐带10%':>9}{'逐带90%':>9}")
    for r in A:
        print(f"{r['room']:<13}{r['s2_hat_over_true']:>11.4f}{r['band_dynamic_range_db']:>11.1f}"
              f"{r['tail_ratio'][0.05]:>13.4f}{r['band_tail']['median']:>13.3f}"
              f"{r['band_tail']['q10']:>9.3f}{r['band_tail']['q90']:>9.3f}")

    _init_corpus(a.corpus)
    print(f"\n语料 {len(_CORPUS)} 条")
    if not _CORPUS:
        raise SystemExit(f"✗ 语料为空：{a.corpus}")

    args = [(r, s, a.rir_root) for r in a.rooms for s in range(a.seeds)]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx,
                             initializer=_init_corpus, initargs=(a.corpus,)) as ex:
        G = list(ex.map(study_b_case, args))
    G = [x for x in G if not x.get("degenerate")]

    print(f"\n=== B. 真实房间混响（{len(a.rooms)} 房间 × {a.seeds} seeds）===")
    print(f"{'room':<13}{'J=0':>7}{'J=1':>7}|{'逐带 FA_NF':>11}{'FA_med':>8}{'SR_NF':>8}{'SR_med':>8}"
          f"{'t_e*':>8}|{'全局 FA_NF':>11}{'FA_med':>8}")
    for r in a.rooms:
        sub = [x for x in G if x["room"] == r]
        if not sub:
            print(f"{r:<13}  （退化，无 J=0 或 J=1 点）")
            continue
        f = lambda k: float(np.nanmean([x[k] for x in sub]))
        print(f"{r:<13}{f('frac_J0'):>7.3f}{f('frac_J1'):>7.3f}|"
              f"{f('fa_nf'):>11.4f}{f('fa_med'):>8.3f}{f('sr_nf'):>8.3f}{f('sr_med'):>8.3f}"
              f"{f('te_star'):>8.2f}|{f('fa_nf_global'):>11.4f}{f('fa_med_global'):>8.3f}")

    with open(a.out, "w") as fh:
        json.dump(dict(records=A + G, rooms=a.rooms, seeds=a.seeds, mics=MICS, poss=POSS,
                       snr_db=SNR_DB, alpha=ALPHA, te_conv=TE_CONV, kappa=KAPPA,
                       nperseg=NPERSEG, noverlap=NOVERLAP, fs=FS,
                       corpus=a.corpus, source="OpenSLR SLR28 (REVERB2014 real RIR + real noise)",
                       elapsed_s=time.perf_counter() - t0), fh)
    print(f"\n记录 {len(A) + len(G)} 条，用时 {time.perf_counter()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
