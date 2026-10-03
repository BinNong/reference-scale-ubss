"""真实语音数据的 UBSS 实验构造（LibriSpeech dev-clean）。

为什么用真实语音
----------------
主实验的源是逐 TF 点独立 Bernoulli 复高斯，其稀疏结构是**合成**的：单源点上
只有一个源活跃，多源点上若干源叠加，且各点独立。这一模型把 UBSS 简化到了
极限，任何门限准则的失效模式都可能被掩盖。真实语音则带来三件事：

1. **时频结构不独立**：语音在 TF 面上呈连续谐波脊与共振峰团块，活跃点在时间
   与频率方向都强相关。SSP 方向的聚集性因此与合成模型不同。
2. **活动模式非 i.i.d.**：每个源有自己的静音期，π₀/π₁ 由语音的停顿结构决定，
   而不是由人为设定的 p 决定。
3. **噪声不是白的**：真实录音的背景噪声有色、非平稳、非高斯。

模型与合成实验**完全同构**（不含任何额外近似）
-----------------------------------------------
    X(f,t) = A·S(f,t) + N(f,t),   A ∈ R^{M×N},  M < N  （欠定）

瞬时混合在 STFT 域**精确**成立（STFT 是线性变换），故

    STFT(Σ_n a_mn·s_n(t)) = Σ_n a_mn·STFT(s_n(t))

本模块在**时域**做瞬时混合与加噪、再统一 STFT，与直接频域按列混合严格等价
（`self_check_equivalence` 给出数值验证），因此真实数据实验与合成实验共用同一
套基线、同一套指标、同一套恢复器，不存在"两套流程"的问题。

噪声
----
  · ``gauss``  —— 复高斯白噪声（与合成实验一致，用于与理论闭式对照）；
  · ``babble`` —— 若干真实语音段叠加成的嘈杂人声（非高斯、有色、非平稳），
                  作为对噪声假设的稳健性检验。

稠密源
------
把 k 个真实语音段叠加成一个几乎在所有 TF 点都活跃的信号，作为"非稀疏源"。
这使真实数据上也能复现"源不稀疏"的一端，而不必人为改动语音本身。
"""
from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.signal import stft as scipy_stft

import data as D

FS_DEFAULT = 16000


# ==========================================================================
# 语料扫描与加载
# ==========================================================================

def scan_corpus(root: str, use_cache: bool = True) -> list[dict]:
    """枚举语料中的音频文件。

    期望布局 ``<root>/LibriSpeech/dev-clean/<speaker>/<chapter>/<file>.flac``；
    若层级不同，退化为"文件所在目录的上一层"作为说话人标识。

    **完整性校验**：``soundfile.info`` 只读文件头，无法发现截断的流（例如从部分
    下载的 tar 包解包得到的文件），这类文件会在真正解码时抛
    ``flac decoder lost sync``，把整个实验打断。因此这里对每个文件做一次完整解码，
    失败者直接跳过；结果缓存到 ``<root>/_valid_files.json``，后续进程直接复用。
    """
    cache_path = os.path.join(root, "_valid_files.json")
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached and all(os.path.exists(it["path"]) for it in cached):
                return cached
        except Exception:
            pass

    items = []
    n_bad = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith((".flac", ".wav")):
                continue
            path = os.path.join(dirpath, fn)
            try:
                x, sr = sf.read(path, dtype="float64", always_2d=True)
            except Exception:
                n_bad += 1
                continue
            if x.shape[0] < 16000:          # 短于 1 秒的丢弃
                continue
            parts = os.path.relpath(path, root).split(os.sep)
            spk = parts[2] if len(parts) >= 4 else (parts[-2] if len(parts) >= 2 else "unknown")
            items.append({"path": path, "speaker": spk,
                          "dur": x.shape[0] / float(sr), "sr": sr})
    items.sort(key=lambda d: (d["speaker"], d["path"]))
    if n_bad:
        print(f"[scan_corpus] 跳过 {n_bad} 个无法解码的文件（截断/损坏）")
    if use_cache and items:
        try:
            with open(cache_path, "w") as f:
                json.dump(items, f)
        except Exception:
            pass
    return items


def load_segment(path: str, dur_s: float, fs: int = FS_DEFAULT,
                 mode: str = "energy", rng: np.random.Generator | None = None) -> np.ndarray:
    """读取一段时长 ``dur_s`` 的音频并归一化到单位功率。

    mode='energy' 取**能量最大的连续窗口**（标准做法：保证取到的是发声段而非
    长静音，否则"源"的大部分时频点是零，会人为把问题变简单）；
    mode='random' 取随机起点（用于构造 babble 这类需要多样性的信号）。
    """
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)                                   # 多声道取平均
    if sr != fs:
        x = resample_poly(x, fs, sr)
    L = int(round(dur_s * fs))
    if x.size < L:
        reps = int(np.ceil(L / max(x.size, 1)))
        x = np.tile(x, reps)
    if mode == "energy":
        # 用滑动能量挑最活跃的窗口（粗步长即可，避免 O(L²)）
        step = max(1, L // 8)
        starts = np.arange(0, x.size - L + 1, step)
        e = np.array([np.mean(x[s:s + L] ** 2) for s in starts])
        s0 = int(starts[int(np.argmax(e))])
    else:
        if rng is None:
            rng = np.random.default_rng()
        s0 = int(rng.integers(0, max(1, x.size - L + 1)))
    seg = x[s0:s0 + L].copy()
    seg = seg - seg.mean()
    pw = float(np.mean(seg ** 2))
    return seg / np.sqrt(max(pw, 1e-30))


def load_pool(corpus: list[dict], n_files: int, dur_s: float,
              fs: int = FS_DEFAULT, seed: int = 0,
              min_speakers: int = 0) -> dict:
    """预载一批语音段到内存，供后续反复组装（避免每个配置都读盘）。

    按"每个说话人至多两条"挑选，保证说话人多样性。
    """
    rng = np.random.default_rng(seed)
    by_spk: dict[str, list[dict]] = {}
    for it in corpus:
        if it["dur"] >= dur_s * 0.8:
            by_spk.setdefault(it["speaker"], []).append(it)
    spks = sorted(by_spk.keys())
    if len(spks) < min_speakers:
        raise RuntimeError(f"可用说话人 {len(spks)} 少于需求 {min_speakers}")
    rng.shuffle(spks)

    picks: list[dict] = []
    per_spk: dict[str, int] = {}
    for spk in spks:
        k = min(2, len(by_spk[spk]))
        idx = rng.permutation(len(by_spk[spk]))[:k]
        for i in idx:
            picks.append(by_spk[spk][int(i)])
            per_spk[spk] = per_spk.get(spk, 0) + 1
        if len(picks) >= n_files:
            break
    picks = picks[:n_files]

    sig = np.stack([load_segment(p["path"], dur_s, fs, mode="energy")
                    for p in picks], axis=0)
    return {"sig": sig, "speaker": np.array([p["speaker"] for p in picks]),
            "path": np.array([p["path"] for p in picks]),
            "dur_s": dur_s, "fs": fs, "n_speakers": len(set(p["speaker"] for p in picks))}


# ==========================================================================
# 稠密源与宽带噪声
# ==========================================================================

def make_babble(pool: dict, length: int, n_streams: int,
                rng: np.random.Generator, exclude_idx=()) -> np.ndarray:
    """把 ``n_streams`` 个真实语音段以随机时移叠加成稠密干扰信号。

    叠加后各语音的静音期被互相填补，得到的信号在几乎每个 TF 点都活跃——
    这正是"非稀疏源"在真实数据上的自然实现，同时它也是标准的
    "babble noise"（嘈杂人声），可兼作有色非高斯噪声。
    """
    src = pool["sig"]
    n = src.shape[0]
    cand = [i for i in range(n) if i not in set(exclude_idx)]
    if len(cand) < n_streams:
        cand = list(range(n))
    chosen = rng.choice(cand, size=n_streams, replace=len(cand) < n_streams)
    out = np.zeros(length, dtype=np.float64)
    for i in chosen:
        s = src[int(i)]
        if s.size >= length:
            s0 = int(rng.integers(0, s.size - length + 1))
            piece = s[s0:s0 + length]
        else:
            piece = np.tile(s, int(np.ceil(length / s.size)))[:length]
        out += rng.uniform(0.6, 1.0) * piece
    pw = float(np.mean(out ** 2))
    return out / np.sqrt(max(pw, 1e-30))


# ==========================================================================
# 问题组装
# ==========================================================================

def _stft(x: np.ndarray, nperseg: int, hop: int) -> np.ndarray:
    """与 data.stft_sources 同参数的 STFT（boundary=None，无边界填充）。"""
    _, _, Z = scipy_stft(x, nperseg=nperseg, noverlap=nperseg - hop,
                         axis=-1, boundary=None)
    return Z


def make_real_problem(pool: dict, n_sources: int = 4, m_obs: int = 2,
                      win: int = 1024, hop: int | None = None,
                      snr_db: float | None = 20.0, noise: str = "gauss",
                      n_dense: int = 0, dur_s: float = 6.0,
                      min_angle_deg: float = 12.0, seed: int = 0,
                      support_drop_db: float = 30.0,
                      active_over_noise: float = 1.0,
                      dom_ratio: float = 0.9,
                      allow_speaker_reuse: bool = True) -> dict:
    """构造一个真实语音的 UBSS 问题实例（接口与 ``data.make_problem`` 兼容）。

    Parameters
    ----------
    n_sources : 源总数 N（含稠密源）
    n_dense   : 其中由 babble 构成的稠密源个数（0 ≤ n_dense < n_sources）
    win, hop  : STFT 窗长与帧移（hop 默认 win//4）
    noise     : 'gauss' | 'babble' | None
    support_drop_db : 支撑集口径下"源活跃"的相对门限（低于源均值该 dB 数视为不活跃）
    active_over_noise : **噪声参考口径**下"源活跃"的门限：源功率 > 该倍数 × 每分量噪声功率
    dom_ratio : 支配性判据门限（最大源功率 / 全部源功率之和 ≥ 该值视为 WDO 成立）
    仅用于**统计**真实数据的稀疏结构，不参与任何算法。
    """
    rng = np.random.default_rng(seed)
    if hop is None:
        hop = win // 4

    if pool["n_speakers"] < n_sources - n_dense and not allow_speaker_reuse:
        raise RuntimeError(
            f"说话人不足：{pool['n_speakers']} < {n_sources - n_dense}"
            "（若允许复用说话人请置 allow_speaker_reuse=True）")
    # ---- 选源：优先让语音源来自不同说话人 ----
    spk = pool["speaker"]
    order = rng.permutation(len(spk))
    used_spk, src_idx = set(), []
    for i in order:
        if len(src_idx) >= n_sources - n_dense:
            break
        if spk[int(i)] in used_spk:
            continue
        used_spk.add(spk[int(i)])
        src_idx.append(int(i))
    if len(src_idx) < n_sources - n_dense:                # 说话人不够时允许重复
        for i in order:
            if len(src_idx) >= n_sources - n_dense:
                break
            if int(i) not in src_idx:
                src_idx.append(int(i))

    L = pool["sig"].shape[1]
    S_time = np.zeros((n_sources, L), dtype=np.float64)
    kind = ["speech"] * n_sources
    for j, i in enumerate(src_idx):
        S_time[j] = pool["sig"][i]
    # 稠密源（babble）
    for j in range(n_sources - n_dense, n_sources):
        S_time[j] = make_babble(pool, L, n_streams=4, rng=rng, exclude_idx=src_idx)
        kind[j] = "babble"

    # ---- 混合矩阵与瞬时混合（时域，STFT 域精确等价）----
    A = D.gen_mixing_matrix(m_obs, n_sources, rng, min_angle_deg)
    X_time = A @ S_time                                   # (M, L)

    # ---- STFT ----
    S_tf = _stft(S_time, win, hop)                        # (N, F, T)
    X_clean_tf = _stft(X_time, win, hop)                  # (M, F, T)

    # ---- 数值验证：时域混合 ⇒ 频域按列混合（应精确相等）----
    X_direct = np.einsum("mn,nft->mft", A, S_tf)
    equiv_err = float(np.max(np.abs(X_direct - X_clean_tf)))

    # ---- 噪声 ----
    M, F, T = X_clean_tf.shape
    sig_pow = float(np.mean(np.abs(X_clean_tf) ** 2))
    noise_tf, nu_true = None, 0.0
    if noise is not None and snr_db is not None:
        noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
        if noise == "gauss":
            n_tf = (rng.standard_normal((M, F, T)) + 1j * rng.standard_normal((M, F, T)))
        elif noise == "babble":
            n_tf = np.zeros((M, F, T), dtype=np.complex128)
            for m in range(M):
                b = make_babble(pool, L, n_streams=5, rng=rng)   # 每个传感器独立的一段
                n_tf[m] = _stft(b, win, hop)
        else:
            raise ValueError(f"未知噪声类型 {noise}")
        cur = float(np.mean(np.abs(n_tf) ** 2))
        n_tf = n_tf * np.sqrt(noise_pow / max(cur, 1e-30))     # 精确命中目标 SNR
        noise_tf = n_tf
        nu_true = float(m_obs * np.mean(np.abs(n_tf) ** 2))     # ν = M·s²，s² 为每分量噪声功率
        X_tf = X_clean_tf + n_tf
    else:
        X_tf = X_clean_tf

    # ---- 真实数据的稀疏统计（用真值源 + 已知噪声，仅用于诊断）----
    #
    # 两套口径，各有用处：
    #   (a) 支撑集口径：源在其均值以下 30 dB 视为"不活跃"。与合成实验的
    #       Bernoulli 激活掩码含义一致，便于横向对照。
    #   (b) **噪声参考口径**：源功率高于噪声功率才算"活跃"。这才是与理论
    #       （π₀ = "没有任何源贡献"的时频点比例）以及盲估计器前提对齐的定义
    #       ——估计器需要的是"纯噪声点"，即所有源都低于噪声底的点。
    #   另外给出**支配性**统计：WDO 近似成立要求每个点上至多一个源贡献显著，
    #   故统计"最大源占总源功率 ≥ ρ 的时点比例"。
    v_n = np.abs(S_tf) ** 2                                     # (N,F,T) 源功率
    tot_src = v_n.sum(axis=0)                                   # (F,T)
    s2_ref = (nu_true / m_obs) if nu_true > 0 else float(np.median(tot_src) / 1e3)

    act = v_n > (10.0 ** (-support_drop_db / 10.0)) * v_n.mean(axis=(1, 2), keepdims=True)
    n_act = act.sum(axis=0)

    act_nf = v_n > active_over_noise * s2_ref                   # 噪声参考口径
    n_act_nf = act_nf.sum(axis=0)

    dom = v_n.max(axis=0) / np.maximum(tot_src, 1e-300)
    sig_present = tot_src > s2_ref
    single_dom = sig_present & (dom >= dom_ratio)

    e_tf = np.sum(np.abs(X_tf) ** 2, axis=0).ravel()
    stats = {
        "pi0": float(np.mean(n_act_nf == 0)),
        "pi1": float(np.mean(n_act_nf == 1)),
        "pi2p": float(np.mean(n_act_nf >= 2)),
        "pi0_supp": float(np.mean(n_act == 0)),
        "pi1_supp": float(np.mean(n_act == 1)),
        "pi2p_supp": float(np.mean(n_act >= 2)),
        "sig_frac": float(np.mean(sig_present)),
        "single_dom_frac": float(np.mean(single_dom)),
        "single_dom_among_sig": float(np.mean(single_dom[sig_present]))
        if np.any(sig_present) else float("nan"),
        "overlap": float(np.mean(n_act)),
        "overlap_nf": float(np.mean(n_act_nf)),
        "gini": float(D.gini_index(S_tf)),
        "active_ratio": float(np.mean(act)),
        "median_e": float(np.median(e_tf)),
        "mean_e": float(np.mean(e_tf)),
        "nu_true": nu_true,
        "r_true": float(np.median(e_tf) / nu_true) if nu_true > 0 else float("nan"),
        "snr_true_db": float(10.0 * np.log10(sig_pow / max(np.mean(np.abs(noise_tf) ** 2), 1e-30)))
        if noise_tf is not None else None,
    }

    return {
        "A": A,
        "S_tf": S_tf,
        "S_time": S_time,
        "S_kind": kind,
        "X_tf": X_tf,
        "X_clean_tf": X_clean_tf,
        "S_all": D.tf_to_real(S_tf),
        "X_all": D.tf_to_real(X_tf),
        "noise_tf": noise_tf,
        "m": m_obs, "n": n_sources, "F": F, "T": T,
        "cfg_key": f"real_w{win}" + (f"_d{n_dense}" if n_dense else ""),
        "cfg_label": f"LibriSpeech, win={win}, N={n_sources}"
                     + (f", {n_dense} dense" if n_dense else ""),
        "snr_db": snr_db,
        "win": win, "hop": hop, "dur_s": dur_s,
        "noise_kind": noise, "n_dense": n_dense,
        "equiv_err": equiv_err,
        "seed": seed,
        **stats,
    }


def self_check_equivalence(pool: dict, seed: int = 0) -> dict:
    """验证"时域瞬时混合 ⇒ 频域按列混合"的精确等价性与 SNR 命中精度。"""
    out = {}
    for key, kw in [("w256", dict(win=256)), ("w1024", dict(win=1024))]:
        p = make_real_problem(pool, n_sources=4, m_obs=2, snr_db=20.0, seed=seed, **kw)
        out[key] = {"equiv_err": p["equiv_err"], "snr_true_db": p["snr_true_db"],
                    "F": p["F"], "T": p["T"], "pi0": p["pi0"], "pi1": p["pi1"],
                    "overlap": p["overlap"], "r_true": p["r_true"],
                    "gini": p["gini"], "nu_true": p["nu_true"],
                    "median_e": p["median_e"]}
    return out
