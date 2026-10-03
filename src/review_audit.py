"""审稿用独立审计脚本：复算论文中所有可检验的统计声明。

只读，不改动任何结果文件。依赖 numpy/scipy（在服务器 .venv 中运行）。
用法：
    ../.venv/bin/python review_audit.py
"""
import json
import numpy as np
from collections import defaultdict
from scipy import stats

R = "../results"


def load(f):
    return json.load(open(f"{R}/{f}"))


def paired(recs, key_a, key_b, group, meth_a, meth_b, seed_key="seed"):
    """对同一 (group, seed) 配对，返回配对 t 检验 p 值。"""
    A, B = {}, {}
    for r in recs:
        if r["method"] not in (meth_a, meth_b):
            continue
        k = (r[group], r[seed_key])
        (A if r["method"] == meth_a else B)[k] = r["A_angle_deg"]
    ks = sorted(set(A) & set(B))
    a = np.array([A[k] for k in ks])
    b = np.array([B[k] for k in ks])
    if len(ks) < 2:
        return np.nan, np.nan, 0
    return float(a.mean()), float(b.mean()), float(stats.ttest_rel(a, b).pvalue)


# ---------------------------------------------------------------- Table 7
def table7():
    d = load("nfr_results.json")["records"]
    print("=== Table 7 运行时间（按实验分组，均值秒）===")
    by = defaultdict(lambda: defaultdict(list))
    for r in d:
        by[r["experiment"]][r["method"]].append(r["time_s"])
    for exp in ["E1_sparsity", "E2_snr", "E3_M", "E4_N", "E6_plugin", "E8_structured"]:
        if exp not in by:
            continue
        row = "  ".join(f"{m}={np.mean(v):.3f}" for m, v in sorted(by[exp].items())
                        if m in ("NF-SSP", "SCA-default", "SCA-L1-tuned", "pf_auto_l1", "sl0_l1"))
        print(f"  {exp:<15}{row}")
    print("  论文 Table 7: NF-SSP 0.159 | SCA-default 0.164 | SCA-tuned 0.139 | pf 0.122 | sl0 1.236")


# ---------------------------------------------------------------- §8.1
def sec81():
    d = load("nfr_results.json")["records"]
    d = [r for r in d if r["experiment"] == "E1_sparsity"]
    print("\n=== §8.1 显著性（配对 t 检验，10 seeds）===")
    print(f"{'对手':<18}{'regime':<10}{'NF':>8}{'对手':>8}{'p':>12}")
    for opp in ["SCA-default", "SCA-fixed-te5", "SCA-L1-tuned"]:
        for c in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]:
            a, b, p = paired(d, None, None, "cfg_key", "NF-SSP", opp)
            A = [r["A_angle_deg"] for r in d if r["cfg_key"] == c and r["method"] == "NF-SSP"]
            B = [r["A_angle_deg"] for r in d if r["cfg_key"] == c and r["method"] == opp]
            p = float(stats.ttest_rel(A, B).pvalue)
            print(f"{opp:<18}{c:<10}{np.mean(A):>8.3f}{np.mean(B):>8.3f}{p:>12.2e}")
        print()
    print("论文 §8.1: vs 传统 p=1.6e-4 / 1.5e-3 / 5.6e-3 (p=.02/.05/.10), 0.088 (p=.20), 6.7e-3 (p=.40), 0.345 (dense)")
    print("论文 §8.1: vs te5  p=0.46 (p=.05), 0.30 (p=.10), 0.023 (p=.20), 2.7e-4 (p=.40), 1.9e-3 (dense)")


# ---------------------------------------------------------------- §9.3
def sec93():
    P = load("real_results.json")
    d = P["records"]
    cases = sorted({r["case"] for r in d})
    print("\n=== §9.3 真实语音：逐配置配对 t 检验（8 seeds）===")
    for opp in ["SCA-median", "SCA-max", "top-K", "SCA-median-opt", "SCA-max-opt", "top-K-opt"]:
        ps, sig = {}, 0
        for c in cases:
            A = [r["A_angle_deg"] for r in d if r["case"] == c and r["method"] == "NF-SSP"]
            B = [r["A_angle_deg"] for r in d if r["case"] == c and r["method"] == opp]
            p = float(stats.ttest_rel(A, B).pvalue)
            ps[c] = p
            if p < 0.05:
                sig += 1
        lo = min(ps.values())
        hi = max(ps.values())
        worst = max(ps, key=lambda k: ps[k])
        print(f"  vs {opp:<18} 显著配置数={sig}/15   p 范围 [{lo:.2e}, {hi:.2e}]  最大p在 {worst.split(':')[-1]} ({hi:.3f})")
    print("  论文 §9.3: 'paired t-tests against the alternatives reject equality in fourteen of fifteen configurations (p between 1.2e-5 and 0.05)'")


# ---------------------------------------------------------------- §9.5 / Table 17
def sec95():
    d = json.load(open(f"{R}/purity_stage2.json"))["records"]
    d = [r for r in d if r["dom"] == "real"]
    print("\n=== §9.5 真实域配对检验（15 配置 × 8 seeds 池化, n=120）===")
    pairs = [("NF", "NF+e"), ("NF", "NF+e·cos4"), ("NF+e", "NF+e·cos4"),
             ("NF+e·cos4", "NF+e·ρ2"), ("NF+e·cos4", "NF+e·ρ4"), ("NF+e·cos4", "NF+e·ρ8"),
             ("NF", "NF+e·ρ8"), ("NF+e", "NF+e·ρ8"), ("NF+e", "base+e")]
    for a, b in pairs:
        A, B = {}, {}
        for r in d:
            k = (r["case"], r["seed"])
            if r["method"] == a:
                A[k] = r["A"]
            elif r["method"] == b:
                B[k] = r["A"]
        ks = sorted(set(A) & set(B))
        av = np.array([A[k] for k in ks]); bv = np.array([B[k] for k in ks])
        p = stats.ttest_rel(av, bv).pvalue
        print(f"  {a:<14}vs {b:<14} n={len(ks):<5}{av.mean():.4f} -> {bv.mean():.4f}"
              f"  差={bv.mean()-av.mean():+.4f}  p={p:.2e}")
    print("  论文: NF 1.12 -> +energy 0.72 p=6e-5 | 0.72 -> 0.68 p=0.23 | |cos|^4 与 rho^2 打平 p=0.81")


# ---------------------------------------------------------------- Table 18
def t18():
    SR = json.load(open(f"{R}/synth_reseed.json"))["records"]
    print("\n=== Table 18 合成阶梯配对检验（池化 n=48）===")
    for grp, pairs in [("weighted", [("NF", "NF-W"), ("NF", "NF-Wc"), ("NF", "hard-opt")]),
                       ("purity", [("NF", "NF+e·ρ8"), ("NF+e", "NF+e·ρ8"), ("NF+e", "NF+e·cos4")])]:
        d = SR[grp]
        for a, b in pairs:
            A, B = {}, {}
            for r in d:
                k = (r["case"], r["seed"])
                if r["method"] == a:
                    A[k] = r["A"]
                elif r["method"] == b:
                    B[k] = r["A"]
            ks = sorted(set(A) & set(B))
            av = np.array([A[k] for k in ks]); bv = np.array([B[k] for k in ks])
            m = np.isfinite(av) & np.isfinite(bv)
            print(f"  {a:<14}vs {b:<14} n={m.sum():<5}{av[m].mean():.4f} -> {bv[m].mean():.4f}"
                  f"  p={stats.ttest_rel(av[m], bv[m]).pvalue:.2e}")
    print("  论文: NF 2.12 -> +energy 2.49 池化 p=0.19；p=.10 单档 0.32 vs 0.58 p=0.03")


# ---------------------------------------------------------------- Table 18
def t18():
    SR = json.load(open(f"{R}/synth_reseed.json"))["records"]
    print("\n=== Table 18 合成阶梯配对检验（池化 n=48）===")
    for grp, pairs in [("weighted", [("NF", "NF-W"), ("NF", "NF-Wc"), ("NF", "hard-opt")]),
                       ("purity", [("NF", "NF+e·ρ8"), ("NF+e", "NF+e·ρ8"), ("NF+e", "NF+e·cos4")])]:
        d = SR[grp]
        for a, b in pairs:
            A, B = {}, {}
            for r in d:
                k = (r["case"], r["seed"])
                if r["method"] == a:
                    A[k] = r["A"]
                elif r["method"] == b:
                    B[k] = r["A"]
            ks = sorted(set(A) & set(B))
            av = np.array([A[k] for k in ks]); bv = np.array([B[k] for k in ks])
            m = np.isfinite(av) & np.isfinite(bv)
            print(f"  {a:<14}vs {b:<14} n={m.sum():<5}{av[m].mean():.4f} -> {bv[m].mean():.4f}"
                  f"  p={stats.ttest_rel(av[m], bv[m]).pvalue:.2e}")
    print("  论文: NF 2.12 -> +energy 2.49 池化 p=0.19；p=.10 单档 0.32 vs 0.58 p=0.03")


# ---------------------------------------------------------------- Table 18
def t18():
    d = load("synth_reseed.json")["records"]
    W = d["weighted"] if isinstance(d, dict) else d
    LAD = ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]
    print("\n=== Table 18 合成阶梯配对检验（池化 n=48）===")
    meths = sorted({r["method"] for r in W})
    print("  可用方法:", meths)
    for b in meths:
        if b == "NF":
            continue
        A, B = {}, {}
        for r in W:
            k = (r["case"], r["seed"])
            (A if r["method"] == "NF" else B)[k] = r["A"] if "A" in r else r["A_angle_deg"]
        ks = sorted(set(A) & set(B))
        av = np.array([A[k] for k in ks], float)
        bv = np.array([B[k] for k in ks], float)
        m = np.isfinite(av) & np.isfinite(bv)
        print(f"  NF {av[m].mean():.3f} -> {b:<18} {bv[m].mean():.3f}  n={m.sum():<4} p={stats.ttest_rel(av[m], bv[m]).pvalue:.2e}")
    print("  论文: NF 2.12 -> +energy 2.49; 池化 p=0.19；p=.10 单档 0.32 vs 0.58 p=0.03")


# ---------------------------------------------------------------- §3.3
def sec33():
    print("\n=== §3.3 SNR_local 公式数值核对 ===")
    for p, N, M, snr in [(0.02, 4, 2, 20.0)]:
        Pact = 1 - (1 - p) ** N
        no_M = snr + 10 * np.log10(1 / Pact)
        with_M = snr + 10 * np.log10(M / Pact)
        print(f"  p={p} N={N} M={M} SNR={snr}: P(act)={Pact:.6f}")
        print(f"    公式(论文印刷版, 含 M): {with_M:.2f} dB")
        print(f"    公式(不含 M):           {no_M:.2f} dB   <- 论文正文数值 31.1 dB")


if __name__ == "__main__":
    table7()
    sec81()
    sec93()
    sec95()
    t18()
    sec33()
