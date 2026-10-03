"""审稿复核（第二批）：Table 18 合成阶梯的配对检验 + §9.3 逐配置 p 值清单。

用法（服务器 src 目录）：../.venv/bin/python review_audit2.py
"""
import json
import numpy as np
from scipy import stats

R = "../results"
SR = json.load(open(f"{R}/synth_reseed.json"))["records"]


def pair_test(recs, a, b, group_key="case"):
    A, B = {}, {}
    for r in recs:
        k = (r[group_key], r["seed"])
        if r["method"] == a:
            A[k] = r["A"]
        if r["method"] == b:
            B[k] = r["A"]
    ks = sorted(set(A) & set(B))
    av = np.array([A[k] for k in ks], float)
    bv = np.array([B[k] for k in ks], float)
    m = np.isfinite(av) & np.isfinite(bv)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan, 0
    return av[m].mean(), bv[m].mean(), stats.ttest_rel(av[m], bv[m]).pvalue, int(m.sum())


print("=== Table 18 合成阶梯（pooled n=48，corrected methods）===")
for grp, pairs in [("weighted", [("NF", "NF-W"), ("NF", "NF-Wc"), ("NF", "base-W"), ("NF", "hard-opt")]),
                   ("purity", [("NF", "NF+e"), ("NF", "NF+e·ρ8"), ("NF+e", "NF+e·ρ8"),
                               ("NF+e", "NF+e·cos4")])]:
    for a, b in pairs:
        ma, mb, p, n = pair_test(SR[grp], a, b)
        print(f"  [{grp:<8}] {a:<10} {ma:6.3f} -> {b:<10} {mb:6.3f}  n={n:<4} p={p:.3e}")

print("\n=== Table 18 逐档（NF vs NF+energy）===")
for c in ["tf_p02", "tf_p05", "tf_p10", "tf_p20", "tf_p40", "tf_gauss"]:
    ma, mb, p, n = pair_test([r for r in SR["weighted"] if r["case"] == c], "NF", "NF-W")
    print(f"  {c:<10}{ma:6.3f} -> {mb:6.3f}  差={mb-ma:+.3f}  n={n}  p={p:.3e}")
print("  论文: p=.10 单档 0.32 vs 0.58 p=0.03；池化 p=0.19")

print("\n=== §9.3 逐配置 p（NF-SSP vs SCA-median，8 seeds）===")
RR = json.load(open(f"{R}/real_results.json"))["records"]
rows = []
for c in sorted({r["case"] for r in RR}):
    A = np.array([r["A_angle_deg"] for r in RR if r["case"] == c and r["method"] == "NF-SSP"])
    B = np.array([r["A_angle_deg"] for r in RR if r["case"] == c and r["method"] == "SCA-median"])
    p = stats.ttest_rel(A, B).pvalue
    rows.append((p, c, A.mean(), B.mean()))
for p, c, ma, mb in sorted(rows):
    print(f"  {'显著' if p < 0.05 else '不显著'} {c.split(':')[-1]:<10} NF={ma:6.2f} SCA-med={mb:6.2f} p={p:.3e}")
sig = [p for p, *_ in rows if p < 0.05]
print(f"  显著 {len(sig)}/15；其中最大 p = {max(sig):.3f}；最小 p = {min(sig):.2e}")
print("  论文 §9.3: 'fourteen of fifteen (p between 1.2e-5 and 0.05)'")
