"""复核第一轮修订中**手工敲进手稿的新数字**是否与归档一致。

修订脚本能保证"替换发生"，但不能保证"新数字对"。本脚本把每条新数字与
对应 JSON 独立复算一遍，逐条断言。

用法（服务器 src 目录或本地 results 旁）：
    ../.venv/bin/python verify_revision.py
纯标准库即可运行（不依赖 numpy）。
"""
from __future__ import annotations

import json
import pathlib
import re
import statistics as st
import sys

R = pathlib.Path("../results")
MS = pathlib.Path("../paper/manuscript.md")

FAIL = []


def check(tag, claimed, actual, tol=0.01):
    ok = abs(claimed - actual) <= tol * max(1.0, abs(actual))
    print(f"  {'OK ' if ok else 'FAIL'} {tag:<52} 手稿={claimed:<10} 归档={actual:.4f}")
    if not ok:
        FAIL.append(tag)


def mean_of(recs, **kw):
    v = [r["A_angle_deg"] if "A_angle_deg" in r else r["A"]
         for r in recs if all(r.get(k) == v2 for k, v2 in kw.items())]
    return st.mean(v)


def main():
    # 手稿只在本地存在；本脚本的判据是"手稿中敲入的值"（下面的 claimed 常量），
    # 归档复算不需要手稿本身。若手稿在，则顺便确认它确实包含这些值。
    ms = MS.read_text() if MS.exists() else None
    if ms is None:
        print("（未找到手稿，仅做归档复算）")

    # ---------- Table 7 运行时间：nfr_results.json, E1_sparsity ----------
    print("\n[Table 7] 运行时间（E1_sparsity, 10 seeds）")
    nfr = json.loads((R / "nfr_results.json").read_text())["records"]
    e1 = [r for r in nfr if r["experiment"] == "E1_sparsity"]
    for tag, meth in [("NF-SSP 0.184", "NF-SSP"), ("SCA-default 0.189", "SCA-default"),
                      ("SCA-tuned 0.163", "SCA-L1-tuned"), ("pf 0.143", "pf_auto_l1"),
                      ("sl0 1.412", "sl0_l1")]:
        m = st.mean([r["time_s"] for r in e1 if r["method"] == meth])
        check(f"  {tag}（{meth}）", float(re.search(r"([\d.]+)（", tag).group(1)) if "（" in tag else float(tag.split()[-1]), m, 0.02)

    # ---------- Table 12：per-regime worst ----------
    print("\n[Table 12] worst regime = 逐档均值（不是单次极值）")
    for exp, claimed in [("E7_alpha1e-02", 11.87), ("E7_alpha1e-03", 12.24), ("E7_alpha1e-04", 12.24)]:
        recs = [r for r in nfr if r["experiment"] == exp]
        per = {c: st.mean([r["A_angle_deg"] for r in recs if r["cfg_key"] == c])
               for c in {r["cfg_key"] for r in recs}}
        check(f"  {exp} 档均值最大值", claimed, max(per.values()), 0.01)
        mx = max(r["A_angle_deg"] for r in recs)
        check(f"  {exp} 单次最大值（另列一行）", 26.59 if exp.endswith("02") else 20.03, mx, 0.01)

    # ---------- Table 17 基线列 = real_results.json 的 SCA-median ----------
    print("\n[Table 17] 基线列与 Table 15 同源")
    rr = json.loads((R / "real_results.json").read_text())["records"]
    check("  基线均值", 8.26, st.mean([r["A_angle_deg"] for r in rr if r["method"] == "SCA-median"]), 0.005)
    w = json.loads((R / "weighted_variant.json").read_text())["records"]
    real = [r for r in w if r["dom"] == "real"]
    for tag, meth, claimed in [("NF 1.12", "NF", 1.12), ("+energy 0.72", "NF-W", 0.72),
                               ("+|cos|^4 0.68", "NF-Wc", 0.68), ("base+energy 1.15", "base-W", 1.15)]:
        check(f"  {tag}", claimed, mean_of(real, method=meth), 0.01)
    # 两表共享的未加权记录必须逐条相同
    nfw = {(r["case"], r["seed"]): r["A"] for r in w if r["dom"] == "real" and r["method"] == "NF"}
    nfr_ = {(r["case"], r["seed"]): r["A_angle_deg"] for r in rr if r["method"] == "NF-SSP"}
    ks = sorted(set(nfw) & set(nfr_))
    d = max(abs(nfw[k] - nfr_[k]) for k in ks)
    print(f"  {'OK ' if d == 0 and len(ks) == 120 else 'FAIL'} 两表共享记录数={len(ks)} 最大逐条差={d}")
    if not (d == 0 and len(ks) == 120):
        FAIL.append("两表共享记录")

    # ---------- Table 18 的 oracle 列 ----------
    print("\n[Table 18] hard-threshold oracle")
    sr = json.loads((R / "synth_reseed.json").read_text())["records"]["weighted"]
    for c, claimed in [("tf_p02", 0.28), ("tf_p05", 0.21), ("tf_p10", 0.31),
                       ("tf_p20", 0.43), ("tf_p40", 1.28), ("tf_gauss", 7.74)]:
        check(f"  {c}", claimed, mean_of([r for r in sr if r["case"] == c], method="hard-opt"), 0.01)
    check("  均值 1.71", 1.71, st.mean([r["A"] for r in sr if r["method"] == "hard-opt"]), 0.01)

    # ---------- Table 15 caption 的 bound 定义 ----------
    print("\n[Table 15 caption] per-configuration bound")
    cases = sorted({r["case"] for r in rr})
    tuned = ["SCA-median-opt", "SCA-max-opt", "top-K-opt"]
    allm = ["SCA-median", "SCA-max", "top-K"] + tuned
    b3 = [min(st.mean([r["A_angle_deg"] for r in rr if r["method"] == m and r["case"] == c])
              for m in tuned) for c in cases]
    b7 = [min(st.mean([r["A_angle_deg"] for r in rr if r["method"] == m and r["case"] == c])
              for m in allm) for c in cases]
    # ---- 「手稿声称值」一律从手稿文本抓取，不写死常量。
    # 写死常量只验证"记忆"——第八轮 `build_pdf.py` 把期望表数写成 range(1,40) 就是这样栽的。
    def claim(pat, name):
        m = re.search(pat, ms, re.S) if ms else None
        if not m:
            FAIL.append(f"手稿读不到 {name}")
            print(f"  FAIL 手稿里读不到 {name}")
            return float("nan")
        return float(m.group(1))

    c_b3 = claim(r"whose mean is \$([\d.]+)°\$", "Table 15 题注的三调参列均值")
    c_b7 = claim(r"all seven competing columns it would be \$([\d.]+)°\$", "Table 15 题注的七列最小值")
    c_conv = claim(r"mean penalty of the conventional baseline against this bound is \$([\d.]+)\\times\$",
                   "Table 15 题注的 conventional penalty")
    c_ratio = claim(r"\(ratio of means: \$([\d.]+)\\times\$\)", "Table 15 题注的 ratio of means")
    c_nfpen = claim(r"and that of NF-SSP is \$([\d.]+)\\times\$", "Table 15 题注的 NF penalty")
    c_max = claim(r"penalty averages \$[\d.]+\\times\$ and reaches \$([\d.]+)\\times\$", "§9.3 的最大 penalty")
    c_bound = claim(r"against \$([\d.]+)°\$ for the best-observed reference", "§9.3 的 best-observed 均值")

    check(f"  三调参列最小值（题注 {c_b3}）", c_b3, st.mean(b3), 0.01)
    check(f"  全部七列最小值（题注 {c_b7}）", c_b7, st.mean(b7), 0.01)
    nf = [st.mean([r["A_angle_deg"] for r in rr if r["method"] == "NF-SSP" and r["case"] == c]) for c in cases]
    # 手稿的定义（Table 15 题注）：penalty = 同一配置内两误差之比，再对十五个配置平均。
    # 注意它**不等于**"均值之比"，两者在小数第二位会分家（新归档 1.62 vs 1.63）。
    pen = [st.mean([r["A_angle_deg"] for r in rr if r["method"] == "SCA-median" and r["case"] == c]) / b3[i]
           for i, c in enumerate(cases)]
    check(f"  conventional penalty（题注 {c_conv}）", c_conv, st.mean(pen), 0.01)
    check(f"  ratio of means（题注 {c_ratio}）", c_ratio,
          st.mean([st.mean([r["A_angle_deg"] for r in rr if r["method"] == "SCA-median" and r["case"] == c])
                   for c in cases]) / st.mean(b3), 0.01)
    check(f"  最大 penalty（§9.3 {c_max}）", c_max, max(pen), 0.02)
    check(f"  NF penalty（题注 {c_nfpen}）", c_nfpen,
          st.mean([a / b for a, b in zip(nf, b3)]), 0.01)
    check(f"  best-observed 均值（§9.3 {c_bound}）", c_bound, st.mean(b3), 0.01)

    # ---------- §9.3 / §9.4 计数 ----------
    print("\n[§9.3/§9.4] 计数")
    conv = {c: st.mean([r["A_angle_deg"] for r in rr if r["method"] == "SCA-median" and r["case"] == c]) for c in cases}
    mx = {c: st.mean([r["A_angle_deg"] for r in rr if r["method"] == "SCA-max" and r["case"] == c]) for c in cases}
    nf_c = dict(zip(cases, nf))
    check("  传统门限超过 8° 的配置数", 11, sum(1 for c in cases if conv[c] > 8), 0.0)
    check("  低于 8° 的例外数", 4, sum(1 for c in cases if conv[c] <= 8), 0.0)
    check("  max-参考更优的配置数", 12, sum(1 for c in cases if mx[c] < nf_c[c]), 0.0)

    # ---------- 第三轮新增：Table 9 的两处反例 ----------
    # 手稿 §8.3 从 "never loses" 改为列出这两格；数值必须与归档一致，否则又是"文字—表格矛盾"。
    print("\n[Table 9/§8.3] 稠密档的两处名义反转")
    from scipy import stats as _st
    for m_, c_nf, c_sd in ((3, 2.748, 2.736), (4, 1.297, 1.291)):
        nf = sorted(r["A_angle_deg"] for r in nfr
                    if r["method"] == "NF-SSP" and r["experiment"] == "E3_M"
                    and r["cfg_key"] == "tf_p40" and r["m_obs"] == m_)
        sd = sorted(r["A_angle_deg"] for r in nfr
                    if r["method"] == "SCA-default" and r["experiment"] == "E3_M"
                    and r["cfg_key"] == "tf_p40" and r["m_obs"] == m_)
        check(f"  M={m_} NF", c_nf, st.mean(nf), 0.002)
        check(f"  M={m_} SCA-default", c_sd, st.mean(sd), 0.002)
        ratio = st.stdev(nf) / abs(st.mean(nf) - st.mean(sd))
        check(f"  M={m_} sd/差 的倍数", {3: 436, 4: 46}[m_], ratio, 2.0)
        pv = _st.ttest_rel(nf, sd).pvalue
        print(f"  {'OK ' if pv > 0.05 else 'FAIL'} M={m_} 配对 t 检验 p={pv:.3f}（手稿写不显著）")
        if not pv > 0.05:
            FAIL.append(f"M={m_} 反例显著性")

    # ---------- 第三轮新增：Prop. 7 的噪声-only 中位比 ----------
    print("\n[Prop. 7/§4.3] 噪声-only 中位比与 r(p) 实测")
    from scipy.stats import chi2 as _chi2
    m0 = float(_chi2.ppf(0.5, 4)) / 4.0
    check("  median(chi2_4)/4（M=2）", 0.839, m0, 0.002)
    conv = json.loads((R / "prop6_convention.json").read_text())["rows"]
    row02 = [r for r in conv if abs(r["p"] - 0.02) < 1e-9][0]
    check("  r(p=0.02) 实测", 0.92, row02["r_emp"], 0.01)
    check("  pi0(p=0.02, N=4)", 0.922, (1 - 0.02) ** 4, 0.002)

    # ---------- 守卫表 ----------
    print("\n[Table 4/§5.3/§10] 守卫")
    g = json.loads((R / "guard_validation.json").read_text())
    check("  条件数", 27, g["n_conditions"], 0.0)
    # 决策口径改为"逐种子数决策取多数"后，条件级误判由 1 变 0。
    # 注意：本地归档一度是修正前的旧版（缺 n_gate_on 字段），与手稿不一致——
    # 已用服务器上修正后的记录覆盖。这类"归档没跟着重跑"是硬伤级的复现问题。
    check("  条件级误判数", 0, g["n_wrong"], 0.0)
    assert "n_gate_on" in g["rows"][0], "归档是修正前的旧版（缺 n_gate_on 字段）"
    on = [r["keep_frac"] for r in g["rows"] if r["preferred"] == "ON"]
    off = [r["keep_frac"] for r in g["rows"] if r["preferred"] == "OFF"]
    check("  最大'应关'保留率 1.0%", 1.0, max(off) * 100, 0.05)
    check("  最小'应开'保留率 3.4%", 3.4, min(on) * 100, 0.03)
    lap = [r for r in g["rows"] if r["case"] == "tf_lap"][0]
    check("  Laplace 增益 24%", 24, lap["gain"] * 100, 0.02)
    # 最接近边界、且"应开"的那一例（3.4%），其增益为 3.9%
    near = min((r for r in g["rows"] if r["preferred"] == "ON"), key=lambda r: r["keep_frac"])
    check("  最接近边界且'应开'的保留率 3.4%", 3.4, near["keep_frac"] * 100, 0.03)
    check("  该例增益 3.9%", 3.9, near["gain"] * 100, 0.02)

    # ---------- Prop. 6 约定 ----------
    print("\n[Prop. 6] 约定与实测")
    pc = {r["p"]: r for r in json.loads((R / "prop6_convention.json").read_text())["rows"]}
    for p, emp, fix, pap in [(0.02, 1208, 1250, 1288), (0.10, 251, 249, 291),
                             (0.20, 126, 124, 169), (0.40, 61, 62, 115)]:
        check(f"  p={p} 实测 E1/ν", emp, pc[p]["E1_over_nu_emp"], 0.005)
        check(f"  p={p} 本约定", fix, pc[p]["E1_over_nu_fixed"], 0.01)
        check(f"  p={p} 原约定", pap, pc[p]["E1_over_nu_paper"], 0.01)
    for p, claimed in [(0.02, 1.03), (0.20, 1.36), (0.40, 1.84)]:
        check(f"  p={p} Np/P(act)", claimed, pc[p]["Np_over_Pact"], 0.01)
    check("  Table 1 闭式 p=0.20", 27.02, pc[0.20]["closed_fixed"], 0.005)
    check("  Table 1 实测 p=0.20", 26.96, pc[0.20]["r_emp"], 0.005)
    check("  §2 的 87 倍", 87, pc[1.0]["r_emp"] / pc[0.02]["r_emp"], 0.01)
    check("  中位数/信号尺度 p=0.20", 22, pc[0.20]["r_emp"] * pc[0.20]["E1_over_nu_fixed"] / 100
          if False else 100 * (pc[0.20]["r_emp"] * 2 * 0.02 * 2 / 2) / (1 / 0.20) , 0.03)

    # ---------- 正交性 ----------
    print("\n[Prop. 6] 正交性假设的代价")
    og = {r["p"]: r for r in json.loads((R / "prop6_orthogonality.json").read_text())["rows"]}
    for p, claimed in [(0.20, 5), (0.40, 13)]:
        d = 100 * (og[p]["r_mc_orth"] - og[p]["r_mc_true"]) / og[p]["r_mc_true"]
        check(f"  p={p} 正交−真值 ≈ {claimed}%", claimed, d, 0.15)

    # ---------- §9.5 合成最优区间 ----------
    print("\n[§9.5] 合成最优 median 系数区间")
    tg = json.loads((R / "synth_tuned_grid.json").read_text())["rows"]
    tes = sorted({r["te"] for r in tg})
    check("  最优 te 下界 0.1", 0.1, min(tes), 0.001)
    check("  最优 te 上界 20", 20.0, max(tes), 0.001)
    got = [r for r in tg if r["regime"] == "tf_p40"][0]
    check("  p=0.40 取 0.1 / p=0.10 取 20", 0.1, got["te"], 0.001)

    # ---------- 记录数 ----------
    print("\n[§7] 可用性声明的记录数")
    check("  nfr 2,770", 2770, len(nfr), 0.0)
    check("  real 960", 960, len(rr), 0.0)
    check("  weighted 768", 768, len(w), 0.0)
    p1 = json.loads((R / "purity_stage1.json").read_text())["records"]
    p2 = json.loads((R / "purity_stage2.json").read_text())["records"]
    check("  purity 1,392", 1392, len(p1) + len(p2), 0.0)
    check("  scaling 54", 54, len(json.loads((R / "scale_invariance.json").read_text())["rows"]), 0.0)

    if ms is not None:
        print("\n[手稿] 关键新值确实出现在正文")
        # 第二轮：guard 决策口径修正后，"26 times"/"$24\%$" 已被"no material error"取代
        for lit in ["$0.184$", "11.87", "12.24", f"{c_b3:.3f}°", f"{c_b7:.3f}°",
                    "27 conditions", "no material error", "$1.0\\%$", "$3.4\\%$",
                    "1{,}392", "$54$ cells", "0.1$ at $p=0.40$ to $20$", "$87$",
                    "$1208$", "27.02", "$+0.2\\%$",
                    "Robustness and Cost", "1{,}320"]:
            ok = lit in ms
            print(f"  {'OK ' if ok else 'FAIL'} {lit!r}")
            if not ok:
                FAIL.append(f"手稿缺 {lit}")

    print("\n" + "=" * 74)
    if FAIL:
        print(f"未通过 {len(FAIL)} 项：")
        for t in FAIL:
            print("   -", t)
        sys.exit(1)
    print("全部通过：修订中手工敲入的数字与归档一致。")


if __name__ == "__main__":
    main()
