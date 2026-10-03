"""理论推导的数值核对（纯标准库，无第三方依赖）。

本脚本不涉及任何项目实验，只核对论文 Section 5 中闭式公式的正确性：

  C1  纯噪声点的 |cos(Re(X), Im(X))| 分布  vs  理论 Beta(1/2,(M-1)/2)
  C2  硬阈值通过率 P(|cos| > c0) 的经验值与闭式值
  C3  污染比 kappa = (K_noise/K_signal) 的理论预测（与稀疏度 p 的关系）
  C4  能量门限对噪声点/信号点的平均门限值，及污染比的改善倍数
  C5  贪心去相关的带宽条件：抑制因子随"源间距/带宽"的变化

用法:
    python3 src/theory_checks.py
"""
from __future__ import annotations

import math
import random

# ----------------------------------------------------------------------
# 正则化不完全 Beta 函数 I_x(a,b)（Numerical Recipes 连分式实现）
# 用于精确计算 Beta 分布的生存函数，避免依赖 scipy。
# ----------------------------------------------------------------------


def _betacf(a: float, b: float, x: float, maxit: int = 300, eps: float = 1e-15) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a,b) ∈ [0,1]。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def p_abs_cos_exceed(M: int, c0: float) -> float:
    """P(|cos| > c0)：M 维两个独立各向同性随机方向的夹角余弦。

    |cos|^2 ~ Beta(1/2, (M-1)/2) ⇒ P = 1 - I_{c0^2}(1/2,(M-1)/2)
    """
    return 1.0 - betai(0.5, (M - 1) / 2.0, c0 * c0)


# ----------------------------------------------------------------------
# C1 / C2：噪声点余弦的分布与硬阈值通过率
# ----------------------------------------------------------------------


def check_cosine_distribution(n_samples: int = 200_000, seed: int = 0) -> None:
    print("=" * 78)
    print("[C1/C2] 纯噪声点 |cos(Re(X), Im(X))| 分布  vs  Beta(1/2,(M-1)/2)")
    print("=" * 78)
    rng = random.Random(seed)
    g = rng.gauss
    for M in (2, 3, 4):
        s2 = 0.0
        hits = {c: 0 for c in (0.90, 0.95, 0.98, 0.99)}
        for _ in range(n_samples):
            dot = 0.0
            nr2 = 0.0
            ni2 = 0.0
            for _d in range(M):
                a, b = g(0, 1), g(0, 1)
                dot += a * b
                nr2 += a * a
                ni2 += b * b
            cos = dot / math.sqrt(nr2 * ni2)
            c2 = cos * cos
            s2 += c2
            ac = abs(cos)
            for c in hits:
                if ac > c:
                    hits[c] += 1
        print(f"  M={M}:  E[cos^2] 经验={s2 / n_samples:.5f}   理论 1/M={1 / M:.5f}")
        for c in sorted(hits):
            emp = hits[c] / n_samples
            th = p_abs_cos_exceed(M, c)
            rel = abs(emp - th) / th * 100 if th > 0 else 0.0
            print(f"      c0={c:.2f}  P(通过): 经验={emp:.5f}  闭式={th:.5f}  "
                  f"相对偏差={rel:.1f}%")
        print()


# ----------------------------------------------------------------------
# C3：污染比的理论预测
# ----------------------------------------------------------------------


def check_contamination(n_sources: int = 4, c0: float = 0.98, M: int = 2) -> None:
    print("=" * 78)
    print(f"[C3] 硬阈值筛出的点中，噪声点与信号点的数量比 kappa")
    print(f"     理论: kappa = [(1-p)/(N p)] * P(|cos|>c0),  N={n_sources}, "
          f"c0={c0}, M={M}")
    print("=" * 78)
    pc = p_abs_cos_exceed(M, c0)
    print(f"  P(|cos| > {c0}) = {pc:.5f}   （纯噪声点通过硬阈值的概率）")
    print()
    print(f"  {'p':>6s} {'pi_s 单源点':>12s} {'pi_0 噪声点':>12s} "
          f"{'kappa':>8s} {'噪声占比':>10s}")
    for p in (0.02, 0.05, 0.10, 0.20, 0.40):
        pi_s = n_sources * p * (1 - p) ** (n_sources - 1)
        pi_0 = (1 - p) ** n_sources
        kappa = pi_0 * pc / pi_s
        frac = kappa / (1 + kappa)
        print(f"  {p:>6.2f} {pi_s:>12.5f} {pi_0:>12.5f} {kappa:>8.3f} {frac:>9.1%}")
    print()
    print("  含义：kappa 即「通过硬阈值的伪点数 / 真单源点数」。")
    print("        p=0.02 时 kappa>1，即通过筛选的点里噪声点占多数。")
    print()


# ----------------------------------------------------------------------
# C4：能量门限的抑制因子
# ----------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def check_energy_gate(tau_e: float = 10.0, sigma_e: float = 3.0,
                      M: int = 2, n_sources: int = 4, c0: float = 0.98) -> None:
    print("=" * 78)
    print("[C4] 能量门限的抑制因子与污染比的改善")
    print(f"     g_en = sigma((e/e_med - tau_e)/sigma_e),  tau_e={tau_e}, "
          f"sigma_e={sigma_e}")
    print("=" * 78)
    g_noise = _sigmoid((1.0 - tau_e) / sigma_e)
    print(f"  噪声点 (e ≈ e_med, 故 e/e_med ≈ 1):  g_en = sigma({(1 - tau_e) / sigma_e:.2f}) "
          f"= {g_noise:.5f}")
    for rho in (3.0, 10.0, 30.0, 100.0):
        print(f"  信号点 (局部 SNR e/e_med = {rho:>6.1f}):  "
              f"g_en = sigma({(rho - tau_e) / sigma_e:.2f}) = "
              f"{_sigmoid((rho - tau_e) / sigma_e):.5f}")
    print()
    print(f"  ⇒ 污染比改善倍数 ≈ 1/g_en(noise) = {1.0 / g_noise:.1f}x")
    print()
    pc = p_abs_cos_exceed(M, c0)
    print(f"  {'p':>6s} {'kappa(硬阈值)':>14s} {'kappa(加能量门限)':>18s} "
          f"{'噪声占比':>10s}")
    for p in (0.02, 0.05, 0.10, 0.20, 0.40):
        pi_s = n_sources * p * (1 - p) ** (n_sources - 1)
        pi_0 = (1 - p) ** n_sources
        k0 = pi_0 * pc / pi_s
        k1 = k0 * g_noise
        print(f"  {p:>6.2f} {k0:>14.3f} {k1:>18.3f} {k1 / (1 + k1):>9.1%}")
    print()


# ----------------------------------------------------------------------
# C5：贪心去相关的带宽条件
# ----------------------------------------------------------------------


def check_deflation_bandwidth() -> None:
    print("=" * 78)
    print("[C5] 贪心去相关的带宽条件")
    print("     suppress(theta) = 1 - exp(-2 sin^2(theta/2) / sigma_d^2)")
    print("     theta = 两源方向夹角；suppress 为已选列对另一源方向的权重保留率")
    print("=" * 78)
    sigmas = (0.05, 0.10, 0.20, 0.40)
    thetas = (5.0, 10.0, 20.0, 45.0, 90.0)
    print("          " + "".join(f"{('th=' + str(t) + 'deg'):>12s}" for t in thetas))
    for s in sigmas:
        row = f"  sigma_d={s:<4.2f}"
        for t in thetas:
            th = math.radians(t)
            d2 = 2.0 - 2.0 * math.cos(th)
            keep = 1.0 - math.exp(-d2 / (2 * s * s))
            row += f"{keep:>12.3f}"
        print(row)
    print()
    print("  含义：sigma_d 相对源间距过小（左侧各行）时，相邻源方向权重几乎被完全保留")
    print("        （去相关失效，贪心会重复选中同一方向）；sigma_d 相对源间距过大")
    print("        （右下）时，相邻源权重被一并抹掉（漏源）。可行区间要求")
    print("        源内角度散布 << sigma_d << sin(theta/2)，当 theta 较小时该区间消失。")
    print()


if __name__ == "__main__":
    check_cosine_distribution()
    check_contamination()
    check_energy_gate()
    check_deflation_bandwidth()
