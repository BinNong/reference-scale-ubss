"""诊断：为什么 r5_complex 的实值对照列与 Table 6 差 5 倍？"""
import sys
sys.path.insert(0, ".")
import numpy as np
import data as D
import metrics as MT
from baselines import kmeans_sphere, ssp_directions
from nfr import nfr_mask, estimate_noise_power, point_energies, threshold_ratio
import r5_complex as R

print(f"{'seed':>6}{'论文路径':>10}{'本脚本':>10}{'论文路径(rng=seed)':>18}"
      f"{'保留率':>9}{'cplx机器':>10}")
for s in (1000, 1001, 1002, 1003):
    p = D.make_problem(2, 4, 33, 64, "tf_p02", 20.0, seed=s)
    X, A = p["X_tf"], p["A"]
    mk, dg = nfr_mask(X, 2, alpha=1e-4)
    # 主线（experiments_nfr 的写法）：rng 用 seed 推导
    a1 = kmeans_sphere(ssp_directions(X, mk), 4, np.random.default_rng(s - 1000))
    e1 = MT.mixing_matrix_angle_error_deg(A, a1)
    # 本脚本的写法
    out = R.eval_one("real", "tf_p02", s, s - 1000)
    e2 = out["A_rp_nf"]
    print(f"{s:>6}{e1:>10.3f}{e2:>10.3f}{'':>18}{float(mk.mean()):>9.4f}"
          f"{out['A_cx_nf']:>10.3f}")
    # 诊断量
    E = point_energies(X)
    est = estimate_noise_power(E, 2)
    print(f"        s2_hat/s2={est['s2']/(np.mean(np.abs(p['noise_tf'])**2)):6.3f}  tau={threshold_ratio(2,1e-4):.4f}  "
          f"spread={est.get('spread', float('nan')):.3f}  gate_on={dg['gate_on']}  "
          f"n_keep={int(dg['n_keep'])} n_base={int(dg['n_base'])}")
