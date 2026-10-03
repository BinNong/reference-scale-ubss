"""诊断：DBSCAN/OPTICS 在方向集上到底聚出了几个簇？"""
import sys
sys.path.insert(0, ".")
import numpy as np
import data as D
from sklearn.cluster import DBSCAN
import r5_density_baseline as RB

for key in ("tf_p02", "tf_p20"):
    p = D.make_problem(2, 4, 33, 64, key, 20.0, seed=1000)
    X = p["X_tf"]
    mk = RB.collin_mask(X)
    U = RB.ssp_directions(X, mk)
    print(f"\n{key}: 准入 {mk.mean():.3f}  方向数 {U.shape[1]}")
    for eps in (0.02, 0.05, 0.1, 0.25):
        lab = DBSCAN(eps=eps, min_samples=10).fit_predict(U.T)
        u, c = np.unique(lab[lab >= 0], return_counts=True)
        big = sorted(c, reverse=True)[:6]
        print(f"  eps={eps:<5} 簇数 {len(u):>3}  最大簇 {big}  噪声点 {int((lab<0).sum())}")
    # 真实方向间的距离尺度
    d = np.linalg.norm(U[:, None, :] - U[:, :, None], axis=0)
    np.fill_diagonal(d, np.inf)
    print(f"  最近邻距离 中位 {np.median(d.min(axis=1)):.4f}  最小 {d.min():.4f}"
          f"  (真值列间夹角 "
          f"{np.degrees(np.arccos(np.abs(np.asarray(p['A']).T @ np.asarray(p['A']))))[0, 1]:.1f}°)")
