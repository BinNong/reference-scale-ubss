#!/usr/bin/env python3
"""本脚本已停用——留着是指路，不是给机器跑的。

原来的三件事各自有了更好的落点：

  V1 闭式 r(p)=median(e)/ν vs 实测
     → `r15_recount.py` 的 Table 1（`t1()`），逐格对 `prop6_convention.json`，
       连 rel. error 那一列也一起复算。
  V2 盲噪声功率估计 ŝ² 的偏差与"下尾幂律斜率 == M"
     → 前半段由 `r15_recount.py` 的 Table 2（`t2()`）对 `r4_table2.json` 逐格复算。
       后半段**整个假设已作废**：§5.1(i) 明写不用小 x 幂律近似（拟合会把 ŝ² 抬高超 60%），
       所以"斜率是否等于 M"这个自诊断不再存在，脚本也就不该再报它。
  V3 闭式 τ(M,α) 下仅噪声点的通过率是否等于 α
     → `r3_tau_mc.py`（写 `results/r3_tau_mc.json`），每格 2e6 个复高斯噪声实现，
       `r15_recount.py` 的 Table 3 逐格复算 τ 并按二项标准误核对经验通过率。

另外，稿件 §7 承诺的"two further scripts that evaluate the regularised incomplete Beta and
chi-square functions directly and reproduce $p_c$ by Monte Carlo"指的是
`theory_checks.py` 与 `theory_validate_data.py`——这两个现在都能跑，与本脚本无关。

停用原因（2026-10-07）：`nfr.py` 的 `estimate_noise_power` 不再返回 `slope`/`r2`，
`reference_scale_ratio` 已改名 `mixture_median_ratio`，本脚本 import 即 ImportError；
而它描述的幂律回归算法 `nfr.py:148-149` 明确拒绝使用。留着一份跑不起来、且描述的是
已废弃算法的脚本，比没有更糟。

用法（确实想跑等效检查时）：

    python3 src/r15_recount.py            # V1/V2/V3 的逐格复算
    python3 src/r3_tau_mc.py              # V3 的记录生成
    python3 src/theory_checks.py          # Beta / chi-square 闭式核对（§7 承诺的两个之一）
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

EQUIV = [
    ("r15_recount.py", "V1/V2/V3：每张表逐格对回记录"),
    ("r3_tau_mc.py", "V3：τ 的蒙特卡洛校准记录"),
    ("theory_checks.py", "§7 承诺的 Beta / chi-square 闭式核对"),
]


def main() -> int:
    print("validate_nfr.py 已停用，见本文件 docstring。等效检查：")
    for script, why in EQUIV:
        print(f"  python3 src/{script:<20s} {why}")
    if "--run" in sys.argv:
        bad = 0
        for script, _ in EQUIV:
            r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                               cwd=ROOT, capture_output=True, text=True)
            tail = [l for l in (r.stdout or "").split("\n") if l.strip()]
            print(f"\n--- {script} (exit {r.returncode}) ---")
            for l in tail[-3:]:
                print("   ", l)
            bad += r.returncode not in (0, 1)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
