"""重跑合成对照：把两个加权脚本的合成阶梯从"单次实现"修正为多个独立问题。

背景
----
`weighted_variant.py` 与 `purity_weight.py` 原先都写成

    recs += eval_synth(a.seeds)

把 `--seeds 8` 当成**单个种子值**传进 eval_synth，于是只评测了
seed=1000+8=1008 这**一个**问题实例（日志里却打着 `seeds=8`）。
主实验 `experiments_nfr.py` 的约定是 `for s in range(n_seeds)`，
每个种子生成一个独立问题，本脚本按该约定重算。

影响范围
--------
· 手稿 Table 18（合成阶梯上加权 vs 硬门限）——受影响，需用本脚本结果替换；
· 纯度加权实验的合成表 —— 同样受影响；
· 真实语音部分 —— **不受影响**，其 `eval_real` 一直是 `for s in range(seeds)`。

用法: ../.venv/bin/python rerun_synth.py --seeds 8 --out ../results/synth_reseed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="../results/synth_reseed.json")
    ap.add_argument("--modules", default="weighted,purity")
    a = ap.parse_args()

    want = [m.strip() for m in a.modules.split(",") if m.strip()]
    out: dict[str, list] = {}

    for which in want:
        t0 = time.perf_counter()
        if which == "weighted":
            import weighted_variant as WV
            recs = WV.eval_synth(a.seeds)
        elif which == "purity":
            import purity_weight as PW
            recs = PW.eval_synth(a.seeds)
        else:
            raise SystemExit(f"未知模块 {which}")
        out[which] = recs
        print(f"[{which}] {len(recs)} 条记录，用时 {time.perf_counter()-t0:.1f}s", flush=True)

    for which, recs in out.items():
        met = sorted({r["method"] for r in recs})
        cases = sorted({r["case"] for r in recs})
        print(f"\n=== {which} :: 合成 A 误差（{a.seeds} seeds 均值）===")
        print(f"{'case':<12s}" + "".join(f"{m:>11s}" for m in met))
        for c in cases:
            row = f"{c:<12s}"
            for m in met:
                v = [r["A"] for r in recs
                     if r["case"] == c and r["method"] == m and np.isfinite(r["A"])]
                row += f"{np.mean(v):>11.2f}" if v else f"{'—':>11s}"
            print(row)
        row = f"{'MEAN':<12s}"
        for m in met:
            v = [r["A"] for r in recs if r["method"] == m and np.isfinite(r["A"])]
            row += f"{np.mean(v):>11.2f}" if v else f"{'—':>11s}"
        print(row)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"seeds": a.seeds, "records": out}, f)
    print(f"\n已写出 {a.out}")


if __name__ == "__main__":
    main()
