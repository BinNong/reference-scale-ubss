#!/usr/bin/env python3
"""只重新生成指定的图（避免整脚本重跑、连带改动别的图）。

用法（服务器 src 目录下）:
    ../.venv/bin/python regen_fig.py architecture
    ../.venv/bin/python regen_fig.py guard sparsity snr
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures_nfr as M   # noqa: E402

FIG = {
    "scale": M.fig_scale,
    "architecture": M.fig_architecture,
    "sparsity": M.fig_sparsity,
    "snr": M.fig_snr,
    "nsources": M.fig_nsources,
    "guard": M.fig_guard,
}
NEEDS_JSON = {"sparsity", "snr", "nsources"}


def main() -> None:
    names = sys.argv[1:] or ["architecture"]
    out = os.environ.get("FIGOUT", "../results/figures_nfr")
    data = None
    for n in names:
        if n not in FIG:
            print(f"未知图名 {n}；可选 {sorted(FIG)}")
            sys.exit(2)
        if n in NEEDS_JSON and data is None:
            data = M.load("../results/nfr_results.json")
        FIG[n](data, out) if n in NEEDS_JSON else FIG[n](out)
    print("已重新生成:", ", ".join(names), "->", os.path.abspath(out))


if __name__ == "__main__":
    main()
