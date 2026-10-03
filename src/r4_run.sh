#!/bin/bash
# R4 批处理：在服务器 src 目录下运行（用 setsid 启动，避免与 ssh 会话同生共死）
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "### [1] guard errors + spread ablation ###"
../.venv/bin/python -u r4_guard_errors.py --out ../results/r4_guard_errors.json 2>&1
echo
echo "### [2] robust floor (synthetic + real) ###"
../.venv/bin/python -u r4_robust_floor.py --seeds 8 --real --real-seeds 4 --out ../results/r4_robust_floor.json 2>&1
echo
echo "### DONE ###"
