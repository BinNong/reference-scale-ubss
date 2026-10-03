#!/bin/bash
# 第二轮审稿修改：实验批量运行脚本（服务器上执行）
# 用法： cd /data/experiment/paper5_ubss_sadun/src && nohup bash r2_run.sh > ../logs/r2_run.log 2>&1 &
cd "$(dirname "$0")"
# 可用 PYTHON=/path/to/python 覆盖；默认用当前 PATH 里的 python（不要硬编码 venv 路径，
# 否则别人 clone 下来这条命令直接失败）
P=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${W:-10}

run () {
  echo "===== $* ====="
  $P -u "$@" || echo "!!!! 失败: $*"
  echo
}

run r2_ssp_quality.py --seeds 8 --workers "$W" --out ../results/r2_ssp_quality.json
run r2_ssp_quality_real.py --seeds 4 --workers "$W" --out ../results/r2_ssp_quality_real.json
run r2_sweeps.py --part qmax  --seeds 8  --workers "$W" --out ../results/r2_qmax_sweep.json
run r2_sweeps.py --part angle --seeds 8  --workers "$W" --out ../results/r2_angle_sweep.json
run r2_sweeps.py --part eta   --seeds 10 --workers "$W" --out ../results/r2_eta_sweep.json
run r2_noise_family.py --seeds 8 --workers "$W" --out ../results/r2_noise_family.json
run r2_heatmap.py --seeds 5 --workers "$W" --out ../results/r2_heatmap_p_snr.json
run r2_runtime.py --seeds 20 --out ../results/r2_runtime.json
run r2_stats.py --out ../results/r2_stats.json
echo "===== 全部完成 ====="
