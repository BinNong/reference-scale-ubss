#!/bin/bash
# 第五轮实验批处理（服务器侧运行）
cd /data/experiment/paper5_ubss_sadun/src || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export R5_NO_SDR=1

run() {
  name="$1"; shift
  echo "### $name  $(date +%H:%M:%S) ###"
  ../.venv/bin/python -u "$@" > ../logs/r5_$name.log 2>&1
  echo "### $name done rc=$? $(date +%H:%M:%S) ###"
}

run density_real  r5_density_baseline.py --real --real-seeds 4 --workers 8 \
                  --out ../results/r5_density_real.json
run alpha         r5_alpha_cell.py --seeds 10 --workers 10 \
                  --out ../results/r5_alpha_cell.json
run srcnum        r5_source_number.py --seeds 10 --workers 10 \
                  --out ../results/r5_source_number.json
run complexity    r5_complexity.py --out ../results/r5_complexity.json
run band          r5_band_floor.py --seeds 4 --workers 8 \
                  --out ../results/r5_band_floor.json
run weight        r5_unified_weight.py --seeds 6 --workers 8 \
                  --out ../results/r5_unified_weight.json
echo ALLDONE > ../logs/r5_done.flag
