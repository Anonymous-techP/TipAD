#!/bin/bash
# Reproduces the TipAD results: train -> eval (seeds 2023 & 2024) -> average
# -> compare against the official baselines. Prints the Avg.RANK table.
#
#   bash run_all.sh        # 4 parallel workers (default)
#   bash run_all.sh 2      # fewer workers, for machines with less RAM
#   bash run_all.sh 1      # serial
#
# Results do NOT depend on the number of workers: the predictor re-seeds before
# every series, so any setting produces identical numbers. Workers only change
# how long the run takes. Finished series are cached in eval_arrays_s<seed>/,
# so an interrupted run can simply be restarted.
set -e
cd "$(dirname "$0")/experiments"

WORKERS="${1:-4}"

echo "=== preflight ==="
python3 launch_eval.py --check "$WORKERS"

echo "=== train ==="
python3 run_tipad.py --phase train

for SEED in 2023 2024; do
  echo "=== eval seed=$SEED ==="
  python3 launch_eval.py --seed "$SEED" --procs "$WORKERS"
  python3 run_tipad.py --phase merge --seed "$SEED"
done

echo "=== averaging seeds 2023 and 2024 ==="
python3 average_seeds.py 2023 2024

echo "=== compare_baselines ==="
python3 compare_baselines.py
