#!/bin/bash
# Reproduces the TipAD results: train -> eval (seeds 2023 & 2024) -> average
# -> compare against the official baselines. Prints the Avg.RANK table.

set -e
trap 'kill 0' INT TERM      # stop every descendant, not just this script
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
