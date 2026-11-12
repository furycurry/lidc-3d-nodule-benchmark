#!/usr/bin/env bash
# Run the full matrix (3 loss configurations x 5 seeds = 15 runs) in parallel, one run per GPU.
# Usage:  bash run_clean_matrix.sh [unet3d_paper|unetr] [NUM_GPUS]
# Runs are launched in batches of NUM_GPUS; each batch finishes before the next starts.
# Logs go to outputs/logs/, results to outputs/results/.
set -e

ARCH="${1:-unet3d_paper}"
NUM_GPUS="${2:-8}"
SEEDS=(42 123 456 789 2024)
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

# Job list: "<flags>|<seed>", ordered baseline, fixed boundary, adaptive boundary
JOBS=()
for s in "${SEEDS[@]}"; do JOBS+=("|$s"); done
for s in "${SEEDS[@]}"; do JOBS+=("--boundary|$s"); done
for s in "${SEEDS[@]}"; do JOBS+=("--boundary --adaptive-boundary|$s"); done

i=0
for job in "${JOBS[@]}"; do
    flags="${job%%|*}"
    seed="${job##*|}"
    if [[ "$flags" == *"--adaptive-boundary"* ]]; then name="boundary_adaptive"
    elif [[ "$flags" == *"--boundary"* ]]; then name="boundary"
    else name="baseline"; fi

    gpu=$(( i % NUM_GPUS ))
    log_file="$LOG_DIR/${name}_${ARCH}_seed${seed}.log"
    echo "[GPU $gpu] $name | $ARCH | seed $seed | log: $log_file"
    CUDA_VISIBLE_DEVICES=$gpu python train.py --seed "$seed" --architecture "$ARCH" $flags > "$log_file" 2>&1 &

    i=$(( i + 1 ))
    if (( i % NUM_GPUS == 0 )); then wait; fi
done
wait
echo "=== Done: ${#JOBS[@]} runs for $ARCH. Results in outputs/results/ ==="
