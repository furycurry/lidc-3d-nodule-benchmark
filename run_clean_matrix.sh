#!/usr/bin/env bash
set -e

LOG_DIR="/workspace/outputs/logs"
mkdir -p "$LOG_DIR"

ARCH="unet3d_paper"

run_task() {
    local gpu=$1
    local flags=$2
    local seed=$3
    
    local name="baseline"
    if [[ "$flags" == *"--adaptive-boundary"* ]]; then
        name="boundary_adaptive"
    elif [[ "$flags" == *"--boundary"* ]]; then
        name="boundary"
    fi
    
    local log_file="$LOG_DIR/${name}_${ARCH}_seed${seed}.log"
    echo "[GPU $gpu] Launching $name (Seed $seed)... Log: $log_file"
    
    CUDA_VISIBLE_DEVICES=$gpu python /workspace/train.py --seed "$seed" --architecture "$ARCH" $flags > "$log_file" 2>&1 &
}

echo "=== BATCH 1: Launching Runs 1-8 on GPUs 0-7 ==="
run_task 0 "" 42
run_task 1 "" 123
run_task 2 "" 456
run_task 3 "" 789
run_task 4 "" 2024
run_task 5 "--boundary" 42
run_task 6 "--boundary" 123
run_task 7 "--boundary" 456

echo "Waiting for Batch 1 processes..."
wait
echo "Batch 1 complete."

echo "=== BATCH 2: Launching Runs 9-15 on GPUs 0-6 ==="
run_task 0 "--boundary" 789
run_task 1 "--boundary" 2024
run_task 2 "--boundary --adaptive-boundary" 42
run_task 3 "--boundary --adaptive-boundary" 123
run_task 4 "--boundary --adaptive-boundary" 456
run_task 5 "--boundary --adaptive-boundary" 789
run_task 6 "--boundary --adaptive-boundary" 2024

echo "Waiting for Batch 2 processes..."
wait
echo "=== All 15 Benchmark Runs Completed Successfully ==="
