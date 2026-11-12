#!/usr/bin/env bash
# Run the full benchmark matrix (3 loss configurations x 5 seeds) for one architecture, sequentially.
# Usage:  bash run_multiseed.sh [unet3d_paper|unetr]
# For the multi-GPU parallel version, see run_clean_matrix.sh.
set -e

ARCH="${1:-unet3d_paper}"
SEEDS=(42 123 456 789 2024)

echo "=== Benchmark matrix: architecture=${ARCH}, seeds=${SEEDS[*]} ==="

for SEED in "${SEEDS[@]}"; do
    echo "[RUN] baseline | ${ARCH} | seed ${SEED}"
    python train.py --seed "${SEED}" --architecture "${ARCH}"
done

for SEED in "${SEEDS[@]}"; do
    echo "[RUN] boundary (fixed) | ${ARCH} | seed ${SEED}"
    python train.py --seed "${SEED}" --architecture "${ARCH}" --boundary
done

for SEED in "${SEEDS[@]}"; do
    echo "[RUN] boundary (adaptive) | ${ARCH} | seed ${SEED}"
    python train.py --seed "${SEED}" --architecture "${ARCH}" --boundary --adaptive-boundary
done

echo "=== Done. Results in outputs/results/ ==="
