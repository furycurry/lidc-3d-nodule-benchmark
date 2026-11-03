#!/usr/bin/env bash
set -e

SEEDS=(42 43 44 45 46)

echo "=========================================="
echo " Starting 3D U-Net Benchmark Execution"
echo "=========================================="

# 1. Baseline Run (DiceFocal Only)
for SEED in "${SEEDS[@]}"; do
    echo "[RUNNING] Baseline | Seed: ${SEED}"
    python train.py --seed "${SEED}" --architecture unet3d_paper
done

# 2. Fixed-Schedule Boundary Loss
for SEED in "${SEEDS[@]}"; do
    echo "[RUNNING] Fixed Boundary | Seed: ${SEED}"
    python train.py --seed "${SEED}" --boundary --architecture unet3d_paper
done

# 3. Adaptive Boundary Loss (PCG-BW)
for SEED in "${SEEDS[@]}"; do
    echo "[RUNNING] Adaptive Boundary | Seed: ${SEED}"
    python train.py --seed "${SEED}" --boundary --adaptive-boundary --architecture unet3d_paper
done

echo "=========================================="
echo " Benchmark Execution Complete!"
echo " Results saved to outputs/results/"
echo "=========================================="