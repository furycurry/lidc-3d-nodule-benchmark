#!/usr/bin/env python3
"""
evaluate_test_set.py

One-look test-set evaluation for the LIDC-IDRI 3D U-Net benchmark.

Runs all 15 checkpoints (best_dice_unet3d.pt per config/seed) against the
held-out 434-patch test partition.  Pure inference — no training, no SDF.

Saves:
  - One JSON per (config, seed):  outputs/test_results/test_{experiment_name}.json
  - Consolidated summary:         outputs/test_results/test_summary_all.json

Usage (on cluster, from repo root):
    python src/evaluate_test_set.py

Dependencies: torch, monai, numpy, config.py
"""

import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import (
    DiceMetric, HausdorffDistanceMetric,
    ConfusionMatrixMetric, SurfaceDistanceMetric,
)
from monai.losses import DiceFocalLoss

import config
from lidc_dataset import LIDCDataset
from unet3d_paper import UNet3DPaper

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CHECKPOINT_BASE = config.OUTPUT_DIR / "checkpoints"
TEST_RESULTS_DIR = config.OUTPUT_DIR / "test_results"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = config.BASE_DIR / "train_val_test_split.json"

# ------------------------------------------------------------------
# Experimental design constants
# ------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 2024]

CONFIGS = {
    "baseline":        "baseline_unet3d_paper",
    "boundary_fixed":  "boundary_unet3d_paper",
    "boundary_adaptive": "boundary_adaptive_unet3d_paper",
}

# ------------------------------------------------------------------
# Size bucket — must match train.py exactly
# ------------------------------------------------------------------
def size_bucket(voxel_count):
    if voxel_count < 100:
        return "small"
    elif voxel_count < 500:
        return "medium"
    else:
        return "large"


# ------------------------------------------------------------------
# Distance-metric aggregation — mirrors train.py's _aggregate_distance
# ------------------------------------------------------------------
max_patch_diag = math.sqrt(3 * (64 ** 2))
max_patch_diag_tensor = torch.tensor(max_patch_diag, device=config.DEVICE, dtype=torch.float32)


def aggregate_distance(metric, n_total, n_skipped):
    """Aggregate a MONAI distance metric, penalising skipped samples."""
    if metric is None:
        return float("nan")
    raw = metric.aggregate()
    metric.reset()
    if raw is None:
        return max_patch_diag
    if isinstance(raw, torch.Tensor):
        if raw.numel() == 0:
            return max_patch_diag
        cat = raw.flatten()
    elif isinstance(raw, list):
        valid_tensors = []
        for r in raw:
            if r is not None and isinstance(r, torch.Tensor) and r.numel() > 0:
                valid_tensors.append(r.flatten())
        if not valid_tensors:
            return max_patch_diag
        cat = torch.cat(valid_tensors)
    else:
        return max_patch_diag

    invalid_mask = torch.isinf(cat) | torch.isnan(cat)
    cat = torch.where(invalid_mask, max_patch_diag_tensor.to(cat.device), cat)

    n_penalties = n_total - len(cat)
    if n_penalties > 0:
        penalties = torch.full(
            (n_penalties,), max_patch_diag, device=cat.device, dtype=cat.dtype
        )
        cat = torch.cat([cat, penalties])

    return cat.mean().item()


# ------------------------------------------------------------------
# Evaluate one checkpoint
# ------------------------------------------------------------------
def evaluate_one(experiment_name, test_keys, device):
    ckpt_path = CHECKPOINT_BASE / experiment_name / "best_dice_unet3d.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    print(f"  Loading checkpoint: {ckpt_path.name}")
    test_ds = LIDCDataset(keys=test_keys, is_train=False, use_sdf=False)
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True,
    )

    model = UNet3DPaper(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(
        include_background=False, percentile=95, reduction="none"
    )
    hd100_metric = HausdorffDistanceMetric(
        include_background=False, percentile=100, reduction="none"
    )
    assd_metric = SurfaceDistanceMetric(
        include_background=False, symmetric=True, reduction="none"
    )
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["sensitivity", "specificity", "precision", "f1_score"],
        reduction="mean",
    )

    hd95_skipped = 0
    sample_count = 0
    tp_total = fp_total = fn_total = 0
    size_bucket_dice = {"small": [], "medium": [], "large": []}
    per_sample_dice = []

    start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            patches, masks = batch
            patches, masks = patches.to(device), masks.to(device)
            outputs = model(patches)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            dice_metric(y_pred=preds, y=masks)
            confusion_metric(y_pred=preds, y=masks)

            for i in range(preds.shape[0]):
                sample_count += 1
                p, t = preds[i:i + 1], masks[i:i + 1]
                pred_has_fg = torch.any(p > 0)
                target_has_fg = torch.any(t > 0)

                if pred_has_fg and target_has_fg:
                    hd95_metric(y_pred=p, y=t)
                    hd100_metric(y_pred=p, y=t)
                    assd_metric(y_pred=p, y=t)
                else:
                    hd95_skipped += 1

                tp = int(((p == 1) & (t == 1)).sum().item())
                fp = int(((p == 1) & (t == 0)).sum().item())
                fn = int(((p == 0) & (t == 1)).sum().item())
                tp_total += tp
                fp_total += fp
                fn_total += fn

                gt_voxels = int(t.sum().item())
                sample_dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                per_sample_dice.append(sample_dice)
                size_bucket_dice[size_bucket(gt_voxels)].append(sample_dice)

    elapsed = time.time() - start

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    mean_hd95 = aggregate_distance(hd95_metric, sample_count, hd95_skipped)
    mean_hd100 = aggregate_distance(hd100_metric, sample_count, hd95_skipped)
    mean_assd = aggregate_distance(assd_metric, sample_count, hd95_skipped)

    cm_results = confusion_metric.aggregate()
    sensitivity, specificity, precision, f1 = [r.item() for r in cm_results]
    confusion_metric.reset()

    iou = tp_total / (tp_total + fp_total + fn_total + 1e-8)

    result = {
        "experiment_name": experiment_name,
        "n_test_samples": len(test_keys),
        "dice": float(mean_dice),
        "hd95": float(mean_hd95),
        "hd100": float(mean_hd100),
        "iou": float(iou),
        "assd": float(mean_assd),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "hd95_skipped": hd95_skipped,
        "hd95_total": sample_count,
        "size_stratified": {
            "small": {
                "mean": float(np.mean(size_bucket_dice["small"])) if size_bucket_dice["small"] else None,
                "std": float(np.std(size_bucket_dice["small"], ddof=0)) if size_bucket_dice["small"] else None,
                "n": len(size_bucket_dice["small"]),
            },
            "medium": {
                "mean": float(np.mean(size_bucket_dice["medium"])) if size_bucket_dice["medium"] else None,
                "std": float(np.std(size_bucket_dice["medium"], ddof=0)) if size_bucket_dice["medium"] else None,
                "n": len(size_bucket_dice["medium"]),
            },
            "large": {
                "mean": float(np.mean(size_bucket_dice["large"])) if size_bucket_dice["large"] else None,
                "std": float(np.std(size_bucket_dice["large"], ddof=0)) if size_bucket_dice["large"] else None,
                "n": len(size_bucket_dice["large"]),
            },
        },
        "per_sample_dice": per_sample_dice,
        "eval_time_sec": elapsed,
    }

    out_path = TEST_RESULTS_DIR / f"test_{experiment_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [OK] Saved → {out_path}")
    return result


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 72)
    print(" LIDC-IDRI Test-Set Evaluation  —  One-Look, All 15 Checkpoints")
    print("=" * 72)

    device = torch.device(config.DEVICE)
    print(f"[INFO] Device: {device}")

    with open(SPLIT_PATH) as f:
        split_data = json.load(f)
    test_keys = split_data.get("test_keys", [])
    print(f"[INFO] Test set: {len(test_keys)} patches (held out)")

    all_results = {}
    missing_ckpts = []

    for cfg_name, exp_prefix in CONFIGS.items():
        print(f"\n--- {cfg_name.upper().replace('_', ' ')} ---")
        for seed in SEEDS:
            exp_name = f"{exp_prefix}_seed{seed}"
            res = evaluate_one(exp_name, test_keys, device)
            if res is not None:
                all_results[exp_name] = res
            else:
                missing_ckpts.append(exp_name)

    # Consolidated summary
    summary = {
        "n_configs": len(CONFIGS),
        "n_seeds": len(SEEDS),
        "n_expected": len(CONFIGS) * len(SEEDS),
        "n_completed": len(all_results),
        "missing_checkpoints": missing_ckpts,
        "results": all_results,
    }
    summary_path = TEST_RESULTS_DIR / "test_summary_all.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f" Completed: {len(all_results)}/{len(CONFIGS) * len(SEEDS)} evaluations")
    if missing_ckpts:
        print(f" Missing checkpoints: {missing_ckpts}")
    print(f" Summary JSON → {summary_path}")
    print("=" * 72)
    print("\n ONE-LOOK DISCIPLINE")
    print(" These test-set numbers are final. Do not use them to select")
    print(" seeds, tune hyperparameters, or revise model/loss choices.")
    print("=" * 72)


if __name__ == "__main__":
    main()