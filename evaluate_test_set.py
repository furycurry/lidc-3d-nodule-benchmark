#!/usr/bin/env python3
"""
evaluate_test_set.py

One-look test-set evaluation for the LIDC-IDRI 3D benchmark.
Runs checkpoints (best_dice_*.pt per config/seed) against the held-out
434-patch test partition. Pure inference — no training, no SDF computation.

Updates & Fixes:
  - DataLoader instantiated once in main() to eliminate multiprocessing worker leaks.
  - Dynamic checkpoint resolution across architecture-specific naming conventions.
  - Safe 3D patch-size tuple resolution for UNETRPaper.
  - Non-destructive summary merging (preserves previously sealed results).
  - CLI argument (--arch) to select target architecture(s) without editing source.

Usage:
    python src/evaluate_test_set.py
    python src/evaluate_test_set.py --arch unet3d_paper
    python src/evaluate_test_set.py --arch unetr
"""

import argparse
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    ConfusionMatrixMetric,
    SurfaceDistanceMetric,
)

import config
from lidc_dataset import LIDCDataset
from unet3d_paper import UNet3DPaper
try:
    from unetr_paper import UNETRPaper
except ImportError:
    UNETRPaper = None

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CHECKPOINT_BASE = config.OUTPUT_DIR / "checkpoints"
TEST_RESULTS_DIR = config.OUTPUT_DIR / "test_results"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = getattr(config, "SPLIT_PATH", config.BASE_DIR / "train_val_test_split.json")

# ------------------------------------------------------------------
# Experimental design constants
# ------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 2024]

CONFIGS = {
    "baseline":          "baseline",
    "boundary_fixed":    "boundary",
    "boundary_adaptive": "boundary_adaptive",
}

DEFAULT_ARCHITECTURES = ["unetr"]

# ------------------------------------------------------------------
# Distance-metric & size helpers
# ------------------------------------------------------------------
def size_bucket(voxel_count):
    if voxel_count < 100:
        return "small"
    elif voxel_count < 500:
        return "medium"
    else:
        return "large"


max_patch_diag = math.sqrt(3 * (64 ** 2))
max_patch_diag_tensor = torch.tensor(max_patch_diag, device=config.DEVICE, dtype=torch.float32)


def aggregate_distance(metric, n_total, n_skipped):
    """Aggregate a MONAI distance metric, penalising skipped/empty samples."""
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


def resolve_checkpoint_path(experiment_name, architecture):
    """Dynamically resolves checkpoint paths across architectures."""
    candidates = [
        CHECKPOINT_BASE / experiment_name / "best_dice_unet3d.pt",
        CHECKPOINT_BASE / experiment_name / f"best_dice_{architecture}.pt",
        CHECKPOINT_BASE / experiment_name / "best_dice_model.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    exp_dir = CHECKPOINT_BASE / experiment_name
    if exp_dir.exists():
        pts = list(exp_dir.glob("best_dice*.pt"))
        if pts:
            return pts[0]
    return None


def get_patch_size_tuple():
    raw_size = getattr(config, "PATCH_SIZE", getattr(config, "TARGET_PATCH_SIZE", (64, 64, 64)))
    if isinstance(raw_size, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in raw_size)
    return (int(raw_size), int(raw_size), int(raw_size))


# ------------------------------------------------------------------
# Evaluate one checkpoint
# ------------------------------------------------------------------
def evaluate_one(experiment_name, test_loader, device, architecture):
    ckpt_path = resolve_checkpoint_path(experiment_name, architecture)
    if ckpt_path is None:
        print(f"  [SKIP] Checkpoint not found for experiment: {experiment_name}")
        return None

    print(f"  Loading checkpoint: {ckpt_path.name}")

    if architecture == "unet3d_paper":
        model = UNet3DPaper(in_channels=1, out_channels=1).to(device)
    elif architecture == "unetr":
        if UNETRPaper is None:
            raise ImportError("unetr_paper.py not found or UNETRPaper failed to import.")
        patch_size_3d = get_patch_size_tuple()
        model = UNETRPaper(in_channels=1, out_channels=1, img_size=patch_size_3d).to(device)
    else:
        raise ValueError(f"Unknown architecture: '{architecture}'")

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

    # Micro-average IoU over all voxels (matching train.py final_val_iou convention)
    iou = tp_total / (tp_total + fp_total + fn_total + 1e-8)

    result = {
        "experiment_name": experiment_name,
        "architecture": architecture,
        "n_test_samples": sample_count,
        "checkpoint_used": str(ckpt_path),
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
    parser = argparse.ArgumentParser(description="LIDC-IDRI Held-Out Test Set Evaluation")
    parser.add_argument(
        "--arch",
        type=str,
        nargs="+",
        default=DEFAULT_ARCHITECTURES,
        help="Architecture(s) to evaluate (e.g. --arch unet3d_paper or --arch unetr)",
    )
    args = parser.parse_args()
    selected_architectures = args.arch

    print("=" * 72)
    print(" LIDC-IDRI Test-Set Evaluation  —  One-Look Inference")
    print(f" Target Architectures: {selected_architectures}")
    print("=" * 72)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    with open(SPLIT_PATH) as f:
        split_data = json.load(f)
    test_keys = split_data.get("test_keys", [])
    print(f"[INFO] Test set: {len(test_keys)} patches (held out)")

    # Instantiate DataLoader ONCE to avoid multiprocessing worker pool leaks
    test_ds = LIDCDataset(keys=test_keys, is_train=False, use_sdf=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    all_results = {}
    missing_ckpts = []

    for architecture in selected_architectures:
        for cfg_name, loss_prefix in CONFIGS.items():
            print(f"\n--- {architecture.upper()} / {cfg_name.upper().replace('_', ' ')} ---")
            for seed in SEEDS:
                exp_name = f"{loss_prefix}_{architecture}_seed{seed}"
                res = evaluate_one(exp_name, test_loader, device, architecture)
                if res is not None:
                    all_results[exp_name] = res
                else:
                    missing_ckpts.append(exp_name)

    n_expected = len(selected_architectures) * len(CONFIGS) * len(SEEDS)

    # Consolidated summary: Non-destructive merge
    summary_path = TEST_RESULTS_DIR / "test_summary_all.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing_summary = json.load(f)
        merged_results = existing_summary.get("results", {})
        prior_architectures = existing_summary.get("architectures", [])
    else:
        merged_results = {}
        prior_architectures = []

    merged_results.update(all_results)
    all_architectures = sorted(set(prior_architectures) | set(selected_architectures))

    summary = {
        "architectures": all_architectures,
        "n_configs": len(CONFIGS),
        "n_seeds": len(SEEDS),
        "n_expected_this_run": n_expected,
        "n_completed_this_run": len(all_results),
        "n_completed_total": len(merged_results),
        "missing_checkpoints_this_run": missing_ckpts,
        "results": merged_results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f" This run ({selected_architectures}): {len(all_results)}/{n_expected} evaluations")
    print(f" Cumulative total in summary file: {len(merged_results)}")
    if missing_ckpts:
        print(f" Missing checkpoints this run: {missing_ckpts}")
    print(f" Summary JSON → {summary_path}")
    print("=" * 72)
    print("\n ONE-LOOK DISCIPLINE")
    print(" These test-set numbers are final. Do not use them to select")
    print(" seeds, tune hyperparameters, or revise model/loss choices.")
    print("=" * 72)


if __name__ == "__main__":
    main()