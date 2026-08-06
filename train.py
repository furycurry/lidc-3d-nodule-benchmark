import argparse
import json
import random
import time
import platform
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric, SurfaceDistanceMetric
from boundary_loss import DiceFocalBoundaryLoss
import config
from lidc_dataset import LIDCDataset
from datetime import datetime


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.BASE_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def size_bucket(voxel_count):
    if voxel_count < 100:
        return "small"
    elif voxel_count < 500:
        return "medium"
    else:
        return "large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--boundary", action="store_true", help="Use DiceFocal + Boundary loss")
    parser.add_argument("--architecture", type=str, default="unet3d")
    args = parser.parse_args()

    seed = args.seed
    use_sdf = args.boundary
    architecture = args.architecture
    loss_variant = "boundary_fixed_schedule" if use_sdf else "none"
    experiment_name = f"{'boundary' if use_sdf else 'baseline'}_seed{seed}"

    checkpoint_dir = config.OUTPUT_DIR / "checkpoints" / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(config.DEVICE)
    print(f"[INFO] Training on device: {device}")
    print(f"[INFO] Experiment: {experiment_name}")
    print(f"[INFO] Seed: {seed} | Boundary loss: {use_sdf}")

    split_path = config.BASE_DIR / "train_val_test_split.json"
    with open(split_path) as f:
        split_data = json.load(f)

    train_keys = split_data["train_keys"]
    val_keys = split_data["val_keys"]
    print(f"[INFO] Loaded patient-level split — Train: {len(train_keys)} patches | Val: {len(val_keys)} patches")

    train_ds = LIDCDataset(keys=train_keys, is_train=True, use_sdf=use_sdf)
    val_ds = LIDCDataset(keys=val_keys, is_train=False, use_sdf=use_sdf)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    base_criterion = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
    if use_sdf:
        criterion = DiceFocalBoundaryLoss(base_criterion, boundary_weight=0.0)
        print(f"[INFO] Using DiceFocal + Boundary Loss (target weight={config.BOUNDARY_LOSS_WEIGHT}, "
              f"warmup={config.BOUNDARY_WARMUP_EPOCHS} epochs)")
    else:
        criterion = base_criterion
        print(f"[INFO] Using baseline DiceFocal Loss only")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    hd100_metric = HausdorffDistanceMetric(include_background=False, percentile=100, reduction="mean")
    assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["sensitivity", "specificity", "precision", "f1_score"],
        reduction="mean"
    )

    best_val_dice = 0.0
    best_val_dice_epoch = None
    best_val_hd95 = float("inf")
    best_val_hd95_epoch = None
    history = []

    for epoch in range(config.EPOCHS):
        if use_sdf:
            if epoch < config.BOUNDARY_WARMUP_EPOCHS:
                effective_weight = 0.0
            else:
                ramp_progress = min(1.0, (epoch - config.BOUNDARY_WARMUP_EPOCHS) / 10.0)
                effective_weight = config.BOUNDARY_LOSS_WEIGHT * ramp_progress
            criterion.boundary_weight = effective_weight

        model.train()
        train_loss = 0.0
        train_dice_component = 0.0
        train_focal_component = 0.0
        train_boundary_component = 0.0

        epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for batch in train_loader:
            if use_sdf:
                patches, masks, sdfs = batch
                sdfs = sdfs.to(device)
            else:
                patches, masks = batch

            patches, masks = patches.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, masks, sdfs) if use_sdf else criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * patches.size(0)

            # Per-component logging (no extra backward, just forward for bookkeeping)
            with torch.no_grad():
                d_loss = base_criterion.dice(outputs, masks).item()
                f_loss = base_criterion.focal(outputs, masks).item()
                train_dice_component += d_loss * patches.size(0)
                train_focal_component += f_loss * patches.size(0)
                if use_sdf:
                    pred_probs = torch.sigmoid(outputs)
                    b_loss = criterion.boundary_loss(pred_probs, sdfs).item()
                    train_boundary_component += b_loss * patches.size(0)

        epoch_time = time.time() - epoch_start_time
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if torch.cuda.is_available() else None

        train_loss /= len(train_ds)
        train_dice_component /= len(train_ds)
        train_focal_component /= len(train_ds)
        train_boundary_component = train_boundary_component / len(train_ds) if use_sdf else None

        # Validation Loop
        model.eval()
        val_loss = 0.0
        dice_metric.reset()
        hd95_metric.reset()
        hd100_metric.reset()
        assd_metric.reset()
        confusion_metric.reset()
        hd95_skipped = 0
        hd95_total = 0
        tp_total = fp_total = fn_total = tn_total = 0
        size_bucket_dice = {"small": [], "medium": [], "large": []}

        with torch.no_grad():
            for batch in val_loader:
                if use_sdf:
                    patches, masks, sdfs = batch
                    sdfs = sdfs.to(device)
                else:
                    patches, masks = batch

                patches, masks = patches.to(device), masks.to(device)
                outputs = model(patches)

                v_loss = criterion(outputs, masks, sdfs) if use_sdf else criterion(outputs, masks)
                val_loss += v_loss.item() * patches.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                dice_metric(y_pred=preds, y=masks)
                confusion_metric(y_pred=preds, y=masks)

                for i in range(preds.shape[0]):
                    hd95_total += 1
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
                    tn = int(((p == 0) & (t == 0)).sum().item())
                    tp_total += tp; fp_total += fp; fn_total += fn; tn_total += tn

                    gt_voxels = int(t.sum().item())
                    sample_dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                    size_bucket_dice[size_bucket(gt_voxels)].append(sample_dice)

        val_loss /= len(val_ds)
        mean_val_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        try:
            mean_hd95 = hd95_metric.aggregate().item()
        except (RuntimeError, ValueError):
            mean_hd95 = float("nan")
        hd95_metric.reset()

        try:
            mean_hd100 = hd100_metric.aggregate().item()
        except (RuntimeError, ValueError):
            mean_hd100 = float("nan")
        hd100_metric.reset()

        try:
            mean_assd = assd_metric.aggregate().item()
        except (RuntimeError, ValueError):
            mean_assd = float("nan")
        assd_metric.reset()

        cm_results = confusion_metric.aggregate()
        sensitivity, specificity, precision, f1 = [r.item() for r in cm_results]
        confusion_metric.reset()

        iou = tp_total / (tp_total + fp_total + fn_total + 1e-8)

        scheduler.step(mean_val_dice)

        bw_str = f" | BoundaryW: {criterion.boundary_weight:.4f}" if use_sdf else ""
        print(f"Epoch [{epoch + 1:02d}/{config.EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {mean_val_dice:.4f} | Val HD95: {mean_hd95:.4f} "
              f"(skipped {hd95_skipped}/{hd95_total}) | IoU: {iou:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}{bw_str}", flush=True)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_dice_component": train_dice_component, "train_focal_component": train_focal_component,
            "train_boundary_component": train_boundary_component,
            "val_dice": mean_val_dice, "val_hd95": mean_hd95, "val_hd100": mean_hd100, "val_assd": mean_assd,
            "val_iou": iou, "val_precision": precision, "val_recall_sensitivity": sensitivity,
            "val_specificity": specificity, "val_f1": f1,
            "tp": tp_total, "fp": fp_total, "fn": fn_total, "tn": tn_total,
            "dice_small_mean": float(np.mean(size_bucket_dice["small"])) if size_bucket_dice["small"] else None,
            "dice_small_n": len(size_bucket_dice["small"]),
            "dice_medium_mean": float(np.mean(size_bucket_dice["medium"])) if size_bucket_dice["medium"] else None,
            "dice_medium_n": len(size_bucket_dice["medium"]),
            "dice_large_mean": float(np.mean(size_bucket_dice["large"])) if size_bucket_dice["large"] else None,
            "dice_large_n": len(size_bucket_dice["large"]),
            "epoch_time_sec": epoch_time, "peak_gpu_mem_mb": peak_mem_mb,
            "lr": optimizer.param_groups[0]['lr'],
            "boundary_weight": criterion.boundary_weight if use_sdf else None,
            "hd95_skipped": hd95_skipped, "hd95_total": hd95_total,
        })

        # Dual checkpointing: best-by-Dice AND best-by-HD95, tracked independently
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            best_val_dice_epoch = epoch + 1
            ckpt_path = checkpoint_dir / "best_dice_unet3d.pt"
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice, "val_hd95": mean_hd95,
            }, ckpt_path)
            print(f"  --> [Dice] Checkpoint saved (Val Dice: {best_val_dice:.4f})", flush=True)

        if not np.isnan(mean_hd95) and mean_hd95 < best_val_hd95:
            best_val_hd95 = mean_hd95
            best_val_hd95_epoch = epoch + 1
            ckpt_path = checkpoint_dir / "best_hd95_unet3d.pt"
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": mean_val_dice, "val_hd95": best_val_hd95,
            }, ckpt_path)
            print(f"  --> [HD95] Checkpoint saved (Val HD95: {best_val_hd95:.4f})", flush=True)

    # --- Final structured result summary ---
    last10 = history[-10:]
    result = {
        "experiment_name": experiment_name,
        "seed": seed,
        "use_boundary_loss": use_sdf,
        "architecture": architecture,
        "loss_variant": loss_variant,
        "code_version": get_git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),

        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "epochs_planned": config.EPOCHS,
        "epochs_completed": len(history),
        "boundary_loss_weight_target": getattr(config, "BOUNDARY_LOSS_WEIGHT", None) if use_sdf else None,
        "boundary_warmup_epochs": getattr(config, "BOUNDARY_WARMUP_EPOCHS", None) if use_sdf else None,
        "train_set_size": len(train_ds),
        "val_set_size": len(val_ds),
        "n_parameters": n_params,

        "best_val_dice": best_val_dice,
        "best_val_dice_epoch": best_val_dice_epoch,
        "best_val_hd95": best_val_hd95,
        "best_val_hd95_epoch": best_val_hd95_epoch,

        "final_val_dice": history[-1]["val_dice"],
        "final_val_hd95": history[-1]["val_hd95"],
        "final_val_iou": history[-1]["val_iou"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_lr": history[-1]["lr"],

        "val_dice_mean_last10": float(np.mean([h["val_dice"] for h in last10])),
        "val_dice_std_last10": float(np.std([h["val_dice"] for h in last10])),
        "val_hd95_mean_last10": float(np.nanmean([h["val_hd95"] for h in last10])),
        "val_hd95_std_last10": float(np.nanstd([h["val_hd95"] for h in last10])),

        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "pytorch_version": torch.__version__,
        "hostname": platform.node(),

        "history": history,
    }
    result_path = config.RESULTS_DIR / f"{experiment_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Result summary written to {result_path}")


if __name__ == "__main__":
    main()