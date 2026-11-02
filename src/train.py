import argparse
import json
import random
import time
import platform
import math
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric, SurfaceDistanceMetric
from boundary_loss import DiceFocalBoundaryLoss
from unet3d_paper import UNet3DPaper
import config
from lidc_dataset import LIDCDataset
from datetime import datetime


def get_code_version():
    files = ["train.py", "config.py", "boundary_loss.py", "lidc_dataset.py", "unet3d_paper.py"]
    hasher = hashlib.sha256()
    for fname in sorted(files):
        fpath = config.BASE_DIR / fname
        if fpath.exists():
            hasher.update(fpath.read_bytes())
    return hasher.hexdigest()[:12]


def size_bucket(voxel_count):
    if voxel_count < 100:
        return "small"
    elif voxel_count < 500:
        return "medium"
    else:
        return "large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary", action="store_true")
    parser.add_argument("--adaptive-boundary", action="store_true",
                        help="Use adaptive/gated boundary-loss weighting (PCG-BW, hysteresis-stabilized)")
    parser.add_argument("--architecture", type=str, default="unet3d_paper")
    parser.add_argument("--interactive-save", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    use_sdf = args.boundary
    adaptive_mode = args.adaptive_boundary and use_sdf
    architecture = args.architecture

    if args.adaptive_boundary and not use_sdf:
        print("[WARNING] --adaptive-boundary ignored because --boundary is not set.")

    loss_variant = ("boundary_adaptive" if adaptive_mode else
                    "boundary_fixed_schedule" if use_sdf else "none")
    loss_tag = ("boundary_adaptive" if adaptive_mode else
                "boundary" if use_sdf else "baseline")
    experiment_name = f"{loss_tag}_{architecture}_seed{seed}"

    checkpoint_dir = config.OUTPUT_DIR / "checkpoints" / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(config.DEVICE)
    print(f"[INFO] Training on device: {device}")
    print(f"[INFO] Experiment: {experiment_name}")
    print(f"[INFO] Seed: {seed} | Boundary: {use_sdf} | Adaptive: {adaptive_mode} | Arch: {architecture}")

    split_path = config.BASE_DIR / "train_val_test_split.json"
    with open(split_path) as f:
        split_data = json.load(f)

    train_keys = split_data["train_keys"]
    val_keys = split_data["val_keys"]
    test_keys = split_data.get("test_keys", [])
    print(f"[INFO] Loaded patient-level split — Train: {len(train_keys)} | Val: {len(val_keys)} | Test: {len(test_keys)} (held out)")

    assert len(set(train_keys) & set(val_keys)) == 0, "LEAKAGE: train ∩ val not empty!"
    assert len(set(train_keys) & set(test_keys)) == 0, "LEAKAGE: train ∩ test not empty!"
    assert len(set(val_keys) & set(test_keys)) == 0, "LEAKAGE: val ∩ test not empty!"

    train_ds = LIDCDataset(keys=train_keys, is_train=True, use_sdf=use_sdf)
    val_ds = LIDCDataset(keys=val_keys, is_train=False, use_sdf=use_sdf)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True,
                               persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True,
                             persistent_workers=True)

    model = UNet3DPaper(in_channels=1, out_channels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            last_conv = module
    if last_conv is not None and last_conv.bias is not None:
        init_bias = -np.log((1.0 - 0.01) / 0.01)
        with torch.no_grad():
            last_conv.bias.fill_(init_bias)
        print(f"[Init] Output Conv3d bias calibrated to {init_bias:.4f}")

    base_criterion = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
    if use_sdf:
        criterion = DiceFocalBoundaryLoss(base_criterion, boundary_weight=0.0)
        if adaptive_mode:
            print(f"[INFO] Using Adaptive DiceFocal + Boundary Loss (PCG-BW, hysteresis-stabilized)")
            print(f"       EMA α={config.ALPHA_EMA}, k={config.K_WINDOW}, "
                  f"enter_τ={config.GATE_TAU_ENTER}, exit_τ={config.GATE_TAU_EXIT}, "
                  f"patience={config.GATE_PATIENCE}, λ_max={config.LAMBDA_MAX}")
        else:
            print(f"[INFO] Using Fixed-Schedule DiceFocal + Boundary Loss (target={config.BOUNDARY_LOSS_WEIGHT}, warmup={config.BOUNDARY_WARMUP_EPOCHS})")
    else:
        criterion = base_criterion
        print(f"[INFO] Using baseline DiceFocal Loss only")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    hd100_metric = HausdorffDistanceMetric(include_background=False, percentile=100, reduction="none")
    assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="none")
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

    max_patch_diag = math.sqrt(3 * (64 ** 2))
    max_patch_diag_tensor = torch.tensor(max_patch_diag, device=device, dtype=torch.float32)

    #-----------------------------------
    # Adaptive gating state (hysteresis)
    #-----------------------------------
    ema_primary_val_losses = []
    gate_active = False
    consecutive_enter_count = 0
    consecutive_exit_count = 0
    smoothed_boundary_weight = 0.0
    if use_sdf:
        criterion.boundary_weight = 0.0

    for epoch in range(config.EPOCHS):
        if use_sdf and not adaptive_mode:
            # Fixed schedule
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

            with torch.no_grad():
                try:
                    d_loss = base_criterion.dice(outputs, masks).item()
                    f_loss = base_criterion.focal(outputs, masks).item()
                except AttributeError:
                    pred_probs = torch.sigmoid(outputs)
                    d_loss = (2.0 * (pred_probs * masks).sum() / (pred_probs.sum() + masks.sum() + 1e-8)).item()
                    f_loss = 0.0

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

        model.eval()
        val_loss = 0.0
        primary_val_loss = 0.0
        dice_metric.reset()
        hd95_metric.reset()
        hd100_metric.reset()
        assd_metric.reset()
        confusion_metric.reset()

        hd95_skipped = 0
        val_sample_count = 0
        tp_total = fp_total = fn_total = tn_total = 0
        size_bucket_dice = {"small": [], "medium": [], "large": []}
        per_sample_dice = []

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

                primary_val_loss += base_criterion(outputs, masks).item() * patches.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                dice_metric(y_pred=preds, y=masks)
                confusion_metric(y_pred=preds, y=masks)

                for i in range(preds.shape[0]):
                    val_sample_count += 1
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
                    per_sample_dice.append(sample_dice)
                    size_bucket_dice[size_bucket(gt_voxels)].append(sample_dice)

        val_loss /= len(val_ds)
        primary_val_loss /= len(val_ds)

        mean_val_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        def _aggregate_distance(metric, n_total, n_skipped):
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
                penalties = torch.full((n_penalties,), max_patch_diag, device=cat.device, dtype=cat.dtype)
                cat = torch.cat([cat, penalties])

            return cat.mean().item()

        mean_hd95 = _aggregate_distance(hd95_metric, val_sample_count, hd95_skipped)
        mean_hd100 = _aggregate_distance(hd100_metric, val_sample_count, hd95_skipped)
        mean_assd = _aggregate_distance(assd_metric, val_sample_count, hd95_skipped)

        cm_results = confusion_metric.aggregate()
        sensitivity, specificity, precision, f1 = [r.item() for r in cm_results]
        confusion_metric.reset()

        iou = tp_total / (tp_total + fp_total + fn_total + 1e-8)

        scheduler.step(mean_val_dice)

        # -------------------------------------------------------------------------
        # Adaptive gating: hysteresis state machine, computes weight for NEXT epoch
        # -------------------------------------------------------------------------
        velocity = None
        ema_val = None
        if use_sdf and adaptive_mode:
            if len(ema_primary_val_losses) == 0:
                ema_val = primary_val_loss
            else:
                ema_val = ((1 - config.ALPHA_EMA) * ema_primary_val_losses[-1] +
                           config.ALPHA_EMA * primary_val_loss)
            ema_primary_val_losses.append(ema_val)

            if len(ema_primary_val_losses) > config.K_WINDOW:
                ema_old = ema_primary_val_losses[-(config.K_WINDOW + 1)]
                ema_new = ema_primary_val_losses[-1]
                velocity = (ema_old - ema_new) / (ema_old + 1e-8)

                if not gate_active:
                    if velocity <= config.GATE_TAU_ENTER:
                        consecutive_enter_count += 1
                    else:
                        consecutive_enter_count = 0

                    if consecutive_enter_count >= config.GATE_PATIENCE:
                        gate_active = True
                        consecutive_exit_count = 0
                        print(f"  [Gate] Activated at epoch {epoch + 1} "
                              f"(velocity={velocity:.5f} sustained {config.GATE_PATIENCE} epochs)", flush=True)
                else:
                    if velocity > config.GATE_TAU_EXIT:
                        consecutive_exit_count += 1
                    else:
                        consecutive_exit_count = 0

                    if consecutive_exit_count >= config.GATE_PATIENCE:
                        gate_active = False
                        consecutive_enter_count = 0
                        print(f"  [Gate] Deactivated at epoch {epoch + 1} "
                              f"(velocity={velocity:.5f} sustained {config.GATE_PATIENCE} epochs)", flush=True)

                if gate_active and velocity > 0:
                    exponent = np.clip((velocity - config.TAU_VELOCITY) / config.GAMMA_SIGMOID, -500, 500)
                    sigmoid_val = 1.0 / (1.0 + math.exp(exponent))
                    target_weight = config.LAMBDA_MAX * sigmoid_val
                else:
                    target_weight = 0.0

                smoothed_boundary_weight = ((1 - config.WEIGHT_EMA_ALPHA) * smoothed_boundary_weight +
                                             config.WEIGHT_EMA_ALPHA * target_weight)
                criterion.boundary_weight = smoothed_boundary_weight

        # ---------------
        # Console logging
        # ---------------
        bw_str = f" | BoundaryW: {criterion.boundary_weight:.4f}" if use_sdf else ""
        hd95_str = f"{mean_hd95:.2f}" if not math.isnan(mean_hd95) else "nan"
        print(f"Epoch [{epoch + 1:02d}/{config.EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {mean_val_dice:.4f} | Val HD95: {hd95_str} "
              f"(skipped {hd95_skipped}/{val_sample_count}) | IoU: {iou:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}{bw_str}", flush=True)

        # -------
        # History
        # -------
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
            "primary_val_loss": primary_val_loss,
            "ema_primary_val_loss": ema_val,
            "velocity": velocity,
            "gate_active": gate_active if (use_sdf and adaptive_mode) else None,
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
            "per_sample_dice": per_sample_dice,
            "epoch_time_sec": epoch_time, "peak_gpu_mem_mb": peak_mem_mb,
            "lr": optimizer.param_groups[0]['lr'],
            "boundary_weight": criterion.boundary_weight if use_sdf else None,
            "hd95_skipped": hd95_skipped, "hd95_total": val_sample_count,
        })

        # -----------
        # Checkpoints
        # -----------
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            best_val_dice_epoch = epoch + 1
            ckpt_path = checkpoint_dir / "best_dice_unet3d.pt"
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_dice": best_val_dice, "val_hd95": mean_hd95}, ckpt_path)
            print(f"  --> [Dice] Checkpoint saved (Val Dice: {best_val_dice:.4f})", flush=True)

        if not math.isnan(mean_hd95) and mean_hd95 < best_val_hd95:
            best_val_hd95 = mean_hd95
            best_val_hd95_epoch = epoch + 1
            ckpt_path = checkpoint_dir / "best_hd95_unet3d.pt"
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_dice": mean_val_dice, "val_hd95": best_val_hd95}, ckpt_path)
            print(f"  --> [HD95] Checkpoint saved (Val HD95: {best_val_hd95:.4f})", flush=True)

    # ----------
    # Final save
    # ----------
    last10 = history[-10:]
    result = {
        "experiment_name": experiment_name, "seed": seed, "use_boundary_loss": use_sdf,
        "adaptive_boundary": adaptive_mode,
        "architecture": architecture, "loss_variant": loss_variant,
        "code_version": get_code_version(), "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch_size": config.BATCH_SIZE, "learning_rate": config.LEARNING_RATE,
        "epochs_planned": config.EPOCHS, "epochs_completed": len(history),
        "boundary_loss_weight_target": getattr(config, "BOUNDARY_LOSS_WEIGHT", None) if use_sdf else None,
        "boundary_warmup_epochs": getattr(config, "BOUNDARY_WARMUP_EPOCHS", None) if use_sdf else None,
        "adaptive_params": {
            "alpha_ema": config.ALPHA_EMA,
            "k_window": config.K_WINDOW,
            "tau_velocity": config.TAU_VELOCITY,
            "gamma_sigmoid": config.GAMMA_SIGMOID,
            "lambda_max": config.LAMBDA_MAX,
            "gate_tau_enter": config.GATE_TAU_ENTER,
            "gate_tau_exit": config.GATE_TAU_EXIT,
            "gate_patience": config.GATE_PATIENCE,
            "weight_ema_alpha": config.WEIGHT_EMA_ALPHA,
        } if adaptive_mode else None,
        "train_set_size": len(train_ds), "val_set_size": len(val_ds), "n_parameters": n_params,
        "best_val_dice": best_val_dice, "best_val_dice_epoch": best_val_dice_epoch,
        "best_val_hd95": best_val_hd95, "best_val_hd95_epoch": best_val_hd95_epoch,
        "final_val_dice": history[-1]["val_dice"], "final_val_hd95": history[-1]["val_hd95"],
        "final_val_iou": history[-1]["val_iou"], "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"], "final_lr": history[-1]["lr"],
        "val_dice_mean_last10": float(np.mean([h["val_dice"] for h in last10])),
        "val_dice_std_last10": float(np.std([h["val_dice"] for h in last10])),
        "val_hd95_mean_last10": float(np.nanmean([h["val_hd95"] for h in last10])),
        "val_hd95_std_last10": float(np.nanstd([h["val_hd95"] for h in last10])),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "pytorch_version": torch.__version__, "hostname": platform.node(),
        "history": history,
    }
    result_path = config.RESULTS_DIR / f"{experiment_name}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Result summary written to {result_path}")

    if args.interactive_save:
        from archiving import promote_to_archive
        answer = input("\nSave this run's full results for graphs/comparisons? [y/N]: ").strip().lower()
        if answer == "y":
            default_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d')}"
            archive_name = input(f"Archive name [{default_name}]: ").strip() or default_name
            notes = input("Notes (why keep this run): ").strip()
            archive_dir = promote_to_archive(experiment_name, archive_name, notes)
            print(f"Archived -> {archive_dir}")
        else:
            print("Not archived. Raw result still available in outputs/results/.")


if __name__ == "__main__":
    main()