"""
One-time diagnostic: measures gradient magnitude of the boundary loss term
vs. the regional (Dice+Focal) loss term, at the output layer, using a
few real batches from a trained checkpoint. Does NOT modify train.py or
re-run any training -- purely inspects gradients on already-trained weights.

Extended to support both architectures via --architecture, using the
fixed-schedule boundary checkpoint for each (matching the original script's
own choice of `boundary_unet3d_paper_seed42`, not an adaptive checkpoint --
kept identical here so the UNETR measurement is directly comparable to the
existing UNet3DPaper figures in gradient_analysis.md).

Usage:
    python grad_telemetry_check.py --architecture unet3d_paper
    python grad_telemetry_check.py --architecture unetr
"""
import argparse
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.losses import DiceFocalLoss
from boundary_loss import DiceFocalBoundaryLoss
from unet3d_paper import UNet3DPaper
from unetr_paper import UNETRPaper
from lidc_dataset import LIDCDataset
import config

parser = argparse.ArgumentParser()
parser.add_argument("--architecture", type=str, default="unet3d_paper",
                     help="unet3d_paper or unetr")
parser.add_argument("--seed", type=int, default=42,
                     help="which fixed-schedule boundary checkpoint's seed to use")
args = parser.parse_args()
architecture = args.architecture

device = torch.device(config.DEVICE)

# Load a representative trained checkpoint (fixed-schedule boundary).
ckpt_path = (config.OUTPUT_DIR / "checkpoints"
             / f"boundary_{architecture}_seed{args.seed}" / "best_dice_unet3d.pt")

if architecture == "unet3d_paper":
    model = UNet3DPaper(in_channels=1, out_channels=1).to(device)
elif architecture == "unetr":
    model = UNETRPaper(in_channels=1, out_channels=1, img_size=config.PATCH_SIZE[0]).to(device)
else:
    raise ValueError(f"Unknown --architecture: '{architecture}'")

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.train()  # need grad computation, but weights are frozen (trained) — just probing gradients

with open(config.BASE_DIR / "train_val_test_split.json") as f:
    split_data = json.load(f)
val_keys = split_data["val_keys"][:32]  # small sample, enough for a stable gradient estimate

val_ds = LIDCDataset(keys=val_keys, is_train=False, use_sdf=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

base_criterion = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
boundary_criterion = DiceFocalBoundaryLoss(base_criterion, boundary_weight=1.0)  # weight=1.0: measure RAW magnitude, not scaled

regional_norms = []
boundary_norms = []

for batch in val_loader:
    patches, masks, sdfs = batch
    patches, masks, sdfs = patches.to(device), masks.to(device), sdfs.to(device)

    outputs = model(patches)

    # --- Regional loss gradient ---
    model.zero_grad()
    regional_loss = base_criterion(outputs, masks)
    regional_grad = torch.autograd.grad(regional_loss, outputs, retain_graph=True)[0]
    regional_norms.append(regional_grad.norm().item())

    # --- Boundary loss gradient (raw, weight=1.0) ---
    model.zero_grad()
    pred_probs = torch.sigmoid(outputs)
    b_loss = boundary_criterion.boundary_loss(pred_probs, sdfs)
    boundary_grad = torch.autograd.grad(b_loss, outputs, retain_graph=True)[0]
    boundary_norms.append(boundary_grad.norm().item())

regional_mean = np.mean(regional_norms)
boundary_mean = np.mean(boundary_norms)
total = regional_mean + boundary_mean
raw_share_pct = 100 * boundary_mean / total
effective_pct = 100 * (0.05 * boundary_mean) / (regional_mean + 0.05 * boundary_mean)

print(f"Architecture: {architecture}  (checkpoint: {ckpt_path.parent.name})")
print(f"Regional (Dice+Focal) gradient norm, mean over batches: {regional_mean:.6f}")
print(f"Boundary (raw, weight=1.0) gradient norm, mean over batches: {boundary_mean:.6f}")
print(f"Boundary share of combined raw gradient magnitude: {raw_share_pct:.2f}%")
print()
print(f"At the actual trained weight (0.05), boundary's effective contribution "
      f"is scaled down further: ~{effective_pct:.2f}%")

out_path = config.OUTPUT_DIR / "test_results" / f"grad_telemetry_{architecture}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "architecture": architecture, "checkpoint": str(ckpt_path),
        "regional_grad_norm_mean": float(regional_mean),
        "boundary_grad_norm_mean_raw": float(boundary_mean),
        "boundary_raw_share_pct": float(raw_share_pct),
        "boundary_effective_share_pct_at_trained_weight": float(effective_pct),
    }, f, indent=2)
print(f"\n[OK] Saved → {out_path}")