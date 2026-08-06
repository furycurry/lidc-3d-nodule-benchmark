import argparse
import json
import shutil
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import config
from lidc_dataset import LIDCDataset
from monai.networks.nets import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="experiment_name of the run to promote")
parser.add_argument("--name", required=True, help="descriptive archive name")
parser.add_argument("--notes", default="", help="why this run was kept")
args = parser.parse_args()

ARCHIVE_DIR = config.OUTPUT_DIR / "archive" / args.name
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copy result JSON + checkpoints
result_src = config.RESULTS_DIR / f"{args.run}.json"
with open(result_src) as f:
    result_data = json.load(f)

shutil.copy(result_src, ARCHIVE_DIR / "training_history.json")

ckpt_src_dir = config.OUTPUT_DIR / "checkpoints" / args.run
for ckpt_file in ckpt_src_dir.glob("*.pt"):
    shutil.copy(ckpt_file, ARCHIVE_DIR / ckpt_file.name)

# 2. Per-sample metrics + qualitative examples on validation set (Tier B)
device = torch.device(config.DEVICE)
with open(config.BASE_DIR / "train_val_test_split.json") as f:
    split_data = json.load(f)
val_keys = split_data["val_keys"]

use_sdf = result_data.get("use_boundary_loss", False)
val_ds = LIDCDataset(keys=val_keys, is_train=False, use_sdf=use_sdf)

model = UNet(spatial_dims=3, in_channels=1, out_channels=1,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)
ckpt = torch.load(ARCHIVE_DIR / "best_dice_unet3d.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

per_sample_records = []
qualitative_examples = []

with torch.no_grad():
    for i, key in enumerate(val_keys):
        item = val_ds[i]
        patch, mask = item[0], item[1]
        patch_b, mask_b = patch.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
        output = model(patch_b)
        pred = (torch.sigmoid(output) > 0.5).float()

        tp = ((pred == 1) & (mask_b == 1)).sum().item()
        fp = ((pred == 1) & (mask_b == 0)).sum().item()
        fn = ((pred == 0) & (mask_b == 1)).sum().item()
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        per_sample_records.append({"key": key, "dice": dice, "tp": tp, "fp": fp, "fn": fn,
                                    "gt_voxels": int(mask_b.sum().item())})
        qualitative_examples.append((key, dice, patch.numpy(), mask.numpy(), pred.cpu().numpy()))

# Save per-sample CSV
import csv
with open(ARCHIVE_DIR / "per_sample_val_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["key", "dice", "tp", "fp", "fn", "gt_voxels"])
    writer.writeheader()
    writer.writerows(per_sample_records)

# Save 3 qualitative examples: best, worst, median dice
qualitative_examples.sort(key=lambda x: x[1])
picks = {"worst": qualitative_examples[0], "median": qualitative_examples[len(qualitative_examples)//2],
         "best": qualitative_examples[-1]}
for label, (key, dice, img, gt, pred) in picks.items():
    np.savez(ARCHIVE_DIR / f"qualitative_{label}_{key}.npz", image=img, ground_truth=gt, prediction=pred, dice=dice)

# 3. Model complexity stats
n_params = sum(p.numel() for p in model.parameters())

# 4. Write archive metadata
metadata = {
    "archived_name": args.name, "source_run": args.run, "notes": args.notes,
    "n_parameters": n_params, "val_set_size": len(val_ds),
    **{k: v for k, v in result_data.items() if k != "history"},
}
with open(ARCHIVE_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Archived '{args.run}' as '{args.name}' -> {ARCHIVE_DIR}")