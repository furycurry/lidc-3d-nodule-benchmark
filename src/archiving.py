import json
import shutil
from pathlib import Path
import torch
import numpy as np
import csv
import config
from lidc_dataset import LIDCDataset
from unet3d_paper import UNet3DPaper


def promote_to_archive(run_experiment_name, archive_name, notes=""):
    archive_dir = config.OUTPUT_DIR / "archive" / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    result_src = config.RESULTS_DIR / f"{run_experiment_name}.json"
    with open(result_src) as f:
        result_data = json.load(f)

    shutil.copy(result_src, archive_dir / "training_history.json")

    ckpt_src_dir = config.OUTPUT_DIR / "checkpoints" / run_experiment_name
    for ckpt_file in ckpt_src_dir.glob("*.pt"):
        shutil.copy(ckpt_file, archive_dir / ckpt_file.name)

    device = torch.device(config.DEVICE)
    with open(config.BASE_DIR / "train_val_test_split.json") as f:
        split_data = json.load(f)
    val_keys = split_data["val_keys"]

    use_sdf = result_data.get("use_boundary_loss", False)
    val_ds = LIDCDataset(keys=val_keys, is_train=False, use_sdf=use_sdf)

    model = UNet3DPaper(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(archive_dir / "best_dice_unet3d.pt", map_location=device)
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

    with open(archive_dir / "per_sample_val_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "dice", "tp", "fp", "fn", "gt_voxels"])
        writer.writeheader()
        writer.writerows(per_sample_records)

    qualitative_examples.sort(key=lambda x: x[1])
    picks = {"worst": qualitative_examples[0], "median": qualitative_examples[len(qualitative_examples) // 2],
             "best": qualitative_examples[-1]}
    for label, (key, dice, img, gt, pred) in picks.items():
        np.savez(archive_dir / f"qualitative_{label}_{key}.npz", image=img, ground_truth=gt, prediction=pred, dice=dice)

    n_params = sum(p.numel() for p in model.parameters())
    metadata = {
        "archived_name": archive_name, "source_run": run_experiment_name, "notes": notes,
        "n_parameters": n_params, "val_set_size": len(val_ds),
        **{k: v for k, v in result_data.items() if k != "history"},
    }
    with open(archive_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return archive_dir