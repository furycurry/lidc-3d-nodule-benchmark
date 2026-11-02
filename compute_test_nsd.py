#!/usr/bin/env python3
"""
compute_test_nsd.py

Evaluates Normalized Surface Distance (NSD) at tau=1.0mm and tau=2.0mm
across all 30 held-out test checkpoints (15 UNet3DPaper + 15 UNETR).
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
from monai.metrics import compute_surface_dice

import config
from lidc_dataset import LIDCDataset
from unet3d_paper import UNet3DPaper
from unetr_paper import UNETRPaper

WORKSPACE = Path("/workspace")
TEST_RESULTS_DIR = WORKSPACE / "outputs" / "test_results"
CHECKPOINTS_DIR = WORKSPACE / "outputs" / "checkpoints"
SPLIT_PATH = WORKSPACE / "train_val_test_split.json"

OUTPUT_REPORT = TEST_RESULTS_DIR / "test_nsd_report.txt"
OUTPUT_JSON = TEST_RESULTS_DIR / "test_nsd_summary.json"

SEEDS = [42, 123, 456, 789, 2024]
CONFIGS = ["baseline", "boundary", "boundary_adaptive"]
ARCHITECTURES = ["unet3d_paper", "unetr"]
THRESHOLDS = [1.0, 2.0]

# 1. Dataset & DataLoader Setup
with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)
test_keys = split_data["test_keys"]
assert len(test_keys) == 434, f"Expected 434 test keys, found {len(test_keys)}"

test_ds = LIDCDataset(keys=test_keys, is_train=False, use_sdf=False)
test_loader = DataLoader(
    test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Computing NSD on device: {device} (Device Name: {torch.cuda.get_device_name(0)})")

# 2. Evaluation Loop
nsd_results = defaultdict(lambda: defaultdict(dict))

for arch in ARCHITECTURES:
    for cfg in CONFIGS:
        for seed in SEEDS:
            exp_name = f"{cfg}_{arch}_seed{seed}"
            ckpt_path = CHECKPOINTS_DIR / exp_name / "best_dice_unet3d.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            if arch == "unet3d_paper":
                model = UNet3DPaper(in_channels=1, out_channels=1).to(device)
            elif arch == "unetr":
                model = UNETRPaper(in_channels=1, out_channels=1, img_size=config.PATCH_SIZE[0]).to(device)

            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            sample_nsd_1mm = []
            sample_nsd_2mm = []

            with torch.no_grad():
                for patches, masks in test_loader:
                    patches, masks = patches.to(device), masks.to(device)
                    outputs = model(patches)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    for i in range(preds.shape[0]):
                        p = preds[i : i + 1]
                        m = masks[i : i + 1]

                        pred_fg = bool(torch.any(p > 0))
                        mask_fg = bool(torch.any(m > 0))

                        if not pred_fg and not mask_fg:
                            score_1 = 1.0
                            score_2 = 1.0
                        elif not pred_fg or not mask_fg:
                            score_1 = 0.0
                            score_2 = 0.0
                        else:
                            sd_1 = compute_surface_dice(
                                y_pred=p, y=m, class_thresholds=[1.0], spacing=[1.0, 1.0, 1.0]
                            ).item()
                            sd_2 = compute_surface_dice(
                                y_pred=p, y=m, class_thresholds=[2.0], spacing=[1.0, 1.0, 1.0]
                            ).item()
                            score_1 = 0.0 if np.isnan(sd_1) else float(sd_1)
                            score_2 = 0.0 if np.isnan(sd_2) else float(sd_2)

                        sample_nsd_1mm.append(score_1)
                        sample_nsd_2mm.append(score_2)

            mean_1 = float(np.mean(sample_nsd_1mm))
            mean_2 = float(np.mean(sample_nsd_2mm))
            nsd_results[arch][cfg][seed] = {"nsd_1mm": mean_1, "nsd_2mm": mean_2}
            print(f"[{exp_name}] NSD@1mm: {mean_1:.4f} | NSD@2mm: {mean_2:.4f}")

# 3. Statistical Analysis & Reporting
def paired_stats(a_vals, b_vals):
    a = np.array(a_vals)
    b = np.array(b_vals)
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except Exception:
        w_p = float("nan")
    d = float(np.mean(diff) / np.std(diff, ddof=0)) if np.std(diff, ddof=0) > 0 else 0.0
    return {
        "mean_a": float(np.mean(a)), "std_a": float(np.std(a, ddof=0)),
        "mean_b": float(np.mean(b)), "std_b": float(np.std(b, ddof=0)),
        "t_p": float(t_p), "w_p": float(w_p), "cohens_d": d
    }

lines = []
lines.append("=" * 80)
lines.append("NORMALIZED SURFACE DISTANCE (NSD) REPORT — HELD-OUT TEST SET (N=434)")
lines.append("=" * 80)

lines.append("\n[1] UNet3DPaper (CNN) — Boundary Loss Comparisons (N=5 seeds):")
lines.append(f"{'Metric':10s} | {'Baseline':16s} | {'Boundary Fixed':16s} | {'t-test p':10s} | {'Wilcoxon p':10s}")
lines.append("-" * 72)
for m_key, m_label in [("nsd_1mm", "NSD@1.0mm"), ("nsd_2mm", "NSD@2.0mm")]:
    base = [nsd_results["unet3d_paper"]["baseline"][s][m_key] for s in SEEDS]
    bnd = [nsd_results["unet3d_paper"]["boundary"][s][m_key] for s in SEEDS]
    res = paired_stats(base, bnd)
    lines.append(
        f"{m_label:10s} | {res['mean_a']:.4f} +- {res['std_a']:.4f}  | {res['mean_b']:.4f} +- {res['std_b']:.4f}  | "
        f"{res['t_p']:.4f}     | {res['w_p']:.4f}"
    )

lines.append("\n[2] UNETR (Transformer) — Boundary Loss Comparisons (N=5 seeds):")
lines.append(f"{'Metric':10s} | {'Baseline':16s} | {'Boundary Fixed':16s} | {'t-test p':10s} | {'Wilcoxon p':10s}")
lines.append("-" * 72)
for m_key, m_label in [("nsd_1mm", "NSD@1.0mm"), ("nsd_2mm", "NSD@2.0mm")]:
    base = [nsd_results["unetr"]["baseline"][s][m_key] for s in SEEDS]
    bnd = [nsd_results["unetr"]["boundary"][s][m_key] for s in SEEDS]
    res = paired_stats(base, bnd)
    lines.append(
        f"{m_label:10s} | {res['mean_a']:.4f} +- {res['std_a']:.4f}  | {res['mean_b']:.4f} +- {res['std_b']:.4f}  | "
        f"{res['t_p']:.4f}     | {res['w_p']:.4f}"
    )

lines.append("\n[3] Cross-Architecture Comparison (Pooled N=15, UNet3DPaper vs. UNETR):")
lines.append(f"{'Metric':10s} | {'UNet3DPaper (N=15)':20s} | {'UNETR (N=15)':20s} | {'t-test p':10s} | {'Wilcoxon p':10s} | {'Cohen d':8s}")
lines.append("-" * 90)
for m_key, m_label in [("nsd_1mm", "NSD@1.0mm"), ("nsd_2mm", "NSD@2.0mm")]:
    cnn_pooled = [nsd_results["unet3d_paper"][c][s][m_key] for c in CONFIGS for s in SEEDS]
    unetr_pooled = [nsd_results["unetr"][c][s][m_key] for c in CONFIGS for s in SEEDS]
    res = paired_stats(cnn_pooled, unetr_pooled)
    p_str = "<0.001" if res["t_p"] < 0.001 else f"{res['t_p']:.4f}"
    w_str = "<0.001" if res["w_p"] < 0.001 else f"{res['w_p']:.4f}"
    lines.append(
        f"{m_label:10s} | {res['mean_a']:.4f} +- {res['std_a']:.4f}         | {res['mean_b']:.4f} +- {res['std_b']:.4f}         | "
        f"{p_str:10s} | {w_str:10s} | {res['cohens_d']:+7.2f}"
    )

report_text = "\n".join(lines)
print(report_text)

with open(OUTPUT_REPORT, "w") as f:
    f.write(report_text)

with open(OUTPUT_JSON, "w") as f:
    json.dump(nsd_results, f, indent=2)

print(f"\n[SUCCESS] NSD reports written to:\n - {OUTPUT_REPORT}\n - {OUTPUT_JSON}")