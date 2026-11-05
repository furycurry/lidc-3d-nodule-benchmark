"""Aggregate multi-seed validation results for one architecture.

Usage (from the repo root):
    python aggregate_results.py                          # UNet3DPaper main matrix
    python aggregate_results.py --architecture unetr
    python aggregate_results.py --tag correctedSDF       # SDF-audit retrains
All +/- values are population standard deviations over seeds.
"""
import argparse
import numpy as np
from results_io import load_runs

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default="outputs/results")
ap.add_argument("--architecture", default="unet3d_paper")
ap.add_argument("--tag", default="")
args = ap.parse_args()

runs = load_runs(args.results_dir, args.architecture, args.tag)
print(f"Architecture: {args.architecture}   tag: {args.tag or '(main matrix)'}\n")
print(f"{'Config':<18} {'N':<4} {'Dice (mean±std)':<20} {'HD95 (mean±std)':<20}")
print("-" * 66)
for key, by_seed in runs.items():
    if not by_seed:
        continue
    seeds = sorted(by_seed)
    dices = [by_seed[s]["best_val_dice"] for s in seeds]
    hd95s = [by_seed[s]["best_val_hd95"] for s in seeds]
    print(f"{key:<18} {len(seeds):<4} {np.mean(dices):.4f}±{np.std(dices):.4f}        "
          f"{np.mean(hd95s):.4f}±{np.std(hd95s):.4f}")
    print(f"  Per-seed Dice: {list(zip(seeds, [round(d, 4) for d in dices]))}")
    print(f"  Per-seed HD95: {list(zip(seeds, [round(h, 4) for h in hd95s]))}")
