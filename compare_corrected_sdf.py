#!/usr/bin/env python3
"""
Compare original vs corrected-SDF boundary-loss results using the same
paired statistical methodology as the original null-result analysis:
paired t-test + Wilcoxon signed-rank test across the 5 matched seeds,
on Dice, HD95, and IoU -- for each config (fixed-schedule, adaptive)
separately.

Usage:
    python compare_corrected_sdf.py --results-dir /workspace/outputs/results
(adjust --original-suffix if the original 10 files live under a different
naming pattern, e.g. if they were moved into an archive directory)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = [42, 123, 456, 789, 2024]
CONFIGS = {
    "fixed_schedule": "boundary",
    "adaptive": "boundary_adaptive",
}
METRICS = {
    "val_dice_mean_last10": "Dice",
    "val_hd95_mean_last10": "HD95",
    "final_val_iou": "IoU",
}


def load_metric(results_dir, loss_tag, seed, suffix, metric_key, architecture):
    fname = f"{loss_tag}_{architecture}_seed{seed}{suffix}.json"
    fpath = Path(results_dir) / fname
    if not fpath.exists():
        return None, fname
    with open(fpath) as f:
        data = json.load(f)
    return data.get(metric_key), fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--original-suffix", default="")
    ap.add_argument("--corrected-suffix", default="_correctedSDF")
    ap.add_argument("--architecture", default="unet3d_paper")
    args = ap.parse_args()

    for config_label, loss_tag in CONFIGS.items():
        print("=" * 78)
        print(f"CONFIG: {config_label}  (loss_tag={loss_tag})")
        print("=" * 78)

        for metric_key, metric_label in METRICS.items():
            orig_vals, corr_vals = [], []
            missing = []

            for seed in SEEDS:
                ov, ofname = load_metric(args.results_dir, loss_tag, seed, args.original_suffix, metric_key, args.architecture)
                cv, cfname = load_metric(args.results_dir, loss_tag, seed, args.corrected_suffix, metric_key, args.architecture)
                if ov is None:
                    missing.append(ofname)
                if cv is None:
                    missing.append(cfname)
                if ov is not None and cv is not None:
                    orig_vals.append(ov)
                    corr_vals.append(cv)

            if missing:
                print(f"  [{metric_label}] MISSING FILES, skipping this metric: {missing}")
                continue

            orig_vals = np.array(orig_vals)
            corr_vals = np.array(corr_vals)
            diffs = corr_vals - orig_vals

            t_stat, t_p = stats.ttest_rel(corr_vals, orig_vals)
            try:
                w_stat, w_p = stats.wilcoxon(corr_vals, orig_vals)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")

            print(f"\n  [{metric_label}]")
            print(f"    original:  mean={orig_vals.mean():.4f}  std={orig_vals.std():.4f}  values={orig_vals.tolist()}")
            print(f"    corrected: mean={corr_vals.mean():.4f}  std={corr_vals.std():.4f}  values={corr_vals.tolist()}")
            print(f"    mean diff (corrected - original): {diffs.mean():+.4f}")
            print(f"    paired t-test:        t={t_stat:.4f}  p={t_p:.4f}  {'SIGNIFICANT (p<0.05)' if t_p < 0.05 else 'not significant'}")
            print(f"    wilcoxon signed-rank: W={w_stat:.4f}  p={w_p:.4f}  {'SIGNIFICANT (p<0.05)' if w_p < 0.05 else 'not significant'}")

        print()


if __name__ == "__main__":
    main()