"""Paired significance tests (t-test and Wilcoxon) across matched seeds, for one architecture.

Usage (from the repo root):
    python statistical_test.py                           # UNet3DPaper main matrix
    python statistical_test.py --architecture unetr
(The SDF-audit retrains are compared with the originals by compare_corrected_sdf.py.)
With 5 seeds the smallest possible two-sided Wilcoxon p is 0.0625.
"""
import argparse
import numpy as np
from scipy import stats
from results_io import load_runs

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", default="outputs/results")
ap.add_argument("--architecture", default="unet3d_paper")
ap.add_argument("--metrics", nargs="+", default=["best_val_dice", "best_val_hd95"])
args = ap.parse_args()

configs = load_runs(args.results_dir, args.architecture)
common = sorted(set.intersection(*[set(v) for v in configs.values()]))
print(f"Architecture: {args.architecture}")
print(f"Paired seeds available: {common}\n")
if len(common) < 2:
    raise SystemExit("Need at least 2 common seeds across all three configurations.")


def paired_test(name_a, name_b, metric):
    a = [configs[name_a][s][metric] for s in common]
    b = [configs[name_b][s][metric] for s in common]
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = float("nan")
    print(f"{name_a} vs {name_b} | {metric}")
    print(f"  {name_a}: {np.mean(a):.4f} ± {np.std(a):.4f}")
    print(f"  {name_b}: {np.mean(b):.4f} ± {np.std(b):.4f}")
    print(f"  paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"  wilcoxon: p={w_p:.4f}\n")


for metric in args.metrics:
    paired_test("baseline", "boundary_fixed", metric)
    paired_test("baseline", "boundary_adaptive", metric)
    paired_test("boundary_fixed", "boundary_adaptive", metric)
