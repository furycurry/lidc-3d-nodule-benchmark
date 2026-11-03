import json, glob, os
import numpy as np
from scipy import stats

print("="*60)
print("1. VOLUME TERCILE RECONCILIATION & EXACT SAMPLE COUNTS")
print("="*60)
# Locate clinical/tercile summary
strat_files = glob.glob("**/clinical_stratification_summary.json", recursive=True) + \
              glob.glob("**/test_results_summary.json", recursive=True)
if strat_files:
    data = json.load(open(strat_files[0]))
    # Print sample counts if recorded
    if "tercile_counts" in data:
        print("Tercile N counts:", data["tercile_counts"])
    elif "counts" in data:
        print("Subgroup counts:", data["counts"])
else:
    print("Stratification JSON not found directly in working directory.")

# Mathematical verification of bin weighting vs Table 1:
# Let N_total = 434. If cutoffs are fixed voxels (<100, 100-500, >500),
# the distribution in LIDC is right-skewed (more small nodules).
print("\nChecking UNet3DPaper Baseline:")
d_small, d_med, d_large = 0.6900, 0.7401, 0.7684
target_table1 = 0.7303
# Solve for N_small, N_med, N_large that yields target_table1
# E.g., if N = [162, 142, 130]:
for ns in [145, 155, 160, 162, 165]:
    nl = int((target_table1 * 434 - ns * d_small - (434 - ns) / 2 * d_med) / (d_large - d_med))
    nm = 434 - ns - nl
    weighted = (ns * d_small + nm * d_med + nl * d_large) / 434
    print(f"  If N = [{ns}, {nm}, {nl}] -> Weighted Mean = {weighted:.4f} (Table 1 target: {target_table1})")

print("\n" + "="*60)
print("2. UNETR ADAPTIVE VS BASELINE PER-SEED AUDIT (NOISE FLOOR)")
print("="*60)
seeds = [42, 123, 456, 789, 2024]
base_dices = []
adapt_dices = []
for s in seeds:
    fb = glob.glob(f"**/baseline_unetr_seed{s}*.json", recursive=True)
    fa = glob.glob(f"**/boundary_adaptive_unetr_seed{s}*.json", recursive=True) or \
         glob.glob(f"**/*adaptive*unetr*seed{s}*.json", recursive=True)
    if fb and fa:
        db = json.load(open(fb[0])).get("test_dice", json.load(open(fb[0])).get("best_val_dice"))
        da = json.load(open(fa[0])).get("test_dice", json.load(open(fa[0])).get("best_val_dice"))
        base_dices.append(db)
        adapt_dices.append(da)
        print(f"Seed {s}: Baseline = {db:.4f} | Adaptive = {da:.4f} | Diff = {da - db:+.4f}")

if len(base_dices) == 5:
    diffs = np.array(adapt_dices) - np.array(base_dices)
    print(f"\nAll 5 seeds mean diff: {np.mean(diffs):+.4f}")
    # Assuming seed 42 (or whichever) was the 1 that fired:
    print(f"Paired t-test over 5 seeds: t = {stats.ttest_rel(adapt_dices, base_dices).statistic:.3f}, p = {stats.ttest_rel(adapt_dices, base_dices).pvalue:.4f}")

print("\n" + "="*60)
print("3. TRUNCATION AUDIT CHECK (UNet3DPaper)")
print("="*60)
for s in seeds:
    fo = glob.glob(f"**/*boundary_adaptive*seed{s}*.json", recursive=True)
    fr = glob.glob(f"**/*truncation*seed{s}*.json", recursive=True) or \
         glob.glob(f"**/*wide128*seed{s}*.json", recursive=True)
    if fo and fr:
        d_orig = json.load(open(fo[0])).get("best_val_dice")
        d_retrain = json.load(open(fr[0])).get("best_val_dice")
        print(f"Seed {s}: Original Val Dice = {d_orig:.4f} | Retrained = {d_retrain:.4f} | Diff = {d_retrain - d_orig:+.4f}")