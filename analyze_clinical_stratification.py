#!/usr/bin/env python3
"""
analyze_clinical_stratification.py

Stratifies held-out test predictions across clinical radiologist ratings
(Texture, Subtlety, Spiculation, Malignancy) for UNet3DPaper vs. UNETR.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

WORKSPACE = Path(".")
SPLIT_PATH = WORKSPACE / "train_val_test_split.json"
METADATA_PATH = WORKSPACE / "nodule_clinical_metadata.json"
TEST_RESULTS_DIR = WORKSPACE / "outputs" / "test_results"

OUTPUT_REPORT = TEST_RESULTS_DIR / "clinical_stratification_report.txt"
OUTPUT_JSON = TEST_RESULTS_DIR / "clinical_stratification_summary.json"

SEEDS = [42, 123, 456, 789, 2024]
CONFIGS = ["baseline", "boundary", "boundary_adaptive"]
ARCHITECTURES = ["unet3d_paper", "unetr"]

# -------------------------------------------------------------------------
# 1. Load Split and Metadata
# -------------------------------------------------------------------------
with open(SPLIT_PATH, "r") as f:
    split_data = json.load(f)
test_keys = split_data["test_keys"]
assert len(test_keys) == 434, f"Expected 434 test keys, found {len(test_keys)}"

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

# Verify all test keys exist in metadata
for k in test_keys:
    assert k in metadata, f"Key {k} missing from clinical metadata!"


# -------------------------------------------------------------------------
# 2. Define Clinical Bins
# -------------------------------------------------------------------------
def get_clinical_bins(key):
    meta = metadata[key]
    tex = meta["texture_mean"]
    sub = meta["subtlety_mean"]
    spic = meta["spiculation_mean"]
    mal = meta["malignancy_mean"]

    bins = {}

    # Texture (Opacity)
    if tex <= 2.0:
        bins["texture"] = "Non-Solid (GGN)"
    elif tex < 4.0:
        bins["texture"] = "Part-Solid"
    else:
        bins["texture"] = "Solid"

    # Subtlety
    if sub <= 2.5:
        bins["subtlety"] = "Subtle (<=2.5)"
    elif sub <= 3.5:
        bins["subtlety"] = "Moderate (2.5-3.5)"
    else:
        bins["subtlety"] = "Obvious (>3.5)"

    # Spiculation
    if spic <= 2.0:
        bins["spiculation"] = "Smooth (<=2.0)"
    else:
        bins["spiculation"] = "Spiculated (>2.0)"

    # Malignancy
    if mal <= 2.5:
        bins["malignancy"] = "Low Suspicion (<=2.5)"
    elif mal <= 3.5:
        bins["malignancy"] = "Indeterminate (2.5-3.5)"
    else:
        bins["malignancy"] = "High Suspicion (>3.5)"

    return bins


# Tally bin sizes across test set
bin_counts = defaultdict(lambda: defaultdict(int))
key_to_bins = {}
for k in test_keys:
    b = get_clinical_bins(k)
    key_to_bins[k] = b
    for axis, val in b.items():
        bin_counts[axis][val] += 1

# -------------------------------------------------------------------------
# 3. Load Test Evaluation JSONs
# -------------------------------------------------------------------------
# results[arch][config][seed] = list of 434 per-sample Dice floats
run_data = defaultdict(lambda: defaultdict(dict))

for arch in ARCHITECTURES:
    for cfg in CONFIGS:
        for seed in SEEDS:
            fname = f"test_{cfg}_{arch}_seed{seed}.json"
            fpath = TEST_RESULTS_DIR / fname
            if not fpath.exists():
                candidates = list(TEST_RESULTS_DIR.glob(f"test_{cfg}_{arch}_seed{seed}*.json"))
                if candidates:
                    fpath = candidates[0]
                else:
                    raise FileNotFoundError(f"Missing test result: {fname}")
            with open(fpath, "r") as f:
                d = json.load(f)
            psd = d["per_sample_dice"]
            assert len(psd) == 434, f"Length mismatch in {fname}: {len(psd)}"
            run_data[arch][cfg][seed] = psd

# -------------------------------------------------------------------------
# 4. Compute Stratified Metrics
# -------------------------------------------------------------------------
# Per-axis, per-bin, per-run mean Dice
stratified_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

for arch in ARCHITECTURES:
    for cfg in CONFIGS:
        for seed in SEEDS:
            scores = run_data[arch][cfg][seed]
            axis_accum = defaultdict(lambda: defaultdict(list))
            for k, score in zip(test_keys, scores):
                b = key_to_bins[k]
                for axis, val in b.items():
                    axis_accum[axis][val].append(score)

            for axis, val_dict in axis_accum.items():
                for val, score_list in val_dict.items():
                    stratified_scores[arch][cfg][axis][val].append(float(np.mean(score_list)))

# -------------------------------------------------------------------------
# 5. Format Output & Statistical Tests
# -------------------------------------------------------------------------
output_lines = []
output_lines.append("=" * 80)
output_lines.append("CLINICAL ATTRIBUTE STRATIFICATION REPORT (HELD-OUT TEST SET, N=434)")
output_lines.append("=" * 80)
output_lines.append("\n[1] Sample Distribution Across Test Partition:")
for axis in ["texture", "subtlety", "spiculation", "malignancy"]:
    output_lines.append(f"  Axis: {axis.upper()}")
    for val, count in sorted(bin_counts[axis].items()):
        output_lines.append(f"    - {val:26s}: N = {count:3d} ({count / 434 * 100:.1f}%)")


def paired_summary(a_list, b_list):
    a = np.array(a_list)
    b = np.array(b_list)
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


output_lines.append("\n" + "=" * 80)
output_lines.append("[2] Cross-Architecture Comparison (Baseline UNet3DPaper vs. Baseline UNETR):")
output_lines.append("=" * 80)

summary_dict = {"bin_counts": dict(bin_counts), "comparisons": {}}

for axis in ["texture", "subtlety", "spiculation", "malignancy"]:
    output_lines.append(f"\n--- {axis.upper()} ---")
    output_lines.append(
        f"{'Category':24s} | {'UNet3DPaper (N=5)':18s} | {'UNETR (N=5)':18s} | {'t-test p':10s} | {'Wilcoxon p':10s} | {'Cohen d':8s}")
    output_lines.append("-" * 96)
    for val in sorted(bin_counts[axis].keys()):
        cnn_vals = stratified_scores["unet3d_paper"]["baseline"][axis][val]
        unetr_vals = stratified_scores["unetr"]["baseline"][axis][val]
        res = paired_summary(cnn_vals, unetr_vals)
        p_str = "<0.001" if res["t_p"] < 0.001 else f"{res['t_p']:.4f}"
        w_str = "<0.001" if res["w_p"] < 0.001 else f"{res['w_p']:.4f}"
        output_lines.append(
            f"{val:24s} | {res['mean_a']:.4f} +- {res['std_a']:.4f}    | "
            f"{res['mean_b']:.4f} +- {res['std_b']:.4f}    | "
            f"{p_str:10s} | {w_str:10s} | {res['cohens_d']:+7.2f}"
        )
        summary_dict["comparisons"][f"{axis}_{val}"] = res

output_lines.append("\n" + "=" * 80)
output_lines.append("[3] Boundary Loss Impact on High-Complexity Margins (UNet3DPaper: Baseline vs Boundary):")
output_lines.append("=" * 80)
output_lines.append(
    f"{'Category':24s} | {'Baseline (N=5)':18s} | {'Boundary Fixed (N=5)':18s} | {'t-test p':10s} | {'Wilcoxon p':10s}")
output_lines.append("-" * 88)
for val in sorted(bin_counts["spiculation"].keys()):
    base_vals = stratified_scores["unet3d_paper"]["baseline"]["spiculation"][val]
    bnd_vals = stratified_scores["unet3d_paper"]["boundary"]["spiculation"][val]
    res = paired_summary(base_vals, bnd_vals)
    p_str = "<0.001" if res["t_p"] < 0.001 else f"{res['t_p']:.4f}"
    w_str = "<0.001" if res["w_p"] < 0.001 else f"{res['w_p']:.4f}"
    output_lines.append(
        f"{val:24s} | {res['mean_a']:.4f} +- {res['std_a']:.4f}    | "
        f"{res['mean_b']:.4f} +- {res['std_b']:.4f}    | "
        f"{p_str:10s} | {w_str:10s}"
    )

report_text = "\n".join(output_lines)
print(report_text)

with open(OUTPUT_REPORT, "w") as f:
    f.write(report_text)

with open(OUTPUT_JSON, "w") as f:
    json.dump(summary_dict, f, indent=2)

print(f"\n[SUCCESS] Reports written to:\n - {OUTPUT_REPORT}\n - {OUTPUT_JSON}")