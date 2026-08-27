#!/usr/bin/env python3
"""
analyze_test_results.py

One-look test-set analysis for the LIDC-IDRI 3D U-Net benchmark.

Produces:
  - Pairwise paired statistical tests (t-test + Wilcoxon) on test metrics
  - Validation-vs-test side-by-side table
  - Size-stratified test Dice table
  - Per-seed breakdown

Saves:
  - JSON summary: outputs/test_results/test_analysis_summary.json
  - Text report:  outputs/test_results/test_analysis_report.txt

Usage (on cluster, from repo root):
    python src/analyze_test_results.py

Dependencies: numpy, scipy, config.py (paths)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

# ------------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------------
# Cluster path for original training result JSONs (15 files):
#   /workspace/outputs/results/*.json
#
# Test evaluation output path (produced by evaluate_test_set.py):
#   /workspace/outputs/test_results/test_*.json
#
# This script reads from both directories and writes its own outputs
# into the test_results directory.
# ------------------------------------------------------------------
try:
    import config
    RESULTS_DIR = config.RESULTS_DIR
    OUTPUT_DIR = config.OUTPUT_DIR
except Exception:
    # Standalone fallback — assumes repo structure: repo/src/this_script.py
    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "outputs" / "results"
    OUTPUT_DIR = BASE_DIR / "outputs"

# Explicit cluster override if the default config paths differ
if Path("/workspace/outputs/results").exists():
    RESULTS_DIR = Path("/workspace/outputs/results")
if Path("/workspace/outputs").exists():
    OUTPUT_DIR = Path("/workspace/outputs")

TEST_RESULTS_DIR = OUTPUT_DIR / "test_results"
ANALYSIS_JSON = TEST_RESULTS_DIR / "test_analysis_summary.json"
REPORT_TXT = TEST_RESULTS_DIR / "test_analysis_report.txt"

# ------------------------------------------------------------------
# Constants matching the project experimental design
# ------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 2024]

# Experiment name prefixes as produced by train.py
CONFIGS = {
    "baseline":        "baseline_unet3d_paper",
    "boundary_fixed":  "boundary_unet3d_paper",
    "boundary_adaptive": "boundary_adaptive_unet3d_paper",
}

# Metrics for primary pairwise comparison (per handoff spec)
PRIMARY_METRICS = ["dice", "hd95", "iou", "assd"]

# All metrics to show in val-vs-test table (includes HD100 if available)
ALL_METRICS = ["dice", "hd95", "hd100", "iou", "assd"]

METRIC_LABELS = {
    "dice":  "Dice",
    "hd95":  "HD95",
    "hd100": "HD100",
    "iou":   "IoU",
    "assd":  "ASSD",
}

PAIRWISE_COMPARISONS = [
    ("baseline", "boundary_fixed"),
    ("baseline", "boundary_adaptive"),
    ("boundary_fixed", "boundary_adaptive"),
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_test_json(experiment_name):
    """Locate test result JSON written by evaluate_test_set.py."""
    exact = TEST_RESULTS_DIR / f"test_{experiment_name}.json"
    if exact.exists():
        return load_json(exact)
    # Fallback glob (defensive)
    candidates = list(TEST_RESULTS_DIR.glob(f"test_{experiment_name}*.json"))
    if candidates:
        return load_json(candidates[0])
    return None


def find_val_json(experiment_name):
    """Locate original training result JSON from outputs/results/."""
    exact = RESULTS_DIR / f"{experiment_name}.json"
    if exact.exists():
        return load_json(exact)
    candidates = list(RESULTS_DIR.glob(f"{experiment_name}*.json"))
    if candidates:
        return load_json(candidates[0])
    return None


def val_assd_last10(val_data):
    """
    ASSD was never pre-aggregated into a top-level summary field
    (unlike Dice/HD95). Compute mean of the last 10 validation epochs
    from the history list.
    """
    history = val_data.get("history", [])
    vals = [epoch.get("val_assd") for epoch in history if epoch.get("val_assd") is not None]
    if len(vals) < 10:
        return None
    return float(np.mean(vals[-10:]))


def extract_test_metric(test_data, metric):
    """
    Flexible metric extraction: handles several plausible key layouts
    produced by evaluate_test_set.py.
    """
    candidates = [
        f"test_{metric}_mean",
        f"mean_test_{metric}",
        f"test_{metric}",
        f"{metric}_mean",
        metric,
    ]
    for k in candidates:
        if k in test_data:
            v = test_data[k]
            return v.get("mean") if isinstance(v, dict) else v
    # Nested under "metrics" dict
    metrics = test_data.get("metrics", {})
    for k in candidates:
        if k in metrics:
            v = metrics[k]
            return v.get("mean") if isinstance(v, dict) else v
    return None


def extract_size_stratified(test_data):
    """Extract size-stratified Dice dict from test result JSON."""
    for key in ["size_stratified", "size_stratified_dice", "stratified_dice", "stratified"]:
        if key in test_data:
            return test_data[key]
    return None


def fmt_mean_std(values):
    """Population SD (ddof=0), consistent with project convention."""
    clean = [v for v in values if v is not None]
    if not clean:
        return "N/A"
    return f"{np.mean(clean):.4f}±{np.std(clean, ddof=0):.4f}"


def fmt_p(p):
    if p is None:
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def paired_stats(a, b):
    a = np.array([x for x in a if x is not None], dtype=float)
    b = np.array([x for x in b if x is not None], dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return None, None, None
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = float("nan")
    return float(t_stat), float(t_p), float(w_p)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 72)
    print(" LIDC-IDRI Test-Set Analysis  —  One-Look Evaluation")
    print("=" * 72)
    print(f"\nScanning original training results from: {RESULTS_DIR}")
    print(f"Scanning test evaluation results from:   {TEST_RESULTS_DIR}")
    print()

    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 1. Load test + validation data for all 15 runs
    # --------------------------------------------------------------
    test_values   = defaultdict(lambda: defaultdict(list))   # cfg -> metric -> [seed values]
    val_values    = defaultdict(lambda: defaultdict(list))
    size_strat    = defaultdict(lambda: defaultdict(list))   # cfg -> size -> [seed dice values]
    per_seed_rows = []
    missing_files = []

    for cfg_name, exp_prefix in CONFIGS.items():
        for seed in SEEDS:
            exp_name = f"{exp_prefix}_seed{seed}"

            tdata = find_test_json(exp_name)
            vdata = find_val_json(exp_name)

            if tdata is None:
                missing_files.append(f"test_{exp_name}.json")
                continue
            if vdata is None:
                missing_files.append(f"{exp_name}.json (val)")
                continue

            row = {"config": cfg_name, "seed": seed, "experiment": exp_name}

            # --- test metrics ---
            for m in ALL_METRICS:
                v = extract_test_metric(tdata, m)
                test_values[cfg_name][m].append(v)
                row[f"test_{m}"] = v

            # --- validation metrics ---
            vd = vdata.get("val_dice_mean_last10")
            vh = vdata.get("val_hd95_mean_last10")
            vi = vdata.get("final_val_iou") or vdata.get("val_iou_mean_last10")
            va = val_assd_last10(vdata)
            # HD100 on val may not exist pre-aggregated; try history last-10
            vh100 = None
            history = vdata.get("history", [])
            if history:
                h100_vals = [epoch.get("val_hd100") for epoch in history if epoch.get("val_hd100") is not None]
                if len(h100_vals) >= 10:
                    vh100 = float(np.mean(h100_vals[-10:]))

            val_values[cfg_name]["dice"].append(vd)
            val_values[cfg_name]["hd95"].append(vh)
            val_values[cfg_name]["hd100"].append(vh100)
            val_values[cfg_name]["iou"].append(vi)
            val_values[cfg_name]["assd"].append(va)

            row.update({
                "val_dice": vd, "val_hd95": vh, "val_hd100": vh100,
                "val_iou": vi, "val_assd": va,
            })

            # --- size-stratified test dice ---
            ss = extract_size_stratified(tdata)
            if ss:
                for sz in ["small", "medium", "large"]:
                    if sz not in ss:
                        continue
                    entry = ss[sz]
                    if isinstance(entry, dict):
                        dice_val = entry.get("dice_mean") or entry.get("mean_dice") or entry.get("dice") or entry.get("mean") or entry.get("mean")
                    else:
                        dice_val = entry
                    if dice_val is not None:
                        size_strat[cfg_name][sz].append(dice_val)
                        row[f"test_dice_{sz}"] = dice_val

            per_seed_rows.append(row)

    if missing_files:
        print("\n[WARNING] Missing result files — analysis will be incomplete:")
        for m in missing_files:
            print(f"  • {m}")
        print()

    # --------------------------------------------------------------
    # 2. Pairwise paired statistics (test set)
    # --------------------------------------------------------------
    report = []
    report.append("-" * 72)
    report.append("PAIRWISE PAIRED STATISTICAL TESTS  —  Test Set (N = 5 seeds)")
    report.append("-" * 72)

    pairwise_out = []

    for cfg_a, cfg_b in PAIRWISE_COMPARISONS:
        label_a = cfg_a.replace("_", " ").title()
        label_b = cfg_b.replace("_", " ").title()
        report.append(f"\n{label_a}  vs  {label_b}")
        report.append("-" * 52)

        pair_entry = {"comparison": f"{cfg_a}_vs_{cfg_b}", "metrics": {}}

        for m in PRIMARY_METRICS:
            a_vals = [v for v in test_values[cfg_a][m] if v is not None]
            b_vals = [v for v in test_values[cfg_b][m] if v is not None]

            if len(a_vals) != len(b_vals) or len(a_vals) < 2:
                report.append(f"  {METRIC_LABELS[m]:6s}: insufficient data")
                pair_entry["metrics"][m] = {"status": "insufficient_data"}
                continue

            t_stat, t_p, w_p = paired_stats(a_vals, b_vals)
            m_a, s_a = float(np.mean(a_vals)), float(np.std(a_vals, ddof=0))
            m_b, s_b = float(np.mean(b_vals)), float(np.std(b_vals, ddof=0))

            report.append(
                f"  {METRIC_LABELS[m]:6s}:  {m_a:.4f}±{s_a:.4f}  vs  {m_b:.4f}±{s_b:.4f}  |  "
                f"t-test p={fmt_p(t_p)}, Wilcoxon p={fmt_p(w_p)}"
            )
            pair_entry["metrics"][m] = {
                "mean_a": m_a, "std_a": s_a,
                "mean_b": m_b, "std_b": s_b,
                "t_stat": t_stat, "t_test_p": t_p, "wilcoxon_p": w_p,
            }

        pairwise_out.append(pair_entry)

    # --------------------------------------------------------------
    # 3. Validation vs. Test side-by-side
    # --------------------------------------------------------------
    report.append("\n" + "=" * 72)
    report.append("VALIDATION  vs  TEST  —  Mean±Std (Population) over 5 Seeds")
    report.append("=" * 72)
    report.append(
        f"{'Config':<20} {'Metric':<7} {'Validation':<18} {'Test':<18} {'Gap (T−V)':<12}"
    )
    report.append("-" * 72)

    val_vs_test_out = []

    for cfg_name in CONFIGS.keys():
        for m in ALL_METRICS:
            v_list = [v for v in val_values[cfg_name][m] if v is not None]
            t_list = [v for v in test_values[cfg_name][m] if v is not None]
            if not v_list or not t_list:
                continue
            v_m = np.mean(v_list)
            v_s = np.std(v_list, ddof=0)
            t_m = np.mean(t_list)
            t_s = np.std(t_list, ddof=0)
            gap = t_m - v_m

            report.append(
                f"{cfg_name:<20} {METRIC_LABELS[m]:<7} "
                f"{v_m:.4f}±{v_s:.4f}   {t_m:.4f}±{t_s:.4f}   {gap:+.4f}"
            )
            val_vs_test_out.append({
                "config": cfg_name, "metric": m,
                "val_mean": float(v_m), "val_std": float(v_s),
                "test_mean": float(t_m), "test_std": float(t_s),
                "gap": float(gap),
            })
        report.append("")  # blank line between configs

    # --------------------------------------------------------------
    # 4. Size-stratified test Dice
    # --------------------------------------------------------------
    report.append("=" * 72)
    report.append("SIZE-STRATIFIED TEST DICE  —  Mean±Std over 5 Seeds")
    report.append("=" * 72)
    report.append(f"{'Config':<20} {'Small':<18} {'Medium':<18} {'Large':<18}")
    report.append("-" * 72)

    size_strat_out = []
    for cfg_name in CONFIGS.keys():
        s_vals = [v for v in size_strat[cfg_name].get("small", []) if v is not None]
        m_vals = [v for v in size_strat[cfg_name].get("medium", []) if v is not None]
        l_vals = [v for v in size_strat[cfg_name].get("large", []) if v is not None]

        report.append(
            f"{cfg_name:<20} {fmt_mean_std(s_vals):<18} "
            f"{fmt_mean_std(m_vals):<18} {fmt_mean_std(l_vals):<18}"
        )
        size_strat_out.append({
            "config": cfg_name,
            "small":  {"mean": float(np.mean(s_vals)) if s_vals else None,
                       "std":  float(np.std(s_vals, ddof=0)) if s_vals else None,
                       "n": len(s_vals)},
            "medium": {"mean": float(np.mean(m_vals)) if m_vals else None,
                       "std":  float(np.std(m_vals, ddof=0)) if m_vals else None,
                       "n": len(m_vals)},
            "large":  {"mean": float(np.mean(l_vals)) if l_vals else None,
                       "std":  float(np.std(l_vals, ddof=0)) if l_vals else None,
                       "n": len(l_vals)},
        })

    # --------------------------------------------------------------
    # 5. Per-seed detail table
    # --------------------------------------------------------------
    report.append("\n" + "=" * 72)
    report.append("PER-SEED TEST RESULTS")
    report.append("=" * 72)

    for cfg_name in CONFIGS.keys():
        report.append(f"\n{cfg_name.upper().replace('_', ' ')}")
        report.append(
            f"{'Seed':>6} {'Dice':>10} {'HD95':>10} {'HD100':>10} {'IoU':>10} {'ASSD':>10}"
        )
        report.append("-" * 60)
        for row in per_seed_rows:
            if row["config"] != cfg_name:
                continue
            report.append(
                f"{row['seed']:>6}  "
                f"{row.get('test_dice', 0.0):>10.4f} "
                f"{row.get('test_hd95', 0.0):>10.4f} "
                f"{row.get('test_hd100', 0.0) or 0.0:>10.4f} "
                f"{row.get('test_iou', 0.0):>10.4f} "
                f"{row.get('test_assd', 0.0):>10.4f}"
            )

    # --------------------------------------------------------------
    # 6. Persist & print
    # --------------------------------------------------------------
    summary = {
        "pairwise_tests": pairwise_out,
        "validation_vs_test": val_vs_test_out,
        "size_stratified_test_dice": size_strat_out,
        "per_seed_records": per_seed_rows,
        "missing_files": missing_files,
    }

    with open(ANALYSIS_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    text = "\n".join(report)
    with open(REPORT_TXT, "w") as f:
        f.write(text + "\n")

    print(text)
    print(f"\n[OK] JSON summary  → {ANALYSIS_JSON}")
    print(f"[OK] Text report   → {REPORT_TXT}")

    if missing_files:
        print(f"\n[!] {len(missing_files)} file(s) missing — some tables may show 'N/A'.")

    print("\n" + "=" * 72)
    print(" ONE-LOOK DISCIPLINE")
    print("=" * 72)
    print(" These test-set numbers are final. Do not use them to select seeds,")
    print(" tune hyperparameters, or revise model / loss / architecture choices.")
    print(" Report them in the paper exactly as they appear above.")
    print("=" * 72)


if __name__ == "__main__":
    main()