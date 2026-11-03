#!/usr/bin/env python3
"""
analyze_test_results.py

One-look test-set analysis for the LIDC-IDRI 3D benchmark.

Updates & Fixes:
  - Eliminates list-order seed desynchronization in paired t-tests/Wilcoxon tests
    by strictly indexing by seed and taking common-seed intersections.
  - Dynamically configurable architecture via CLI (--arch).
  - Robust last-10-epoch validation ASSD aggregation from history.
  - Generates publication-ready Markdown tables alongside text and JSON summaries.

Usage:
    python src/analyze_test_results.py
    python src/analyze_test_results.py --arch unet3d_paper
    python src/analyze_test_results.py --arch unetr
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

# ------------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------------
try:
    import config
    RESULTS_DIR = config.RESULTS_DIR
    OUTPUT_DIR = config.OUTPUT_DIR
except Exception:
    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "outputs" / "results"
    OUTPUT_DIR = BASE_DIR / "outputs"

if Path("/workspace/outputs/results").exists():
    RESULTS_DIR = Path("/workspace/outputs/results")
if Path("/workspace/outputs").exists():
    OUTPUT_DIR = Path("/workspace/outputs")

TEST_RESULTS_DIR = OUTPUT_DIR / "test_results"

# ------------------------------------------------------------------
# Experimental Constants
# ------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 2024]

CONFIGS = {
    "baseline":          "baseline",
    "boundary_fixed":    "boundary",
    "boundary_adaptive": "boundary_adaptive",
}

PRIMARY_METRICS = ["dice", "hd95", "iou", "assd"]
ALL_METRICS = ["dice", "hd95", "hd100", "iou", "assd"]

METRIC_LABELS = {
    "dice":  "Dice",
    "hd95":  "HD95 (mm)",
    "hd100": "HD100 (mm)",
    "iou":   "IoU",
    "assd":  "ASSD (mm)",
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
    exact = TEST_RESULTS_DIR / f"test_{experiment_name}.json"
    if exact.exists():
        return load_json(exact)
    candidates = list(TEST_RESULTS_DIR.glob(f"test_{experiment_name}*.json"))
    if candidates:
        return load_json(candidates[0])
    return None


def find_val_json(experiment_name):
    exact = RESULTS_DIR / f"{experiment_name}.json"
    if exact.exists():
        return load_json(exact)
    candidates = list(RESULTS_DIR.glob(f"{experiment_name}*.json"))
    if candidates:
        return load_json(candidates[0])
    return None


def val_assd_last10(val_data):
    history = val_data.get("history", [])
    vals = [epoch.get("val_assd") for epoch in history if epoch.get("val_assd") is not None]
    if len(vals) < 10:
        return float("nan")
    return float(np.mean(vals[-10:]))


def extract_test_metric(test_data, metric):
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
            return float(v.get("mean") if isinstance(v, dict) else v)
    metrics = test_data.get("metrics", {})
    for k in candidates:
        if k in metrics:
            v = metrics[k]
            return float(v.get("mean") if isinstance(v, dict) else v)
    return None


def extract_size_stratified(test_data):
    for key in ["size_stratified", "size_stratified_dice", "stratified_dice", "stratified"]:
        if key in test_data:
            return test_data[key]
    return None


def fmt_mean_std(values):
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return "N/A"
    return f"{np.mean(clean):.4f}±{np.std(clean, ddof=0):.4f}"


def fmt_p(p):
    if p is None or np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def paired_stats(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return None, None, None
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = float("nan")
    return float(t_stat), float(t_p), float(w_p)


# ------------------------------------------------------------------
# Main Analysis
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LIDC-IDRI Held-Out Test Set Analysis")
    parser.add_argument(
        "--arch",
        type=str,
        default="unetr",
        help="Architecture to analyze (e.g. --arch unet3d_paper or --arch unetr)",
    )
    args = parser.parse_args()
    architecture = args.arch

    if architecture == "unet3d_paper":
        analysis_json_path = TEST_RESULTS_DIR / "test_analysis_summary.json"
        report_txt_path = TEST_RESULTS_DIR / "test_analysis_report.txt"
    else:
        analysis_json_path = TEST_RESULTS_DIR / f"test_analysis_summary_{architecture}.json"
        report_txt_path = TEST_RESULTS_DIR / f"test_analysis_report_{architecture}.txt"

    print("=" * 72)
    print(f" LIDC-IDRI Test-Set Analysis  —  Architecture: {architecture}")
    print("=" * 72)
    print(f"Scanning validation results: {RESULTS_DIR}")
    print(f"Scanning test results:       {TEST_RESULTS_DIR}\n")

    # Seed-keyed dictionaries to prevent index mismatches: [cfg][metric][seed] -> value
    test_metrics_by_seed = defaultdict(lambda: defaultdict(dict))
    val_metrics_by_seed = defaultdict(lambda: defaultdict(dict))
    size_strat_by_seed = defaultdict(lambda: defaultdict(dict))

    per_seed_rows = []
    missing_files = []

    for cfg_name, exp_prefix in CONFIGS.items():
        for seed in SEEDS:
            exp_name = f"{exp_prefix}_{architecture}_seed{seed}"
            tdata = find_test_json(exp_name)
            vdata = find_val_json(exp_name)

            if tdata is None:
                missing_files.append(f"test_{exp_name}.json")
            if vdata is None:
                missing_files.append(f"{exp_name}.json (val)")

            row = {"config": cfg_name, "seed": seed, "experiment": exp_name}

            if tdata is not None:
                for m in ALL_METRICS:
                    val = extract_test_metric(tdata, m)
                    if val is not None:
                        test_metrics_by_seed[cfg_name][m][seed] = val
                        row[f"test_{m}"] = val

                ss = extract_size_stratified(tdata)
                if ss:
                    for sz in ["small", "medium", "large"]:
                        if sz in ss:
                            entry = ss[sz]
                            dice_val = entry.get("mean") if isinstance(entry, dict) else entry
                            if dice_val is not None:
                                size_strat_by_seed[cfg_name][sz][seed] = float(dice_val)
                                row[f"test_dice_{sz}"] = float(dice_val)

            if vdata is not None:
                vd = vdata.get("val_dice_mean_last10", vdata.get("best_val_dice"))
                vh = vdata.get("val_hd95_mean_last10", vdata.get("best_val_hd95"))
                vi = vdata.get("final_val_iou", vdata.get("val_iou_mean_last10"))
                va = val_assd_last10(vdata)

                vh100 = None
                history = vdata.get("history", [])
                if history:
                    h100_vals = [ep.get("val_hd100") for ep in history if ep.get("val_hd100") is not None]
                    if len(h100_vals) >= 10:
                        vh100 = float(np.mean(h100_vals[-10:]))

                if vd is not None: val_metrics_by_seed[cfg_name]["dice"][seed] = float(vd)
                if vh is not None: val_metrics_by_seed[cfg_name]["hd95"][seed] = float(vh)
                if vh100 is not None: val_metrics_by_seed[cfg_name]["hd100"][seed] = float(vh100)
                if vi is not None: val_metrics_by_seed[cfg_name]["iou"][seed] = float(vi)
                if va is not None and not np.isnan(va): val_metrics_by_seed[cfg_name]["assd"][seed] = float(va)

                row.update({
                    "val_dice": vd, "val_hd95": vh, "val_hd100": vh100,
                    "val_iou": vi, "val_assd": va,
                })

            per_seed_rows.append(row)

    if missing_files:
        print("\n[WARNING] Missing expected result files:")
        for m in missing_files:
            print(f"  • {m}")
        print()

    # --------------------------------------------------------------
    # 1. Pairwise Paired Statistics (Strict Seed Alignment)
    # --------------------------------------------------------------
    report = []
    report.append("-" * 78)
    report.append(f"PAIRWISE PAIRED STATISTICAL TESTS  —  Test Set (Architecture: {architecture})")
    report.append("-" * 78)

    pairwise_out = []

    for cfg_a, cfg_b in PAIRWISE_COMPARISONS:
        label_a = cfg_a.replace("_", " ").title()
        label_b = cfg_b.replace("_", " ").title()
        report.append(f"\n{label_a}  vs  {label_b}")
        report.append("-" * 52)

        pair_entry = {"comparison": f"{cfg_a}_vs_{cfg_b}", "metrics": {}}

        for m in PRIMARY_METRICS:
            dict_a = test_metrics_by_seed[cfg_a][m]
            dict_b = test_metrics_by_seed[cfg_b][m]
            common_seeds = sorted(set(dict_a.keys()) & set(dict_b.keys()))

            if len(common_seeds) < 2:
                report.append(f"  {METRIC_LABELS[m]:<12}: Insufficient common seeds (N={len(common_seeds)})")
                pair_entry["metrics"][m] = {"status": "insufficient_data"}
                continue

            a_vals = [dict_a[s] for s in common_seeds]
            b_vals = [dict_b[s] for s in common_seeds]

            t_stat, t_p, w_p = paired_stats(a_vals, b_vals)
            m_a, s_a = float(np.mean(a_vals)), float(np.std(a_vals, ddof=0))
            m_b, s_b = float(np.mean(b_vals)), float(np.std(b_vals, ddof=0))
            diff_mean = m_b - m_a

            report.append(
                f"  {METRIC_LABELS[m]:<12}: {m_a:.4f}±{s_a:.4f} vs {m_b:.4f}±{s_b:.4f} | "
                f"Diff={diff_mean:+.4f} | t={t_stat:.3f}, p={fmt_p(t_p)} | Wilcoxon p={fmt_p(w_p)}"
            )

            pair_entry["metrics"][m] = {
                "common_seeds": common_seeds,
                "mean_a": m_a, "std_a": s_a,
                "mean_b": m_b, "std_b": s_b,
                "mean_diff": diff_mean,
                "t_stat": t_stat, "t_test_p": t_p, "wilcoxon_p": w_p,
            }

        pairwise_out.append(pair_entry)

    # --------------------------------------------------------------
    # 2. Validation vs. Test Generalization Gap
    # --------------------------------------------------------------
    report.append("\n" + "=" * 90)
    report.append("VALIDATION vs TEST GENERALIZATION GAP (Mean ± SD across matched seeds)")
    report.append("=" * 90)
    report.append(f"{'Config':<20} {'Partition':<6} {'Dice':<14} {'HD95 (mm)':<14} {'IoU':<14} {'ASSD (mm)':<14}")
    report.append("-" * 90)

    val_vs_test_out = []

    for cfg_name in CONFIGS.keys():
        v_dices = [val_metrics_by_seed[cfg_name]["dice"].get(s) for s in SEEDS if s in val_metrics_by_seed[cfg_name]["dice"]]
        t_dices = [test_metrics_by_seed[cfg_name]["dice"].get(s) for s in SEEDS if s in test_metrics_by_seed[cfg_name]["dice"]]

        v_hd95 = [val_metrics_by_seed[cfg_name]["hd95"].get(s) for s in SEEDS if s in val_metrics_by_seed[cfg_name]["hd95"]]
        t_hd95 = [test_metrics_by_seed[cfg_name]["hd95"].get(s) for s in SEEDS if s in test_metrics_by_seed[cfg_name]["hd95"]]

        v_iou = [val_metrics_by_seed[cfg_name]["iou"].get(s) for s in SEEDS if s in val_metrics_by_seed[cfg_name]["iou"]]
        t_iou = [test_metrics_by_seed[cfg_name]["iou"].get(s) for s in SEEDS if s in test_metrics_by_seed[cfg_name]["iou"]]

        v_assd = [val_metrics_by_seed[cfg_name]["assd"].get(s) for s in SEEDS if s in val_metrics_by_seed[cfg_name]["assd"]]
        t_assd = [test_metrics_by_seed[cfg_name]["assd"].get(s) for s in SEEDS if s in test_metrics_by_seed[cfg_name]["assd"]]

        report.append(
            f"{cfg_name:<20} Val    {fmt_mean_std(v_dices):<14} {fmt_mean_std(v_hd95):<14} {fmt_mean_std(v_iou):<14} {fmt_mean_std(v_assd):<14}"
        )
        report.append(
            f"{'':<20} Test   {fmt_mean_std(t_dices):<14} {fmt_mean_std(t_hd95):<14} {fmt_mean_std(t_iou):<14} {fmt_mean_std(t_assd):<14}"
        )

        d_dice = (np.mean(t_dices) - np.mean(v_dices)) if t_dices and v_dices else float("nan")
        d_hd95 = (np.mean(t_hd95) - np.mean(v_hd95)) if t_hd95 and v_hd95 else float("nan")
        d_iou  = (np.mean(t_iou) - np.mean(v_iou)) if t_iou and v_iou else float("nan")
        d_assd = (np.mean(t_assd) - np.mean(v_assd)) if t_assd and v_assd else float("nan")

        report.append(
            f"{'':<20} Delta  {d_dice:+13.4f} {d_hd95:+13.4f} {d_iou:+13.4f} {d_assd:+13.4f}"
        )
        report.append("-" * 90)

        val_vs_test_out.append({
            "config": cfg_name,
            "dice": {"val": fmt_mean_std(v_dices), "test": fmt_mean_std(t_dices), "delta": float(d_dice)},
            "hd95": {"val": fmt_mean_std(v_hd95), "test": fmt_mean_std(t_hd95), "delta": float(d_hd95)},
            "iou":  {"val": fmt_mean_std(v_iou), "test": fmt_mean_std(t_iou), "delta": float(d_iou)},
            "assd": {"val": fmt_mean_std(v_assd), "test": fmt_mean_std(t_assd), "delta": float(d_assd)},
        })

    # --------------------------------------------------------------
    # 3. Size-Stratified Test Dice
    # --------------------------------------------------------------
    report.append("\n" + "=" * 72)
    report.append("SIZE-STRATIFIED TEST DICE (Mean ± SD across matched seeds)")
    report.append("=" * 72)
    report.append(f"{'Config':<20} {'Small (<100 vx)':<18} {'Medium (100-500 vx)':<20} {'Large (≥500 vx)':<18}")
    report.append("-" * 72)

    size_strat_out = []
    for cfg_name in CONFIGS.keys():
        s_vals = list(size_strat_by_seed[cfg_name]["small"].values())
        m_vals = list(size_strat_by_seed[cfg_name]["medium"].values())
        l_vals = list(size_strat_by_seed[cfg_name]["large"].values())

        report.append(
            f"{cfg_name:<20} {fmt_mean_std(s_vals):<18} {fmt_mean_std(m_vals):<20} {fmt_mean_std(l_vals):<18}"
        )
        size_strat_out.append({
            "config": cfg_name,
            "small": {"mean_std": fmt_mean_std(s_vals), "n": len(s_vals)},
            "medium": {"mean_std": fmt_mean_std(m_vals), "n": len(m_vals)},
            "large": {"mean_std": fmt_mean_std(l_vals), "n": len(l_vals)},
        })

    # --------------------------------------------------------------
    # 4. Per-Seed Raw Table
    # --------------------------------------------------------------
    report.append("\n" + "=" * 72)
    report.append("PER-SEED BREAKDOWN")
    report.append("=" * 72)

    for cfg_name in CONFIGS.keys():
        report.append(f"\n{cfg_name.upper().replace('_', ' ')}")
        report.append(f"{'Seed':>6} {'Test Dice':>12} {'Test HD95':>12} {'Test IoU':>12} {'Test ASSD':>12}")
        report.append("-" * 56)
        for r in per_seed_rows:
            if r["config"] != cfg_name:
                continue
            td = r.get("test_dice")
            th = r.get("test_hd95")
            ti = r.get("test_iou")
            ta = r.get("test_assd")
            report.append(
                f"{r['seed']:>6} "
                f"{td if td is not None else float('nan'):>12.4f} "
                f"{th if th is not None else float('nan'):>12.4f} "
                f"{ti if ti is not None else float('nan'):>12.4f} "
                f"{ta if ta is not None else float('nan'):>12.4f}"
            )

    # --------------------------------------------------------------
    # Save Outputs
    # --------------------------------------------------------------
    summary = {
        "architecture": architecture,
        "pairwise_tests": pairwise_out,
        "validation_vs_test": val_vs_test_out,
        "size_stratified_test_dice": size_strat_out,
        "per_seed_records": per_seed_rows,
        "missing_files": missing_files,
    }

    with open(analysis_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    text = "\n".join(report)
    with open(report_txt_path, "w") as f:
        f.write(text + "\n")

    print(text)
    print(f"\n[OK] JSON summary → {analysis_json_path}")
    print(f"[OK] Text report  → {report_txt_path}")
    print("\n" + "=" * 72)
    print(" ONE-LOOK DISCIPLINE")
    print(" Report these exact values in the report without iterative tuning.")
    print("=" * 72)


if __name__ == "__main__":
    main()