#!/usr/bin/env python3
"""
make_figures.py

Generates all report figures directly from the raw per-seed result JSONs in
outputs/results/ and outputs/test_results/ — no hand-typed numbers. Each
figure maps to a specific section of docs/methodology.md,
docs/statistical_analysis.md, or docs/gradient_analysis.md, noted in each
function's docstring.

Usage (from repo root):
    python scripts/make_figures.py                  # generate all figures
    python scripts/make_figures.py --only val_null   # generate just one
    python scripts/make_figures.py --list            # list figure names

Output: docs/figures/*.png

Dependencies: matplotlib, numpy, config.py (for paths; falls back to
../outputs/results and ../outputs/test_results relative to this script if
config.py isn't importable)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Path resolution — mirrors analyze_test_results.py's fallback logic
# --------------------------------------------------------------------------
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
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

# --------------------------------------------------------------------------
# Experimental design constants — must match train.py / evaluate_test_set.py
# --------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 2024]
CONFIGS = {
    "baseline":          "baseline_unet3d_paper",
    "boundary_fixed":    "boundary_unet3d_paper",
    "boundary_adaptive": "boundary_adaptive_unet3d_paper",
}
CONFIG_LABELS = {
    "baseline": "Baseline",
    "boundary_fixed": "Fixed\nSchedule",
    "boundary_adaptive": "Adaptive\nGated",
}
# Only the two boundary-loss configs have SDF-corrected retrain variants —
# baseline doesn't use SDF at all.
SDF_CONFIGS = ["boundary_fixed", "boundary_adaptive"]

COLORS = {
    "baseline": "#2c5f8a",
    "boundary_fixed": "#c9691f",
    "boundary_adaptive": "#3a8a5f",
    "validation": "#2c5f8a",
    "test": "#c9691f",
    "original": "#2c5f8a",
    "corrected": "#c9691f",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})


# --------------------------------------------------------------------------
# JSON loading helpers
# --------------------------------------------------------------------------
def _load(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_original(cfg_name, seed):
    """Original (uncorrected) per-seed training result JSON."""
    prefix = CONFIGS[cfg_name]
    return _load(RESULTS_DIR / f"{prefix}_seed{seed}.json")


def load_corrected(cfg_name, seed, correction):
    """
    correction: 'correctedSDF' (truncation fix) or 'rotationCorrected'.
    Only valid for boundary_fixed / boundary_adaptive.
    """
    prefix = CONFIGS[cfg_name]
    return _load(RESULTS_DIR / f"{prefix}_seed{seed}_{correction}.json")


def load_test(cfg_name, seed):
    prefix = CONFIGS[cfg_name]
    return _load(TEST_RESULTS_DIR / f"test_{prefix}_seed{seed}.json")


def _first_present(d, candidates):
    """Return the first candidate key present in dict d, else None."""
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return None


def get_val_dice(data):
    return _first_present(data, ["val_dice_mean_last10", "best_val_dice"])


def get_val_hd95(data):
    return _first_present(data, ["val_hd95_mean_last10", "best_val_hd95"])


def get_val_iou(data):
    return _first_present(data, ["final_val_iou", "val_iou_mean_last10"])


def get_val_assd(data):
    """ASSD was never pre-aggregated — average the last 10 epochs of history."""
    history = data.get("history", [])
    vals = [e.get("val_assd") for e in history if e.get("val_assd") is not None]
    if len(vals) < 10:
        return None
    return float(np.mean(vals[-10:]))


def get_size_stratified_val(data):
    """
    Best-effort lookup for validation size-stratified Dice. Field name isn't
    fully pinned down from the docs alone — if this returns None for your
    files, open one result JSON and check the actual key, then add it to
    the `candidates` list below.
    """
    candidates = ["size_stratified", "size_stratified_dice", "stratified_dice"]
    for k in candidates:
        if k in data:
            return data[k]
    return None


def get_boundary_weight_history(data):
    """
    Per-epoch realized boundary weight for the adaptive schedule. Tries a
    few plausible key names; if none match, prints the actual keys found
    in the first history entry so you can fix WEIGHT_KEYS below.
    """
    WEIGHT_KEYS = ["boundary_weight", "lambda_b", "weight", "gate_weight"]
    history = data.get("history", [])
    if not history:
        return None
    for k in WEIGHT_KEYS:
        if k in history[0]:
            return [e.get(k) for e in history]
    print(f"  [warn] No known boundary-weight key found. Keys present in "
          f"history[0]: {list(history[0].keys())}")
    return None


# --------------------------------------------------------------------------
# Figure 1 — Validation null result (statistical_analysis.md, "Aggregate
# Results" / "Pairwise Statistical Tests" sections)
# --------------------------------------------------------------------------
def fig_validation_null_result():
    metrics = [("Dice", get_val_dice, "higher is better"),
               ("HD95 (mm)", get_val_hd95, "lower is better"),
               ("IoU", get_val_iou, "higher is better")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    any_data = False

    for ax, (label, getter, direction) in zip(axes, metrics):
        means, sds = [], []
        for cfg in CONFIGS:
            vals = []
            for seed in SEEDS:
                d = load_original(cfg, seed)
                if d is None:
                    continue
                v = getter(d)
                if v is not None:
                    vals.append(v)
            if vals:
                means.append(np.mean(vals))
                sds.append(np.std(vals, ddof=0))
            else:
                means.append(np.nan)
                sds.append(0)

        if all(np.isnan(m) for m in means):
            ax.text(0.5, 0.5, f"No data found for {label}", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue
        any_data = True

        x = np.arange(len(CONFIGS))
        colors = [COLORS[c] for c in CONFIGS]
        ax.bar(x, means, yerr=sds, color=colors, capsize=5,
               edgecolor="#222222", linewidth=0.7, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS])
        ax.set_ylabel(label)
        ax.set_title(f"{label}\n({direction})", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", alpha=0.15)

    if not any_data:
        print("  [skip] fig_validation_null_result: no result JSONs found — "
              f"check RESULTS_DIR = {RESULTS_DIR}")
        plt.close(fig)
        return

    fig.suptitle("Validation Set: No Configuration Differs Significantly (N=5 seeds)",
                  fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    _save(fig, "validation_null_result.png")


# --------------------------------------------------------------------------
# Figure 2 — Size-stratified Dice, validation and test
# (statistical_analysis.md, "Size-Stratified Analysis" sections)
# --------------------------------------------------------------------------
def fig_size_stratified(dataset="test"):
    """dataset: 'val' or 'test'."""
    sizes = ["small", "medium", "large"]
    data_by_cfg = {}

    for cfg in CONFIGS:
        vals = defaultdict(list)
        for seed in SEEDS:
            d = load_test(cfg, seed) if dataset == "test" else load_original(cfg, seed)
            if d is None:
                continue
            ss = d.get("size_stratified") if dataset == "test" else get_size_stratified_val(d)
            if not ss:
                continue
            for sz in sizes:
                entry = ss.get(sz)
                if entry is None:
                    continue
                v = entry.get("mean") if isinstance(entry, dict) else entry
                if v is not None:
                    vals[sz].append(v)
        data_by_cfg[cfg] = vals

    if all(len(v) == 0 for cfg_vals in data_by_cfg.values() for v in cfg_vals.values()):
        print(f"  [skip] fig_size_stratified({dataset}): no size-stratified data found.")
        return

    x = np.arange(len(sizes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for i, cfg in enumerate(CONFIGS):
        means = [np.mean(data_by_cfg[cfg][sz]) if data_by_cfg[cfg][sz] else np.nan for sz in sizes]
        sds = [np.std(data_by_cfg[cfg][sz], ddof=0) if data_by_cfg[cfg][sz] else 0 for sz in sizes]
        ax.bar(x + (i - 1) * width, means, width, yerr=sds, label=CONFIG_LABELS[cfg].replace("\n", " "),
               color=COLORS[cfg], capsize=4, edgecolor="#222222", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(["Small", "Medium", "Large"])
    ax.set_ylabel("Dice")
    ax.set_title(f"Size-Stratified Dice ({'Held-Out Test' if dataset == 'test' else 'Validation'} Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.15)

    plt.tight_layout()
    _save(fig, f"size_stratified_{dataset}.png")


# --------------------------------------------------------------------------
# Figure 3 — SDF-correction retrain comparison (statistical_analysis.md,
# "SDF-Approximation Verification" section)
# --------------------------------------------------------------------------
def fig_sdf_correction(correction="correctedSDF"):
    """correction: 'correctedSDF' (truncation fix) or 'rotationCorrected'."""
    label = "Patch-Boundary Truncation" if correction == "correctedSDF" else "Rotation Approximation"
    metrics = [("Dice", get_val_dice), ("HD95 (mm)", get_val_hd95), ("IoU", get_val_iou)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    any_data = False

    for ax, (mlabel, getter) in zip(axes, metrics):
        x = np.arange(len(SDF_CONFIGS))
        width = 0.32
        orig_means, orig_sds, corr_means, corr_sds = [], [], [], []

        for cfg in SDF_CONFIGS:
            ov, cv = [], []
            for seed in SEEDS:
                od = load_original(cfg, seed)
                cd = load_corrected(cfg, seed, correction)
                if od is not None and getter(od) is not None:
                    ov.append(getter(od))
                if cd is not None and getter(cd) is not None:
                    cv.append(getter(cd))
            orig_means.append(np.mean(ov) if ov else np.nan)
            orig_sds.append(np.std(ov, ddof=0) if ov else 0)
            corr_means.append(np.mean(cv) if cv else np.nan)
            corr_sds.append(np.std(cv, ddof=0) if cv else 0)

        if all(np.isnan(m) for m in orig_means + corr_means):
            ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue
        any_data = True

        ax.bar(x - width/2, orig_means, width, yerr=orig_sds, label="Original",
               color=COLORS["original"], capsize=4, edgecolor="#222222", linewidth=0.6)
        ax.bar(x + width/2, corr_means, width, yerr=corr_sds, label="Corrected",
               color=COLORS["corrected"], capsize=4, edgecolor="#222222", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c].replace("\n", " ") for c in SDF_CONFIGS])
        ax.set_title(mlabel, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", alpha=0.15)

    if not any_data:
        print(f"  [skip] fig_sdf_correction({correction}): no matching JSONs found.")
        plt.close(fig)
        return

    fig.suptitle(f"SDF {label}: Original vs. Corrected Retrain (N=5 seeds, all p>0.11)",
                  fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    _save(fig, f"sdf_{correction}.png")


# --------------------------------------------------------------------------
# Figure 4 — Validation vs. test, all 4 headline metrics
# (statistical_analysis.md, "Held-Out Test-Set Evaluation" section)
# --------------------------------------------------------------------------
def fig_val_vs_test():
    metrics = [("Dice", get_val_dice, "dice"), ("HD95 (mm)", get_val_hd95, "hd95"),
               ("IoU", get_val_iou, "iou"), ("ASSD (mm)", get_val_assd, "assd")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    any_data = False

    for ax, (label, val_getter, test_key) in zip(axes.flat, metrics):
        x = np.arange(len(CONFIGS))
        width = 0.32
        val_means, val_sds, test_means, test_sds = [], [], [], []

        for cfg in CONFIGS:
            vv, tv = [], []
            for seed in SEEDS:
                vd = load_original(cfg, seed)
                td = load_test(cfg, seed)
                if vd is not None and val_getter(vd) is not None:
                    vv.append(val_getter(vd))
                if td is not None and td.get(test_key) is not None:
                    tv.append(td[test_key])
            val_means.append(np.mean(vv) if vv else np.nan)
            val_sds.append(np.std(vv, ddof=0) if vv else 0)
            test_means.append(np.mean(tv) if tv else np.nan)
            test_sds.append(np.std(tv, ddof=0) if tv else 0)

        if all(np.isnan(m) for m in val_means + test_means):
            ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue
        any_data = True

        ax.bar(x - width/2, val_means, width, yerr=val_sds, label="Validation",
               color=COLORS["validation"], capsize=4, edgecolor="#222222", linewidth=0.6)
        ax.bar(x + width/2, test_means, width, yerr=test_sds, label="Test",
               color=COLORS["test"], capsize=4, edgecolor="#222222", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c].replace("\n", " ") for c in CONFIGS])
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", alpha=0.15)

    if not any_data:
        print("  [skip] fig_val_vs_test: no result/test JSONs found.")
        plt.close(fig)
        return

    fig.suptitle("Validation vs. Held-Out Test: All Four Headline Metrics",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "validation_vs_test_full.png")


# --------------------------------------------------------------------------
# Figure 5 — Adaptive schedule realized boundary weight over training
# (methodology.md, Section 3.3 "Realized weight, in practice")
# --------------------------------------------------------------------------
def fig_adaptive_weight_trajectory():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False

    for seed in SEEDS:
        d = load_original("boundary_adaptive", seed)
        if d is None:
            continue
        weights = get_boundary_weight_history(d)
        if weights is None:
            continue
        any_data = True
        ax.plot(range(1, len(weights) + 1), weights, label=f"seed {seed}", linewidth=1.6, alpha=0.85)

    if not any_data:
        print("  [skip] fig_adaptive_weight_trajectory: no boundary-weight history found "
              "— check get_boundary_weight_history()'s WEIGHT_KEYS against your actual JSON schema.")
        plt.close(fig)
        return

    ax.axhline(0.05, color="#c9691f", linestyle="--", linewidth=1.3, label="λ_max = 0.05 (ceiling)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Realized boundary weight")
    ax.set_title("Adaptive Schedule Never Approaches Its Own Weight Ceiling",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.15)

    plt.tight_layout()
    _save(fig, "adaptive_weight_trajectory.png")


# --------------------------------------------------------------------------
# Figure 6 — Gradient magnitude comparison (gradient_analysis.md, Section 3)
# --------------------------------------------------------------------------
def fig_gradient_magnitude(grad_json_path=None):
    """
    Unlike the other figures, this one number-set comes from a single
    diagnostic run (grad_telemetry_check.py), not a multi-seed JSON
    directory. If grad_telemetry_check.py writes its own JSON output,
    point --grad-json at it; otherwise this falls back to the values
    already reported in docs/gradient_analysis.md.
    """
    if grad_json_path and Path(grad_json_path).exists():
        d = _load(Path(grad_json_path))
        regional_pct = d["regional_share_pct"]
        boundary_raw_pct = d["boundary_raw_share_pct"]
        boundary_eff_pct = d["boundary_effective_share_pct"]
    else:
        # Values as reported in docs/gradient_analysis.md Section 3 —
        # update here if grad_telemetry_check.py is re-run.
        regional_pct, boundary_raw_pct, boundary_eff_pct = 99.06, 0.94, 0.047

    labels = ["Regional\n(Dice + Focal)", "Boundary\n(raw, \u03bb=1.0)", "Boundary\n(effective, \u03bb=0.05)"]
    values = [regional_pct, boundary_raw_pct, boundary_eff_pct]
    colors = ["#2c5f8a", "#c9691f", "#c9691f"]
    alphas = [1.0, 1.0, 0.45]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.8, width=0.55)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    ax.set_yscale("log")
    ax.set_ylim(0.01, 200)
    ax.set_ylabel("Share of combined gradient magnitude (%, log scale)")
    ax.set_title("Boundary Loss Is Gradient-Starved at Its Trained Weight",
                 fontsize=13, fontweight="bold", pad=14)

    for bar, v in zip(bars, values):
        label = f"{v:.2f}%" if v >= 1 else f"{v:.3f}%"
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.15)

    plt.tight_layout()
    _save(fig, "gradient_magnitude_comparison.png")


# --------------------------------------------------------------------------
def _save(fig, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {out}")


FIGURES = {
    "val_null":       fig_validation_null_result,
    "size_val":        lambda: fig_size_stratified("val"),
    "size_test":       lambda: fig_size_stratified("test"),
    "sdf_truncation":  lambda: fig_sdf_correction("correctedSDF"),
    "sdf_rotation":    lambda: fig_sdf_correction("rotationCorrected"),
    "val_vs_test":     fig_val_vs_test,
    "adaptive_weight": fig_adaptive_weight_trajectory,
    "gradient":        fig_gradient_magnitude,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=list(FIGURES.keys()), default=None,
                         help="Generate just one figure instead of all.")
    parser.add_argument("--list", action="store_true", help="List available figure names and exit.")
    parser.add_argument("--grad-json", default=None,
                         help="Optional path to grad_telemetry_check.py's JSON output.")
    args = parser.parse_args()

    if args.list:
        for name in FIGURES:
            print(name)
        return

    print(f"RESULTS_DIR      = {RESULTS_DIR}")
    print(f"TEST_RESULTS_DIR = {TEST_RESULTS_DIR}")
    print(f"FIGURES_DIR      = {FIGURES_DIR}\n")

    targets = [args.only] if args.only else list(FIGURES.keys())
    for name in targets:
        print(f"Generating: {name}")
        if name == "gradient":
            fig_gradient_magnitude(grad_json_path=args.grad_json)
        else:
            FIGURES[name]()


if __name__ == "__main__":
    main()