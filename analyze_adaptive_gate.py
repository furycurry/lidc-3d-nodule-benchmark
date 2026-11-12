#!/usr/bin/env python3
"""
analyze_adaptive_gate.py

Adaptive boundary-loss gate (PCG-BW) behavior analysis, for a given
architecture. Reads per-epoch `gate_active`/`boundary_weight` history
already saved in the boundary_adaptive_{architecture}_seed*.json result
files -- no retraining, no new runs. Mirrors the analysis originally done
manually for UNet3DPaper (methodology.md Section 3.3 / gradient_analysis.md
Section 4.2), applied here to a different architecture.

Usage:
    python analyze_adaptive_gate.py --architecture unetr

Dependencies: numpy, config.py (paths)
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import config
    RESULTS_DIR = config.RESULTS_DIR
    OUTPUT_DIR = config.OUTPUT_DIR
except Exception:
    BASE_DIR = Path(__file__).resolve().parent.parent
    RESULTS_DIR = BASE_DIR / "outputs" / "results"
    OUTPUT_DIR = BASE_DIR / "outputs"

if Path("outputs/results").exists():
    RESULTS_DIR = Path("outputs/results")
if Path("outputs").exists():
    OUTPUT_DIR = Path("outputs")

SEEDS = [42, 123, 456, 789, 2024]

# GATE_TAU_ENTER: velocity must drop to/below this for a qualifying epoch.
# Read from config.py if available so this always matches the actual run;
# falls back to the project's documented default otherwise.
try:
    GATE_TAU_ENTER = config.GATE_TAU_ENTER
except Exception:
    GATE_TAU_ENTER = 0.005


def activation_periods(gate_active_seq):
    """
    Count distinct True-runs in a gate_active boolean sequence (None/False
    treated as inactive). Returns (n_periods, ended_active) where
    ended_active tells you whether the run finished in an active state.
    """
    n_periods = 0
    was_active = False
    for g in gate_active_seq:
        active = bool(g)
        if active and not was_active:
            n_periods += 1
        was_active = active
    return n_periods, was_active


def longest_streak(qualifies_seq):
    """Longest run of consecutive True values in a boolean sequence."""
    longest = current = 0
    for q in qualifies_seq:
        current = current + 1 if q else 0
        longest = max(longest, current)
    return longest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, required=True,
                         help="e.g. unetr or unet3d_paper")
    args = parser.parse_args()
    arch = args.architecture

    print("=" * 72)
    print(f" ADAPTIVE GATE (PCG-BW) BEHAVIOR ANALYSIS  —  {arch}")
    print("=" * 72)

    per_seed_summary = []
    all_weights_flat = []

    for seed in SEEDS:
        path = RESULTS_DIR / f"boundary_adaptive_{arch}_seed{seed}.json"
        if not path.exists():
            print(f"[SKIP] Missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)

        history = data.get("history", [])
        gate_seq = [h.get("gate_active") for h in history]
        weight_seq = [h.get("boundary_weight") for h in history if h.get("boundary_weight") is not None]

        n_periods, ended_active = activation_periods(gate_seq)
        active_weights = [h.get("boundary_weight", 0.0) or 0.0
                           for h in history if h.get("gate_active")]
        full_run_weights = [h.get("boundary_weight", 0.0) or 0.0 for h in history]

        max_w = max(weight_seq) if weight_seq else 0.0
        mean_active_w = float(np.mean(active_weights)) if active_weights else 0.0
        mean_full_w = float(np.mean(full_run_weights)) if full_run_weights else 0.0

        all_weights_flat.extend(weight_seq)

        # --- Velocity diagnostics: WHY did/didn't the gate activate? ---
        # velocity is None for the first K_WINDOW+1 epochs (insufficient
        # history) -- only look at epochs where it was actually computed.
        velocities = [h.get("velocity") for h in history if h.get("velocity") is not None]
        qualifies = [v <= GATE_TAU_ENTER for v in velocities]
        n_qualifying = sum(qualifies)
        streak = longest_streak(qualifies)
        min_velocity = min(velocities) if velocities else None

        pattern = "clean single activation" if n_periods == 1 and ended_active else \
                  "never activated" if n_periods == 0 else \
                  f"{n_periods} activation period(s), {'ends active' if ended_active else 'ends inactive'}"

        print(f"\nSeed {seed}:")
        print(f"  Activation pattern : {pattern}")
        print(f"  Max weight observed: {max_w:.4f}")
        print(f"  Mean weight (active epochs only): {mean_active_w:.4f}")
        print(f"  Mean weight (full {len(history)}-epoch run): {mean_full_w:.4f}")
        if velocities:
            min_v_str = f"{min_velocity:.5f}" if min_velocity is not None else "N/A"
            print(f"  Min velocity observed: {min_v_str}  (threshold: {GATE_TAU_ENTER})")
            print(f"  Qualifying epochs (velocity <= threshold): {n_qualifying}/{len(velocities)}")
            print(f"  Longest consecutive qualifying streak: {streak}  (needs 3 to trigger)")
        else:
            print("  Velocity data: not present in history")

        per_seed_summary.append({
            "seed": seed, "n_activation_periods": n_periods, "ends_active": ended_active,
            "pattern": pattern, "max_weight": max_w,
            "mean_weight_active_epochs": mean_active_w, "mean_weight_full_run": mean_full_w,
            "min_velocity": min_velocity, "n_qualifying_epochs": n_qualifying,
            "n_epochs_with_velocity": len(velocities), "longest_qualifying_streak": streak,
        })

    if all_weights_flat:
        overall_max = max(all_weights_flat)
        print("\n" + "=" * 72)
        print(f" Max weight observed across ALL seeds/epochs: {overall_max:.4f}  (λ_max = 0.05)")
        print("=" * 72)
    else:
        overall_max = None

    out_dir = OUTPUT_DIR / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adaptive_gate_analysis_{arch}.json"
    with open(out_path, "w") as f:
        json.dump({"architecture": arch, "overall_max_weight": overall_max,
                    "per_seed": per_seed_summary}, f, indent=2)
    print(f"\n[OK] Saved → {out_path}")


if __name__ == "__main__":
    main()