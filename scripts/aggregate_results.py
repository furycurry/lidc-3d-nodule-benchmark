import json
import numpy as np
from pathlib import Path
import config

RESULTS_DIR = config.RESULTS_DIR

configs = {"baseline": [], "boundary_fixed": [], "boundary_adaptive": []}

for f in RESULTS_DIR.glob("*.json"):
    with open(f) as fp:
        data = json.load(fp)
    if not data.get("use_boundary_loss", False):
        key = "baseline"
    elif data.get("adaptive_boundary", False):
        key = "boundary_adaptive"
    else:
        key = "boundary_fixed"
    configs[key].append(data)

print(f"{'Config':<18} {'N':<4} {'Dice (mean±std)':<20} {'HD95 (mean±std)':<20}")
print("-" * 66)
for key, runs in configs.items():
    if not runs:
        continue
    dices = [r["best_val_dice"] for r in runs]
    hd95s = [r["best_val_hd95"] for r in runs]
    seeds = [r["seed"] for r in runs]
    print(f"{key:<18} {len(runs):<4} {np.mean(dices):.4f}±{np.std(dices):.4f}        "
          f"{np.mean(hd95s):.4f}±{np.std(hd95s):.4f}")
    print(f"  Per-seed Dice: {sorted(zip(seeds, [round(d,4) for d in dices]))}")
    print(f"  Per-seed HD95: {sorted(zip(seeds, [round(h,4) for h in hd95s]))}")