import json
import numpy as np
from pathlib import Path
from scipy import stats
import config

RESULTS_DIR = config.RESULTS_DIR

configs = {"baseline": {}, "boundary_fixed": {}, "boundary_adaptive": {}}

for f in RESULTS_DIR.glob("*.json"):
    with open(f) as fp:
        data = json.load(fp)
    if not data.get("use_boundary_loss", False):
        key = "baseline"
    elif data.get("adaptive_boundary", False):
        key = "boundary_adaptive"
    else:
        key = "boundary_fixed"
    configs[key][data["seed"]] = data

common_seeds = sorted(set(configs["baseline"]) & set(configs["boundary_fixed"]) & set(configs["boundary_adaptive"]))
print(f"Paired seeds available: {common_seeds}\n")

def paired_test(name_a, name_b, metric):
    a = [configs[name_a][s][metric] for s in common_seeds]
    b = [configs[name_b][s][metric] for s in common_seeds]
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    print(f"{name_a} vs {name_b} | {metric}")
    print(f"  {name_a}: {np.mean(a):.4f} ± {np.std(a):.4f}")
    print(f"  {name_b}: {np.mean(b):.4f} ± {np.std(b):.4f}")
    print(f"  paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"  wilcoxon: p={w_p:.4f}\n")

for metric in ["best_val_dice", "best_val_hd95"]:
    paired_test("baseline", "boundary_fixed", metric)
    paired_test("baseline", "boundary_adaptive", metric)
    paired_test("boundary_fixed", "boundary_adaptive", metric)