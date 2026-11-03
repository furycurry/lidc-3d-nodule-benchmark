import json, glob, os
import numpy as np

print("="*60)
print("1. SEARCHING FOR ALL TEST & AUDIT JSON ARTIFACTS")
print("="*60)
all_jsons = sorted(glob.glob("/workspace/**/*.json", recursive=True))
for p in all_jsons:
    if any(k in os.path.basename(p) for k in ["strat", "nsd", "test", "trunc", "wide", "lambda", "epoch"]):
        print(p)

print("\n" + "="*60)
print("2. CLINICAL & VOLUME STRATIFICATION COUNTS")
print("="*60)
strat_path = "/workspace/outputs/test_results/clinical_stratification_summary.json"
if os.path.exists(strat_path):
    d = json.load(open(strat_path))
    print("Top-level keys in stratification JSON:", list(d.keys()))
    # Print sub-cohort counts if available
    for k in ["terciles", "volume", "subgroups", "counts"]:
        if k in d:
            print(f"Key '{k}':", json.dumps(d[k], indent=2))
else:
    print(f"File not found: {strat_path}")

print("\n" + "="*60)
print("3. HELD-OUT TEST DICE PER SEED (UNETR & UNET3D)")
print("="*60)
test_files = sorted(glob.glob("/workspace/outputs/**/test*.json", recursive=True))
for tf in test_files:
    try:
        data = json.load(open(tf))
        if isinstance(data, dict):
            # Print entries containing per-seed or summary test metrics
            print(f"\n--- File: {tf} ---")
            for k in ["test_dice", "dice", "per_seed", "metrics", "summary"]:
                if k in data:
                    print(f"  {k}:", data[k])
    except Exception as e:
        pass

print("\n" + "="*60)
print("4. TRUNCATION & ARTIFACT RE-TRAINING LOGS")
print("="*60)
audit_files = sorted(glob.glob("/workspace/outputs/**/*trunc*.json", recursive=True)) + \
              sorted(glob.glob("/workspace/outputs/**/*wide*.json", recursive=True)) + \
              sorted(glob.glob("/workspace/outputs/**/*audit*.json", recursive=True))
if audit_files:
    for af in audit_files:
        print(f"Found audit file: {af}")
        print(open(af).read()[:300])
else:
    print("No direct *trunc* or *wide* JSONs found. Checking log files in /workspace:")
    for log in sorted(glob.glob("/workspace/*.log")):
        print(" ", log)

