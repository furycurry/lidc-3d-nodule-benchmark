#!/usr/bin/env python3
"""Deeper schema dump for the test-side JSONs. Run from the repo root:
    python tools/inspect_test_schema.py > test_schema.txt
and paste test_schema.txt back. Reads only outputs/test_results/*.json.
"""
import json, os, sys
D = "outputs/test_results"

def shape(v, depth=0, max_depth=6, key=""):
    pad = "  " * depth
    if isinstance(v, dict):
        print(f"{pad}{key}: dict[{len(v)}]")
        if depth < max_depth:
            for k in list(v)[:30]:
                shape(v[k], depth + 1, max_depth, k)
            if len(v) > 30:
                print(f"{pad}  ... {len(v) - 30} more keys")
    elif isinstance(v, list):
        print(f"{pad}{key}: list[{len(v)}]")
        if v and depth < max_depth:
            shape(v[0], depth + 1, max_depth, "[0]")
    else:
        print(f"{pad}{key}: {type(v).__name__} = {repr(v)[:50]}")

files = [
    "test_baseline_unet3d_paper_seed42.json", "test_boundary_adaptive_unetr_seed2024.json",
    "adaptive_gate_analysis_unetr.json", "architecture_comparison_summary.json",
    "clinical_stratification_summary.json", "test_analysis_summary.json",
    "test_nsd_summary.json",
]
for f in files:
    p = os.path.join(D, f)
    print("=" * 72); print(p)
    if not os.path.exists(p):
        print("  (missing)"); continue
    shape(json.load(open(p)))
# per-run entry inside the merged summary, one per architecture
p = os.path.join(D, "test_summary_all.json")
if os.path.exists(p):
    r = json.load(open(p))["results"]
    for k in ("baseline_unet3d_paper_seed42", "baseline_unetr_seed42"):
        print("=" * 72); print("test_summary_all.json -> results ->", k)
        shape(r[k], max_depth=3)
