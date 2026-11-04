#!/usr/bin/env python3
"""Step 1 of reconciling repo numbers: print the structure of the raw result JSONs.

Run from the repo root:
    python tools/inspect_schema.py
Paste the full output back so make_tables.py can be written against the real keys.
"""
import glob, json, os, re, sys
from collections import Counter

RESULT_DIRS = ["outputs/results", "outputs/test_results"]

def shape(v, depth=0, max_depth=3, key=""):
    pad = "  " * depth
    if isinstance(v, dict):
        print(f"{pad}{key}: dict[{len(v)}]")
        if depth < max_depth:
            for k in list(v)[:40]:
                shape(v[k], depth + 1, max_depth, k)
            if len(v) > 40:
                print(f"{pad}  ... {len(v) - 40} more keys")
    elif isinstance(v, list):
        print(f"{pad}{key}: list[{len(v)}]")
        if v and depth < max_depth:
            shape(v[0], depth + 1, max_depth, "[0]")
    else:
        s = repr(v)
        print(f"{pad}{key}: {type(v).__name__} = {s[:60]}")

def main():
    for d in RESULT_DIRS:
        if not os.path.isdir(d):
            print(f"!! {d} not found (run from the repo root)"); continue
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        print("=" * 72)
        print(f"{d}: {len(files)} JSON files")
        # filename patterns: replace seed numbers so groups collapse
        pat = Counter(re.sub(r"\d+", "N", os.path.basename(f)) for f in files)
        for k, c in pat.most_common():
            print(f"  {c:3d} x {k}")
        print("  first 5 names:", [os.path.basename(f) for f in files[:5]])
        if not files:
            continue
        # show the structure of the first non-summary file
        pick = next((f for f in files if "summary" not in f and "analysis" not in f), files[0])
        print("-" * 72)
        print("structure of:", pick)
        with open(pick) as fh:
            shape(json.load(fh))
        # show the structure of summary-like files too (top level only)
        for f in files:
            b = os.path.basename(f)
            if "summary" in b or "analysis" in b:
                print("-" * 72)
                print("structure of:", f)
                with open(f) as fh:
                    shape(json.load(fh), max_depth=2)
    print("=" * 72)
    print("Also paste: ls outputs/ and the first 20 lines of one per-epoch 'history' entry if present.")

if __name__ == "__main__":
    sys.exit(main())
