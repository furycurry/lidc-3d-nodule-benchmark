#!/usr/bin/env python3
"""Patient-disjoint train/val/test split with a nodule-size balance search.

Reproduces the published split: 30 candidate seeds (0-29) of a two-stage GroupShuffleSplit
(15% of patients to test, then 17.65% of the remainder to validation). Each candidate is scored by
the minimum pairwise two-sample Kolmogorov-Smirnov p-value of foreground voxel counts across the
three partitions, and the highest-scoring seed is kept (seed 5 for LIDC-IDRI, min p = 0.553).

Usage:
    python data_prep/patient_split.py --h5 lidc_patches_int16.h5 \
        --mapping patch_patient_mapping.json --out split_regenerated.json \
        --verify train_val_test_split.json
"""
import argparse
import json

import h5py
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="lidc_patches_int16.h5")
    ap.add_argument("--mapping", default="patch_patient_mapping.json")
    ap.add_argument("--out", default="train_val_test_split.json")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--verify", default=None, help="existing split JSON to compare against")
    args = ap.parse_args()

    with open(args.mapping) as f:
        mapping = json.load(f)
    with h5py.File(args.h5, "r") as f:
        keys = sorted(f["images"].keys())
        fg = {k: int(f["masks"][k][()].sum()) for k in keys}
    patient_ids = np.array([mapping[k] for k in keys])
    keys_arr = np.array(keys)

    def try_split(seed):
        gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        trainval_idx, test_idx = next(gss_test.split(X=np.arange(len(keys)), groups=patient_ids))
        trainval_keys, trainval_patients = keys_arr[trainval_idx], patient_ids[trainval_idx]
        test_keys = keys_arr[test_idx].tolist()
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
        tr_rel, va_rel = next(gss_val.split(X=np.arange(len(trainval_keys)), groups=trainval_patients))
        train_keys, val_keys = trainval_keys[tr_rel].tolist(), trainval_keys[va_rel].tolist()

        tr_fg, va_fg, te_fg = ([fg[k] for k in ks] for ks in (train_keys, val_keys, test_keys))
        p_tv = stats.ks_2samp(tr_fg, va_fg).pvalue
        p_tt = stats.ks_2samp(tr_fg, te_fg).pvalue
        p_vt = stats.ks_2samp(va_fg, te_fg).pvalue

        tr_p, va_p, te_p = set(trainval_patients[tr_rel]), set(trainval_patients[va_rel]), set(patient_ids[test_idx])
        assert not (tr_p & va_p) and not (tr_p & te_p) and not (va_p & te_p), "patient leakage"
        return dict(seed=seed, min_p=min(p_tv, p_tt, p_vt), p_tv=p_tv, p_tt=p_tt, p_vt=p_vt,
                    train_keys=train_keys, val_keys=val_keys, test_keys=test_keys)

    results = [try_split(s) for s in range(args.n_seeds)]
    best = max(results, key=lambda r: r["min_p"])
    print(f"best seed {best['seed']}  min pairwise KS p = {best['min_p']:.3f}  "
          f"(train/val/test patches: {len(best['train_keys'])}/{len(best['val_keys'])}/{len(best['test_keys'])})")

    split = {
        "train_keys": best["train_keys"], "val_keys": best["val_keys"], "test_keys": best["test_keys"],
        "seed": best["seed"],
        "ks_pvalues": {"train_vs_val": best["p_tv"], "train_vs_test": best["p_tt"], "val_vs_test": best["p_vt"]},
        "notes": "Patient-disjoint 3-way split, seed selected via KS-test balance search over "
                 f"{args.n_seeds} candidates. Test set must only be used once, for final reporting.",
    }
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"wrote {args.out}")

    if args.verify:
        with open(args.verify) as f:
            ref = json.load(f)
        for part in ("train_keys", "val_keys", "test_keys"):
            same_set = set(ref[part]) == set(split[part])
            same_order = ref[part] == split[part]
            print(f"  {part}: same members={same_set}, same order={same_order}")
        print(f"  seed: reference {ref.get('seed')} vs regenerated {split['seed']}")


if __name__ == "__main__":
    main()
