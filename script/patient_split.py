import json
import h5py
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from scipy import stats
import config

with open(config.BASE_DIR / "patch_patient_mapping.json") as f:
    mapping = json.load(f)

with h5py.File(config.H5_PATH, "r") as f:
    keys = sorted(f["images"].keys())
    fg_voxel_counts = {k: int(f["masks"][k][()].sum()) for k in keys}

patient_ids = np.array([mapping[k] for k in keys])
keys_arr = np.array(keys)

# --- Split 1: separate out test set (15%) from the rest ---
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
trainval_idx, test_idx = next(gss_test.split(X=np.arange(len(keys)), groups=patient_ids))

trainval_keys = keys_arr[trainval_idx]
trainval_patients = patient_ids[trainval_idx]
test_keys = keys_arr[test_idx]
test_patients = patient_ids[test_idx]

# --- Split 2: split remaining 85% into train (~70% of total) / val (~15% of total) ---
# test_size here is relative to trainval, so 0.1765 of 85% ≈ 15% of total
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
train_idx_rel, val_idx_rel = next(gss_val.split(X=np.arange(len(trainval_keys)), groups=trainval_patients))

train_keys = trainval_keys[train_idx_rel].tolist()
val_keys = trainval_keys[val_idx_rel].tolist()
test_keys = test_keys.tolist()

train_patients = set(trainval_patients[train_idx_rel])
val_patients = set(trainval_patients[val_idx_rel])
test_patients_set = set(test_patients)

# --- Leakage checks across all three partitions (pairwise) ---
assert len(train_patients & val_patients) == 0, "LEAKAGE: train/val patient overlap"
assert len(train_patients & test_patients_set) == 0, "LEAKAGE: train/test patient overlap"
assert len(val_patients & test_patients_set) == 0, "LEAKAGE: val/test patient overlap"

print(f"Train: {len(train_keys)} patches, {len(train_patients)} patients")
print(f"Val:   {len(val_keys)} patches, {len(val_patients)} patients")
print(f"Test:  {len(test_keys)} patches, {len(test_patients_set)} patients")
print(f"Total: {len(train_keys) + len(val_keys) + len(test_keys)} / {len(keys)} patches")
print("✓ No patient overlap across any of the three partitions")

# --- Stratification sanity check: nodule size (foreground voxel count) distribution ---
train_fg = [fg_voxel_counts[k] for k in train_keys]
val_fg = [fg_voxel_counts[k] for k in val_keys]
test_fg = [fg_voxel_counts[k] for k in test_keys]

print(f"\nNodule size (foreground voxels) — median [IQR]:")
print(f"  Train: {np.median(train_fg):.0f} [{np.percentile(train_fg,25):.0f}-{np.percentile(train_fg,75):.0f}]")
print(f"  Val:   {np.median(val_fg):.0f} [{np.percentile(val_fg,25):.0f}-{np.percentile(val_fg,75):.0f}]")
print(f"  Test:  {np.median(test_fg):.0f} [{np.percentile(test_fg,25):.0f}-{np.percentile(test_fg,75):.0f}]")

# KS-test: are these distributions plausibly from the same underlying population?
ks_train_val = stats.ks_2samp(train_fg, val_fg)
ks_train_test = stats.ks_2samp(train_fg, test_fg)
ks_val_test = stats.ks_2samp(val_fg, test_fg)

print(f"\nKS-test p-values (low p-value = distributions differ significantly):")
print(f"  Train vs Val:  p={ks_train_val.pvalue:.4f}")
print(f"  Train vs Test: p={ks_train_test.pvalue:.4f}")
print(f"  Val vs Test:   p={ks_val_test.pvalue:.4f}")

if min(ks_train_val.pvalue, ks_train_test.pvalue, ks_val_test.pvalue) < 0.05:
    print("  ⚠ WARNING: at least one pair of splits has significantly different nodule size "
          "distributions (p<0.05). Consider re-seeding or using stratified grouping.")
else:
    print("  ✓ No significant size-distribution difference detected across splits")

# --- Persist the split ---
split_data = {
    "train_keys": train_keys,
    "val_keys": val_keys,
    "test_keys": test_keys,
    "seed": 42,
    "notes": "Patient-disjoint 3-way split. Test set must only be used once, for final reporting."
}
with open(config.BASE_DIR / "train_val_test_split.json", "w") as f:
    json.dump(split_data, f, indent=2)
print(f"\nSplit saved to {config.BASE_DIR / 'train_val_test_split.json'}")