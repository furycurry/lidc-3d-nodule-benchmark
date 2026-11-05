#!/usr/bin/env python3
"""
Generate the corrected (non-truncated) SDF cache for the boundary-loss
retrain experiment.

For each of the 2,651 nodules, computes the "truth" SDF -- EDT on the full
128^3 wide mask (from lidc_wide_masks_and_provenance.h5), center-cropped to
64^3, clipped to +-20mm (matching the existing training pipeline's clip) --
and saves it as a new float32 HDF5 dataset, keyed identically to the
original dataset (nodule_XXXX), so it can be substituted directly wherever
the training pipeline currently sources its SDF.

Does NOT touch the rotation-under-augmentation approximation (RandRotated's
bilinear rotation of the cached SDF) -- that stays exactly as it is in the
current lidc_dataset.py. Leaving it unchanged is deliberate: it isolates
the truncation fix as the single changed variable, so if the result changes,
we know it was truncation and not some conflated second fix.

Usage:
    python generate_corrected_sdf.py --wide-h5 lidc_wide_masks_and_provenance.h5
"""
import argparse

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_distance(mask):
    mask = mask.astype(bool)
    if not mask.any():
        return np.full(mask.shape, 1e6, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -1e6, dtype=np.float32)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return (outside - inside).astype(np.float32)


def center_crop(volume, size):
    c = np.array(volume.shape) // 2
    h = size // 2
    return volume[c[0]-h:c[0]-h+size, c[1]-h:c[1]-h+size, c[2]-h:c[2]-h+size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide-h5", required=True)
    ap.add_argument("--target-size", type=int, default=64)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--out-h5", default="lidc_corrected_sdf_64.h5")
    args = ap.parse_args()

    with h5py.File(args.wide_h5, "r") as wf, h5py.File(args.out_h5, "w") as of:
        sdf_grp = of.create_group("sdf")
        keys = sorted(wf["wide_masks"].keys())
        print(f"Generating corrected SDF for {len(keys)} nodules...")

        for i, key in enumerate(keys):
            wide_mask = wf["wide_masks"][key][()]
            truth_sdf_full = signed_distance(wide_mask)
            truth_sdf_64 = center_crop(truth_sdf_full, args.target_size)
            truth_sdf_64 = np.clip(truth_sdf_64, -args.clip, args.clip).astype(np.float32)

            sdf_grp.create_dataset(
                key, data=truth_sdf_64, compression="gzip", compression_opts=4
            )

            if (i + 1) % 500 == 0:
                print(f"  ...{i+1}/{len(keys)}")

    print(f"\nWrote corrected SDF cache for {len(keys)} nodules to {args.out_h5}")
    print("Keys match the original dataset's nodule_XXXX naming exactly.")


if __name__ == "__main__":
    main()
