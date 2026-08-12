import argparse
import traceback
import warnings
import h5py
import numpy as np
import pylidc as pl
from pylidc.utils import consensus
from scipy.ndimage import zoom
from tqdm import tqdm

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Build LIDC-IDRI HDF5 patch dataset")
parser.add_argument(
    "--output",
    type=str,
    default="./lidc_patches_int16.h5",
    help="Path to output HDF5 file (default: ./lidc_patches_int16.h5)",
)
args = parser.parse_args()

OUTPUT_H5 = args.output

TARGET_PATCH_SIZE = np.array([64, 64, 64])  # (Z, Y, X) at 1.0mm^3

scans = pl.query(pl.Scan).all()
print(f"[BUILD] Extracting isotropic 3D nodule patches across {len(scans)} scans...")

nodule_counter = 0
failed_scans = 0

with h5py.File(OUTPUT_H5, "w") as h5f:
    img_grp = h5f.create_group("images")
    mask_grp = h5f.create_group("masks")

    for scan in tqdm(scans, desc="Processing LIDC Scans"):
        clusters = scan.cluster_annotations()
        if not clusters:
            continue

        try:
            # 1. Load volume in canonical (Z, Y, X) orientation
            # scan.to_volume() returns (Y, X, Z) -> transpose(2, 0, 1) converts to (Z, Y, X)
            vol_orig = np.transpose(scan.to_volume(), (2, 0, 1)).astype(np.int16)
            mask_orig = np.zeros(vol_orig.shape, dtype=np.uint8)

            # 2. Build consensus mask & calculate cluster centroids
            cluster_centroids = []
            for cluster in clusters:
                cmask, bbox, _ = consensus(cluster, clevel=0.5)
                # bbox is (y_slice, x_slice, z_slice)
                y_slice, x_slice, z_slice = bbox[0], bbox[1], bbox[2]

                # Align cmask (Y, X, Z) -> (Z, Y, X)
                mask_orig[z_slice, y_slice, x_slice] = np.logical_or(
                    mask_orig[z_slice, y_slice, x_slice], cmask.transpose(2, 0, 1)
                )

                # Centroid in original voxel frame (Z, Y, X)
                z_ctr = (z_slice.start + z_slice.stop) / 2.0
                y_ctr = (y_slice.start + y_slice.stop) / 2.0
                x_ctr = (x_slice.start + x_slice.stop) / 2.0
                cluster_centroids.append(np.array([z_ctr, y_ctr, x_ctr]))

            orig_spacing = np.array(
                [scan.slice_thickness, scan.pixel_spacing, scan.pixel_spacing],
                dtype=np.float64,
            )

            # Native voxel dimensions corresponding to 64mm physical window (+4mm margin for interpolation)
            native_crop_radius = np.ceil((TARGET_PATCH_SIZE / 2.0 + 2.0) / orig_spacing).astype(int)

            for ctr in cluster_centroids:
                ctr_int = np.round(ctr).astype(int)

                # Native crop boundaries
                src_start = np.maximum(0, ctr_int - native_crop_radius)
                src_end = np.minimum(np.array(vol_orig.shape), ctr_int + native_crop_radius)

                v_sub = vol_orig[src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]]
                m_sub = mask_orig[src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]]

                # Resample sub-volume to 1.0mm^3 isotropic
                v_resampled = zoom(v_sub, orig_spacing, order=1, prefilter=False).astype(np.int16)
                m_resampled = zoom(m_sub, orig_spacing, order=0, prefilter=False).astype(np.uint8)

                # Center crop to exact 64x64x64 target shape
                r_ctr = np.array(v_resampled.shape) // 2
                half_p = TARGET_PATCH_SIZE // 2

                c_start = np.maximum(0, r_ctr - half_p)
                c_end = np.minimum(np.array(v_resampled.shape), r_ctr + half_p)

                v_crop = v_resampled[c_start[0]:c_end[0], c_start[1]:c_end[1], c_start[2]:c_end[2]]
                m_crop = m_resampled[c_start[0]:c_end[0], c_start[1]:c_end[1], m_start[2]:c_end[2]]

                # Pad with -1000 HU (air) if touching scan boundaries
                if v_crop.shape != tuple(TARGET_PATCH_SIZE):
                    v_padded = np.full(TARGET_PATCH_SIZE, -1000, dtype=np.int16)
                    m_padded = np.zeros(TARGET_PATCH_SIZE, dtype=np.uint8)

                    p_start = half_p - (r_ctr - c_start)
                    p_end = p_start + (c_end - c_start)

                    v_padded[p_start[0]:p_end[0], p_start[1]:p_end[1], p_start[2]:p_end[2]] = v_crop
                    m_padded[p_start[0]:p_end[0], p_start[1]:p_end[1], p_start[2]:p_end[2]] = m_crop

                    v_crop, m_crop = v_padded, m_padded

                key = f"nodule_{nodule_counter:04d}"
                img_grp.create_dataset(key, data=v_crop, compression="gzip", compression_opts=4, dtype="i2")
                mask_grp.create_dataset(key, data=m_crop, compression="gzip", compression_opts=4, dtype="u1")
                nodule_counter += 1

        except Exception as e:
            failed_scans += 1
            print(f"\n[ERROR] Scan {scan.patient_id} failed: {e}")
            traceback.print_exc()
            continue

print(f"\n[COMPLETE] Extracted {nodule_counter} patches across dataset. Failed scans: {failed_scans}")