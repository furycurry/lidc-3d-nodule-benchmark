import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import config

SOURCE_H5 = config.H5_PATH
SDF_H5 = config.SDF_H5_PATH


def compute_sdf_single(mask_np):
    posmask = mask_np.astype(bool)
    if posmask.any() and not posmask.all():
        negmask = ~posmask
        pos_dist = distance_transform_edt(posmask)
        neg_dist = distance_transform_edt(negmask)
        return (neg_dist - pos_dist).astype(np.float32)
    return np.zeros_like(mask_np, dtype=np.float32)


with h5py.File(SOURCE_H5, "r") as src, h5py.File(SDF_H5, "w") as dst:
    sdf_grp = dst.create_group("sdf")
    keys = sorted(src["masks"].keys())
    for key in tqdm(keys, desc="Precomputing SDFs"):
        mask = src["masks"][key][()]
        sdf = compute_sdf_single(mask)
        sdf_grp.create_dataset(key, data=sdf, compression="gzip", compression_opts=4, dtype="f4")

print(f"Done. SDF cache written to {SDF_H5}")