import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt
import monai.transforms as MT
import config


def _signed_distance(mask):
    """EDT-based signed distance field: negative inside, positive outside."""
    mask = mask.astype(bool)
    if not mask.any():
        return np.full(mask.shape, 1e6, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -1e6, dtype=np.float32)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return (outside - inside).astype(np.float32)


def _center_crop(volume, size):
    c = np.array(volume.shape) // 2
    h = size // 2
    return volume[c[0]-h:c[0]-h+size, c[1]-h:c[1]-h+size, c[2]-h:c[2]-h+size]


class LIDCDataset(Dataset):
    def __init__(self, h5_path=config.H5_PATH, sdf_path=config.SDF_H5_PATH,
                 indices=None, keys=None, is_train=True, use_sdf=False,
                 wide_mask_path=None):
        self.h5_path = str(h5_path)
        self.sdf_path = str(sdf_path)
        self.wide_mask_path = str(wide_mask_path) if wide_mask_path else None
        self.is_train = is_train
        self.use_sdf = use_sdf
        self.h5_file = None
        self.sdf_file = None
        self.wide_mask_file = None

        with h5py.File(self.h5_path, "r") as f:
            all_keys = sorted(list(f["images"].keys()))

        if keys is not None:
            self.keys = list(keys)
        elif indices is not None:
            self.keys = [all_keys[i] for i in indices]
        else:
            self.keys = all_keys

        if self.is_train:
            rotation_corrected_sdf = self.use_sdf and self.wide_mask_path is not None
            if rotation_corrected_sdf:
                # Rotate the wide mask alongside image/label (same shared random
                # draw), recompute the SDF from the rotated wide mask after the
                # transform runs, instead of bilinear-interpolating a precomputed
                # SDF field.
                transform_keys = ["image", "label", "wide_mask"]
                rotate_modes = ["bilinear", "nearest", "nearest"]
            else:
                transform_keys = ["image", "label", "sdf"] if self.use_sdf else ["image", "label"]
                rotate_modes = ["bilinear", "nearest", "bilinear"] if self.use_sdf else ["bilinear", "nearest"]

            self.spatial_transforms = MT.Compose([
                MT.RandRotated(keys=transform_keys, range_x=0.3, range_y=0.3, range_z=0.3, prob=0.5,
                               mode=rotate_modes),
                MT.RandFlipd(keys=transform_keys, prob=0.5, spatial_axis=0),
                MT.RandFlipd(keys=transform_keys, prob=0.5, spatial_axis=1),
                MT.RandFlipd(keys=transform_keys, prob=0.5, spatial_axis=2),
                MT.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
            ])
        else:
            self.spatial_transforms = None

    def _init_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        rotation_corrected_sdf = self.use_sdf and self.wide_mask_path is not None and self.is_train
        if self.use_sdf and not rotation_corrected_sdf and self.sdf_file is None:
            self.sdf_file = h5py.File(self.sdf_path, "r")
        if rotation_corrected_sdf and self.wide_mask_file is None:
            self.wide_mask_file = h5py.File(self.wide_mask_path, "r")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_h5()
        key = self.keys[idx]

        patch = self.h5_file["images"][key][()]
        mask = self.h5_file["masks"][key][()]

        patch = torch.from_numpy(patch).float().clamp(config.HU_MIN, config.HU_MAX)
        patch = (patch - config.HU_MIN) / (config.HU_MAX - config.HU_MIN)
        mask = torch.from_numpy(mask).float()

        patch = patch.unsqueeze(0)
        mask = mask.unsqueeze(0)

        data_dict = {"image": patch, "label": mask}

        rotation_corrected_sdf = self.use_sdf and self.wide_mask_path is not None and self.is_train
        if rotation_corrected_sdf:
            wide_mask = self.wide_mask_file["wide_masks"][key][()]
            data_dict["wide_mask"] = torch.from_numpy(wide_mask.astype(np.float32)).unsqueeze(0)
        elif self.use_sdf:
            sdf = self.sdf_file["sdf"][key][()]
            sdf = torch.from_numpy(sdf).float().unsqueeze(0)
            data_dict["sdf"] = sdf

        if self.spatial_transforms:
            data_dict = self.spatial_transforms(data_dict)

        if rotation_corrected_sdf:
            rotated_wide = data_dict["wide_mask"].squeeze(0).numpy() > 0.5
            sdf_full = _signed_distance(rotated_wide)
            sdf_64 = _center_crop(sdf_full, 64)
            sdf_64 = np.clip(sdf_64, -20.0, 20.0).astype(np.float32)
            data_dict["sdf"] = torch.from_numpy(sdf_64).unsqueeze(0)

        if self.use_sdf:
            return data_dict["image"], data_dict["label"], data_dict["sdf"]
        return data_dict["image"], data_dict["label"]

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
        if self.sdf_file is not None:
            self.sdf_file.close()