"""
unetr_paper.py

MONAI-backed UNETR wrapper for the LIDC-IDRI pulmonary nodule segmentation
benchmark, built to sit alongside `unet3d_paper.py` and be swapped in via
the same `--architecture` mechanism in `train.py`.

Reference:
    Hatamizadeh et al., "UNETR: Transformers for 3D Medical Image
    Segmentation", WACV 2022. https://arxiv.org/abs/2103.10504

Design notes / assumptions -- confirm these match your pipeline before
running anything on the cluster, since I built this from the project's
documented conventions, not the actual `unet3d_paper.py` / `train.py`
source:

  - Input: single-channel (CT intensity) 64x64x64 patches, matching the
    existing `lidc_patches_int16.h5` pipeline. No resizing is done here --
    if your DataLoader already produces (B, 1, 64, 64, 64) tensors for
    UNet3DPaper, no changes are needed upstream.
  - Output: single-channel RAW LOGITS (no sigmoid applied here). This
    assumes UNet3DPaper also returns raw logits and relies on
    DiceFocalLoss(sigmoid=True) to apply the activation, matching this
    project's documented focal-loss output-bias calibration. If
    UNet3DPaper instead applies sigmoid inside its own forward(), do the
    same here so both architectures are trained under an identical loss
    interface.
  - MONAI's UNETR fixes its internal ViT patch size at 16x16x16, so a 64
    voxel edge length gives a 4x4x4 = 64-token sequence. This is a real
    architectural difference from the 96 or 128 cubes UNETR was originally
    benchmarked at (much shorter sequence, less spatial context per
    token) -- worth a line in your methods/limitations section.
  - feature_size/hidden_size/mlp_dim/num_heads (16/768/3072/12) are MONAI's
    own defaults and match the "ViT-Base" configuration from the original
    paper -- not tuned for this dataset.
  - `proj_type` is the current (MONAI >=1.2) argument name for what used to
    be called `pos_embed`. This wrapper introspects the installed MONAI
    version and uses whichever name it actually supports, so it works
    either way without you needing to check `monai.__version__` yourself.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
from monai.networks.nets import UNETR as _MonaiUNETR


class UNETRPaper(nn.Module):
    """
    Thin, explicit wrapper around monai.networks.nets.UNETR so this
    project's architecture files (`unet3d_paper.py`, `unetr_paper.py`)
    share a common, predictable interface: single-channel in, single-
    channel logits out, fixed 64^3 patches.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 64,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        proj_type: str = "conv",
    ) -> None:
        super().__init__()

        if img_size % 16 != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by 16 -- MONAI's "
                "UNETR fixes its internal ViT patch size at 16x16x16."
            )

        common_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=3,
        )

        # MONAI renamed `pos_embed` -> `proj_type` in 1.2 (removed in 1.4).
        # Introspect the installed signature rather than guessing the
        # version -- this is what actually needs to work unattended on the
        # air-gapped cluster, where you can't just pip install a fix.
        sig_params = inspect.signature(_MonaiUNETR.__init__).parameters
        if "proj_type" in sig_params:
            common_kwargs["proj_type"] = proj_type
        elif "pos_embed" in sig_params:
            common_kwargs["pos_embed"] = "perceptron" if proj_type == "perceptron" else "conv"
        else:
            raise RuntimeError(
                "Installed MONAI's UNETR has neither `proj_type` nor "
                "`pos_embed` in its signature -- check `import monai; "
                "monai.__version__` and confirm this MONAI build actually "
                "ships UNETR."
            )

        self.net = _MonaiUNETR(**common_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, D, H, W) -> (B, out_channels, D, H, W), raw logits
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick self-test: confirms the model builds and a forward pass runs at
    # this project's actual patch size, before touching the cluster.
    model = UNETRPaper(in_channels=1, out_channels=1, img_size=64)
    dummy = torch.randn(2, 1, 64, 64, 64)
    out = model(dummy)
    n_params = count_parameters(model)

    print(f"Output shape: {tuple(out.shape)}  (expected: (2, 1, 64, 64, 64))")
    print(f"Trainable parameters: {n_params:,}  (UNet3DPaper is ~19M, for comparison)")