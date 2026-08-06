import torch
import torch.nn as nn


class BoundaryLoss(nn.Module):
    """
    Boundary Loss (Kervadec et al., 2019), stabilized with SDF clipping.

    Raw signed distance fields can have large magnitude far from the object
    boundary (tens of voxels in this dataset). Without clipping, the loss
    is dominated by distant background voxels, creating a strong incentive
    for the model to collapse to predicting nothing (empty volumes), since
    that trivially minimizes (pred_probs * sdf).mean() regardless of actual
    segmentation quality. Clipping bounds this contribution.
    """

    def __init__(self, clip_value=20.0):
        super().__init__()
        self.clip_value = clip_value

    def forward(self, pred_probs, sdf):
        sdf_clipped = torch.clamp(sdf, min=-self.clip_value, max=self.clip_value)
        loss = (pred_probs * sdf_clipped).mean()
        return loss


class DiceFocalBoundaryLoss(nn.Module):
    """
    Wraps an existing MONAI DiceFocalLoss and adds a weighted BoundaryLoss term.
    boundary_weight is mutable (set externally by train.py) to support a
    warmup schedule — the boundary term should not be applied at full
    strength before the model has learned basic region-level segmentation
    via Dice/Focal alone.
    """

    def __init__(self, dice_focal_loss, boundary_weight=0.05, clip_value=20.0):
        super().__init__()
        self.dice_focal_loss = dice_focal_loss
        self.boundary_loss = BoundaryLoss(clip_value=clip_value)
        self.boundary_weight = boundary_weight

    def forward(self, logits, target_mask, sdf):
        base_loss = self.dice_focal_loss(logits, target_mask)
        pred_probs = torch.sigmoid(logits)
        b_loss = self.boundary_loss(pred_probs, sdf)
        return base_loss + self.boundary_weight * b_loss