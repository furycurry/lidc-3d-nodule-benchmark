import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

class LossGradientInspector:
    """
    Computes L2 gradient norms and relative shares for individual loss terms
    with respect to designated model parameters.
    """
    def __init__(self, model: nn.Module, target_layer_name: Optional[str] = None):
        """
        Args:
            model: PyTorch model being trained.
            target_layer_name: Name of specific layer to evaluate gradients against (e.g. final conv).
                               If None, evaluates across all model parameters.
        """
        self.model = model
        if target_layer_name:
            # Isolate specific layer parameters (e.g., final output layer)
            layer = dict(model.named_modules()).get(target_layer_name)
            if layer is None:
                raise ValueError(f"Layer '{target_layer_name}' not found in model.")
            self.target_params = [p for p in layer.parameters() if p.requires_grad]
        else:
            self.target_params = [p for p in model.parameters() if p.requires_grad]

    def compute_gradient_norms(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calculates L2 gradient norm ||d(L_i)/d(theta)||_2 for each loss term.

        Args:
            loss_dict: Dictionary mapping loss names to their respective scalar Tensors.
                       e.g. {'dice': loss_dice, 'focal': loss_focal, 'boundary': loss_boundary}
        Returns:
            Dictionary containing absolute L2 norms and relative percentage contributions.
        """
        metrics = {}
        total_norm_sum = 0.0

        for name, loss_tensor in loss_dict.items():
            if loss_tensor is None or not loss_tensor.requires_grad or loss_tensor.item() == 0:
                metrics[f"grad_norm/{name}"] = 0.0
                continue

            # Compute gradients for the individual loss component
            grads = torch.autograd.grad(
                outputs=loss_tensor,
                inputs=self.target_params,
                retain_graph=True,
                allow_unused=True
            )

            # Compute L2 norm across all targeted parameters
            sq_norm = sum(g.pow(2).sum().item() for g in grads if g is not None)
            norm = math.sqrt(sq_norm)
            metrics[f"grad_norm/{name}"] = norm
            total_norm_sum += norm

        # Compute relative share (%) of total gradient magnitude
        denom = total_norm_sum if total_norm_sum > 0 else 1e-12
        for name in loss_dict.keys():
            abs_norm = metrics.get(f"grad_norm/{name}", 0.0)
            metrics[f"grad_share_pct/{name}"] = (abs_norm / denom) * 100.0

        return metrics