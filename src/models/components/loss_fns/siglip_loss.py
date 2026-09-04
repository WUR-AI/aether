from typing import Dict, override

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class SigLipLoss(BaseLossFn):
    def __init__(
        self,
        t_prime: float = 10.0,
        bias: float = -10.0,
    ) -> None:
        """
        Args:
            t_prime: Initial scale factor (similar to 1/temperature in CLIP).
                        The SigLIP paper typically initializes this to 10.0.
            bias: Initial bias term. The SigLIP paper typically initializes this to -10.0.
        # https://github.com/ahmdtaha/distributed_sigmoid_loss/blob/main/rwightman_sigmoid_loss.py
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.log(torch.tensor(t_prime)))
        self.bias = nn.Parameter(torch.tensor(bias))
        self.name = "SigLIPLoss"

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        """Create labels: 1 for matching pairs (diagonal), -1 for non-matching pairs"""
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    @override
    def forward(
        self,
        eo_mod: torch.Tensor,
        text_mod: torch.Tensor,
        mode: str | None = None,
        negative_only: bool = False,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        # Normalise inputs
        eo_mod = F.normalize(eo_mod, dim=-1)
        text_mod = F.normalize(text_mod, dim=-1)

        # Get logits: (x @ y.T) * scale + bias
        scale = self.log_scale.exp()
        logits = (eo_mod @ text_mod.T) * scale
        if self.bias is not None:
            logits += self.bias

        # Create labels
        labels = self.get_ground_truth(eo_mod.device, eo_mod.dtype, eo_mod.shape[0], negative_only)

        # Compute Sigmoid loss
        loss = -F.logsigmoid(labels * logits).sum() / eo_mod.shape[0]

        if "return_label" in kwargs:
            return {f"{mode}_{self.name}": loss}
        else:
            return loss


if __name__ == "__main__":
    _ = SigLipLoss()
