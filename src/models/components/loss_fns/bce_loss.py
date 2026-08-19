from typing import Dict, override

import torch
import torch.nn.functional as F

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class BCELoss(BaseLossFn):
    def __init__(self, weighting: bool = False, pos_weight_scale: float = 1.0) -> None:
        super().__init__()

        self.name: str = "bce_loss"
        self.weighting = weighting
        self.pos_weight_scale = pos_weight_scale

    @override
    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        mode: str | None = None,
        **kwargs,
    ) -> torch.Tensor or Dict[str, torch.Tensor]:
        """Forward pass to get BCE loss."""

        labels = labels if labels is not None else batch.get("target")
        labels = labels.to(pred.dtype)

        if self.weighting:
            weight = (labels > 0).float() * self.pos_weight_scale + 1.0
            loss = F.binary_cross_entropy(pred, labels, weight=weight, reduction="mean")
        else:
            loss = F.binary_cross_entropy(pred, labels, reduction="mean")

        if "return_label" in kwargs:
            return {f"{mode}_{self.name}": loss}
        else:
            return loss


if __name__ == "__main__":
    _ = BCELoss()
