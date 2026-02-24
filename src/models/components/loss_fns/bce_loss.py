from typing import Dict, override

import torch
from torch import nn

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class BCELoss(BaseLossFn):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCELoss(reduction="mean")
        self.name: str = "bce_loss"

    @override
    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor or Dict[str, torch.Tensor]:

        labels = labels if labels is not None else batch.get("target")
        loss = self.criterion(pred, labels)

        if "return_label" in kwargs:
            return {self.name: loss}
        else:
            return loss


if __name__ == "__main__":
    _ = BCELoss()
