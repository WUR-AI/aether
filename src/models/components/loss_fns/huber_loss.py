from typing import Dict, override

import torch

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class HuberLoss(BaseLossFn):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.HuberLoss(delta=1.0, reduction="mean")
        self.name = "huber_loss"

    @override
    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        labels = labels if labels is not None else batch.get("target")
        huber_loss = self.criterion(pred, labels)

        if "return_label" in kwargs:
            return {self.name: huber_loss}
        else:
            return huber_loss
