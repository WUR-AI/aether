from typing import override

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class BCELoss(BaseLossFn):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCELoss(reduction="mean")

    @override
    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalise inputs
        return self.criterion(pred, labels)


if __name__ == "__main__":
    _ = BCELoss()
