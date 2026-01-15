from typing import override

import torch
from torch import nn

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class BCELoss(BaseLossFn):
    def __init__(self) -> None:
        super().__init__()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.BCELoss(reduction="mean")

    @override
    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, labels)


if __name__ == "__main__":
    _ = BCELoss()
