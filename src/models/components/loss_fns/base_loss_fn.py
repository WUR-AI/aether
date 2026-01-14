from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseLossFn(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.criterion: nn.Module | None = None

    @abstractmethod
    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass
