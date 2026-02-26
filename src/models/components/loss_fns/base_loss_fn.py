from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseLossFn(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.criterion: nn.Module | None = None
        self.name: str | None = None

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        pass
