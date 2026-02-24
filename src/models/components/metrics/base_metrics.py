from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseMetrics(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
            self, pred: torch.Tensor,
            labels: torch.Tensor,
            batch: Dict[str, torch.Tensor] | None = None,
            **kwargs
    ) -> Dict[str, torch.float]:
        pass
