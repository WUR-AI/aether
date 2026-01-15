from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseEOEncoder(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.eo_encoder: nn.Module | None = None
        self.output_dim: int | None = None

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass
