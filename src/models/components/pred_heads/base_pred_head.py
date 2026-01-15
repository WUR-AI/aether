from abc import ABC, abstractmethod
from typing import final

import torch
from torch import nn


class BasePredictionHead(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.net: nn.Module | None = None
        self.input_dim: int | None = None
        self.output_dim: int | None = None

    @abstractmethod
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pass

    @final
    def set_dim(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def configure_nn(self) -> None:
        pass
