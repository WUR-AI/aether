from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn

class BaseTextEncoder(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.processor: nn.Module | None = None
        self.text_model: nn.Module = None
        self.projector: nn.Module | None = None
        self.output_dim: int | None = None
        self.extra_projector: nn.Module | None = None

    @abstractmethod
    def forward(
            self,
            batch: Dict[str, torch.Tensor],
            mode: str
    ) -> torch.Tensor:
        pass

    def add_projector(self, projected_dim: int) -> None:
        self.extra_projector = nn.Linear(self.output_dim, projected_dim)
        self.output_dim = projected_dim