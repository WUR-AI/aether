from abc import ABC, abstractmethod
from typing import Any, Dict

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
        mode: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def setup(self, **kwargs: Any) -> None:
        """Setup method for losses in case they need some parameters from the dataset (e.g., for
        standardisation in the soft contrastive loss."""
        pass
