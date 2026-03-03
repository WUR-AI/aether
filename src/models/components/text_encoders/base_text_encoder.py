from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseTextEncoder(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.processor: nn.Module | None = None
        self.model: nn.Module = None
        self.projector: nn.Module | None = None
        self.output_dim: int | None = None
        self.extra_projector: nn.Module | None = None

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        pass

    def add_projector(self, projected_dim: int) -> None:
        """Adds an extra linear projection layer to the text encoder.

        NB: is not used by default, needs to be called explicitly in forward().
        """
        self.extra_projector = nn.Linear(self.output_dim, projected_dim, dtype=self.dtype)
        print(
            f"Extra linear projection layer added with mapping dimension {self.output_dim} to {projected_dim}"
        )
        self.output_dim = projected_dim

    @property
    def device(self) -> torch.device:
        devices = {p.device for p in self.parameters()}
        if len(devices) != 1:
            raise RuntimeError("Text encoder is on multiple devices")
        return devices.pop()

    @property
    def dtype(self) -> torch.dtype:
        dtypes = {p.dtype for p in self.parameters()}
        if len(dtypes) != 1:
            raise RuntimeError("Text encoder has multiple dtypes")
        return dtypes.pop()
