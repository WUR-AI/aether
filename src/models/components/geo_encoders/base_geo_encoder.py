from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseGeoEncoder(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.geo_encoder: nn.Module | None = None
        self.output_dim: int | None = None

        # placeholders
        self.allowed_geo_data_names: list[str] | None = None
        self.geo_data_name: str | None = None
        self.extra_projector: nn.Module | None = None

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device | None:
        devices = {p.device for p in self.parameters()}
        if len(devices) > 1:
            raise RuntimeError("GEO encoder is on multiple devices")
        elif len(devices) == 0:
            return None
        return devices.pop()

    @property
    def dtype(self) -> torch.dtype | None:
        dtypes = {p.dtype for p in self.parameters()}
        if len(dtypes) > 1:
            raise RuntimeError("GEO encoder has multiple dtypes")
        elif len(dtypes) == 0:
            return None
        return dtypes.pop()

    @abstractmethod
    def setup(self) -> list[str]:
        """Configures networks, data-dependent parts.

        Gets called in model.setup() method. Returns names of any new module configured to be added
        to the trainable modules list.
        """
        pass

    def add_projector(self, projected_dim: int) -> None:
        """Adds an extra linear projection layer to the geo encoder.

        NB: is not used by default, needs to be called explicitly in forward().
        """
        self.extra_projector = nn.Linear(self.output_dim, projected_dim, dtype=self.dtype)
        print(
            f"Extra linear projection layer added with mapping dimension {self.output_dim} to {projected_dim}"
        )
        self.output_dim = projected_dim
