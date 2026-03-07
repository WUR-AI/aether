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

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device:
        devices = {p.device for p in self.parameters()}
        if len(devices) != 1:
            raise RuntimeError("GEO encoder is on multiple devices")
        return devices.pop()

    @property
    def dtype(self) -> torch.dtype:
        dtypes = {p.dtype for p in self.parameters()}
        if len(dtypes) != 1:
            raise RuntimeError("GEO encoder has multiple dtypes")
        return dtypes.pop()


if __name__ == "__main__":
    _ = BaseGeoEncoder(None)
