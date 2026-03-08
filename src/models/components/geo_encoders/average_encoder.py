from typing import Dict, override

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder


class AverageEncoder(BaseGeoEncoder):
    def __init__(
        self,
        output_dim: int | None = None,
        geo_data_name="aef",
    ) -> None:
        super().__init__()

        dict_n_bands_default = {"s2": 4, "aef": 64, "tessera": 128}
        self.allowed_geo_data_names: list[str] = list(dict_n_bands_default.keys())

        assert (
            geo_data_name in dict_n_bands_default
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

        if output_dim is None or output_dim == dict_n_bands_default[geo_data_name]:
            self.output_dim = dict_n_bands_default[geo_data_name]
            self.extra_projector = None
            self.geo_encoder = self._average
        else:
            assert (
                type(output_dim) is int and output_dim > 0
            ), f"output_dim must be positive int, got {output_dim}"
            self.output_dim = output_dim
            self.extra_projector = nn.Linear(dict_n_bands_default[geo_data_name], output_dim)
            self.geo_encoder = self._average_and_project

    def _average(self, x: torch.Tensor) -> torch.Tensor:
        """Averages the input tensor over spatial dimensions.

        :param x: input tensor of shape (B, C, H, W)
        :return: averaged tensor of shape (B, C)
        """
        return x.mean(dim=(-2, -1))

    def _average_and_project(self, x: torch.Tensor) -> torch.Tensor:
        """Averages the input tensor over spatial dimensions and projects to output_dim.

        :param x: input tensor of shape (B, C, H, W)
        :return: projected tensor of shape (B, output_dim)
        """
        x_avg = x.mean(dim=(-2, -1))
        x_proj = self.extra_projector(x_avg)
        return x_proj

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tile = batch.get("eo", {}).get(self.geo_data_name)
        # Determine target dtype from parameters when available (e.g. when the
        # optional projection layer exists); otherwise keep the input dtype.
        params = list(self.parameters())
        dtype = params[0].dtype if params else tile.dtype
        if tile.dtype != dtype:
            tile = tile.to(dtype)
        feats = self.geo_encoder(tile)
        return feats.to(dtype)


if __name__ == "__main__":
    _ = AverageEncoder(None, None)
