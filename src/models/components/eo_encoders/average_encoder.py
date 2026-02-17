from typing import Dict, override

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder


class AverageEncoder(BaseEOEncoder):
    def __init__(
        self,
        output_dim: int | None = None,
        eo_data_name="aef",
    ) -> None:
        super().__init__()

        dict_n_bands_default = {"s2": 4, "aef": 64, "tessera": 128}
        self.allowed_eo_data_names: list[str] = list(dict_n_bands_default.keys())

        assert (
            eo_data_name in dict_n_bands_default
        ), f"eo_data_name must be one of {self.allowed_eo_data_names}, got {eo_data_name}"
        self.eo_data_name = eo_data_name

        if output_dim is None or output_dim == dict_n_bands_default[eo_data_name]:
            self.output_dim = dict_n_bands_default[eo_data_name]
            self.extra_projector = None
            self.eo_encoder = self._average
        else:
            assert (
                type(output_dim) is int and output_dim > 0
            ), f"output_dim must be positive int, got {output_dim}"
            self.output_dim = output_dim
            self.extra_projector = nn.Linear(dict_n_bands_default[eo_data_name], output_dim)
            self.eo_encoder = self._average_and_project

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
        eo_data = batch.get("eo", {})
        dtype = self.dtype
        if eo_data.dtype != dtype:
            eo_data = eo_data.to(dtype)
        feats = self.eo_encoder(eo_data[self.eo_data_name])
        # n_nans = torch.sum(torch.isnan(feats)).item()
        # assert (
        #     n_nans == 0
        # ), f"AverageEncoder output contains {n_nans}/{feats.numel()} NaNs PRIOR to normalization with data min {eo_data[self.eo_data_name].min()} and max {eo_data[self.eo_data_name].max()}."

        return feats.to(dtype)


if __name__ == "__main__":
    _ = AverageEncoder(None, None)
