from typing import Dict, override

import torch
from geoclip import LocationEncoder
from torch.nn import functional as F

from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder


class GeoClipCoordinateEncoder(BaseEOEncoder):
    def __init__(
        self,
        eo_data_name="coords",
    ) -> None:
        super().__init__()
        self.eo_encoder = LocationEncoder()
        self.output_dim = self.eo_encoder.LocEnc0.head[0].out_features
        self.allowed_eo_data_names = ["coords"]
        assert (
            eo_data_name in self.allowed_eo_data_names
        ), f"eo_data_name must be one of {self.allowed_eo_data_names}, got {eo_data_name}"
        self.eo_data_name = eo_data_name

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        coords = batch.get("eo", {}).get("coords")

        dtype = self.dtype
        if coords.dtype != dtype:
            coords = coords.to(dtype)
        feats = self.eo_encoder(coords)

        return feats.to(dtype)


if __name__ == "__main__":
    _ = GeoClipCoordinateEncoder(None)
