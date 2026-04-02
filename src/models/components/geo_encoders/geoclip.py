from typing import Dict, List, override

import torch
from geoclip import LocationEncoder
from torch.nn import functional as F

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder


class GeoClipCoordinateEncoder(BaseGeoEncoder):
    def __init__(
        self,
        geo_data_name="coords",
    ) -> None:
        super().__init__()

        self.allowed_geo_data_names = ["coords"]
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

    @override
    def setup(self) -> List[str]:
        self.geo_encoder = LocationEncoder()
        self.output_dim = self.geo_encoder.LocEnc0.head[0].out_features
        print("Model setup with GeoClip coordinate encoder")
        return []

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        coords = batch.get("eo", {}).get("coords")

        dtype = self.dtype
        if coords.dtype != dtype:
            coords = coords.to(dtype)
        feats = self.geo_encoder(coords)
        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats.to(dtype)


if __name__ == "__main__":
    _ = GeoClipCoordinateEncoder(None)
