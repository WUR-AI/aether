from typing import Dict, override

import torch
from geoclip import LocationEncoder

from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder


class GeoClipCoordinateEncoder(BaseEOEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.eo_encoder = LocationEncoder()
        self.output_dim = self.eo_encoder.LocEnc0.head[0].out_features

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        coords = batch.get("eo", {}).get("coords")
        feats = self.eo_encoder(coords)
        return feats


if __name__ == "__main__":
    _ = GeoClipCoordinateEncoder()
