from typing import Dict, override

import torch
from geoclip import LocationEncoder
from torch.nn import functional as F

from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder


class GeoClipCoordinateEncoder(BaseEOEncoder):
    def __init__(self, output_normalization="l2") -> None:
        super().__init__()
        self.eo_encoder = LocationEncoder()
        self.output_dim = self.eo_encoder.LocEnc0.head[0].out_features

        self.output_normalization = output_normalization
        if self.output_normalization not in ["l2", "none"]:
            raise ValueError(f"Unsupported output_normalization: {self.output_normalization}")

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

        if self.output_normalization == "l2":
            feats = F.normalize(feats, p=2, dim=-1)  # L2 normalization (per feature vector)

        return feats.to(dtype)


if __name__ == "__main__":
    _ = GeoClipCoordinateEncoder(None)
