from typing import Dict, List, override

import torch
from torch import nn

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder


class IdentityEncoder(BaseGeoEncoder):
    def __init__(
        self,
        geo_data_name="aef",
    ) -> None:
        """Encoder to avreage tile values into a 1D vector.

        :param geo_data_name: modality name
        """
        super().__init__()

        self.dict_n_bands_default = {"s2": 4, "aef": 64, "tessera": 128}
        self.allowed_geo_data_names: list[str] = list(self.dict_n_bands_default.keys())
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

    @override
    def _setup(self) -> List[str]:
        """Configures modules and returns newly initialised, trainable module names."""

        self.output_dim = None
        self.geo_encoder = nn.Identity()
        return []

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Data forward pass through the encoder."""
        tile = batch.get("eo", {}).get(self.geo_data_name)

        if self.output_dim is None:
            self.output_dim = tile.shape

        feats = self.geo_encoder(tile)
        if self.extra_projector:
            feats = self.extra_projector(feats)
        return feats

    @property
    def device(self):
        return
