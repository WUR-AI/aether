from typing import Dict, List, override

import torch
from huggingface_hub import hf_hub_download
from satclip.load import get_satclip

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder


class SatClipCoordinateEncoder(BaseGeoEncoder):
    def __init__(
        self,
        geo_data_name="coords",
        hf_cache_dir: str = "../.cache",
        accelerator: torch.device = torch.device("cpu"),
    ) -> None:
        """SatClip coordinate encoder :param geo_data_name: type of geo data used for this encoder
        (supports only coordinates) :param hf_cache_dir: hugging face cache directory to store data
        :param accelerator: where to load model (as it is float64, mps is not supported)"""
        super().__init__()

        self.allowed_geo_data_names = ["coords"]
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

        self.cache_dir = hf_cache_dir
        assert accelerator != torch.device("mps"), f"accelerator {accelerator} is not supported"
        self.accelerator = accelerator

    @override
    def _setup(self) -> List[str]:
        """Setup satclip encoder from hugging face hub and set output dimension."""
        self.geo_encoder = get_satclip(
            hf_hub_download(
                "microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt", cache_dir=self.cache_dir
            ),
            device=self.accelerator,
        )

        self.output_dim = self.geo_encoder.nnet.last_layer.dim_out
        return []

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of satclip encoder."""

        coords = batch.get("eo", {}).get("coords")

        # Swap coordinates
        coords = coords[:, [1, 0]]

        # SatClip needs float64
        dtype = self.dtype
        if coords.dtype != dtype:
            coords = coords.to(dtype)

        feats = self.geo_encoder(coords)
        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats
