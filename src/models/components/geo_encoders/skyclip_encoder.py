from typing import Dict

import torch
import torchvision.transforms as T
from torch import nn

from models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class SkyClipImgEncoder(BaseGeoEncoder):
    """SkyCLIP Image Encoder."""

    def __init__(
        self,
        geo_encoder: nn.Module,
        out_dim: int,
        geo_data_name="s2",
    ):
        super().__init__()
        self.allowed_geo_data_names = ["s2"]
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

        self.geo_encoder = geo_encoder

        self.resize_crop = T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(224),
            ]
        )
        self.normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

        self.output_dim = out_dim

    def _setup(self):
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward function through the SkyCLIP Image Encoder."""

        # Get images
        img = batch["eo"]["s2"]

        img = self.resize_crop(img)
        img = self.normalize(img)

        feats = self.geo_encoder(img)
        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats
