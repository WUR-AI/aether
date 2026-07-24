from typing import Dict

import torch
import torchvision.transforms as T
from torch import nn

from models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class RemoteClipImgEncoder(BaseGeoEncoder):
    """Remote Clip Image Encoder."""

    def __init__(
        self,
        geo_encoder: nn.Module,
        out_dim: int,
        clip_percentile=(2, 98),
        geo_data_name="s2",
    ):
        super().__init__()
        self.allowed_geo_data_names = ["s2"]
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

        self.geo_encoder = geo_encoder

        self.clip_percentile = clip_percentile
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
        """Forward function through the Remote Clip Image Encoder."""

        # Get images
        img = batch["eo"]["s2"]

        # Image batch processing
        B, C, H, W = img.shape
        flat = img.reshape(B, C, -1)

        lo = torch.quantile(flat, self.clip_percentile[0] / 100, dim=-1, keepdim=True)
        hi = torch.quantile(flat, self.clip_percentile[1] / 100, dim=-1, keepdim=True)
        lo = lo.unsqueeze(-1)
        hi = hi.unsqueeze(-1)

        img = torch.clamp((img - lo) / (hi - lo + 1e-6), 0, 1)
        img = self.resize_crop(img)
        img = self.normalize(img)

        feats = self.geo_encoder(img)
        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats
