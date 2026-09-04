from typing import Dict

import torch
import torchvision.transforms as T
from torch import nn

from models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder

wvs = {
    # 's2_9b': torch.tensor([0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19]),
    "s2_4c": torch.tensor([0.665, 0.56, 0.49, 0.842]),
    "s2_rgb": torch.tensor([0.665, 0.560, 0.490]),
}


class DOFAClipImgEncoder(BaseGeoEncoder):
    """DOFA Clip Image Encoder."""

    def __init__(
        self,
        geo_encoder: nn.Module,
        out_dim: int,
        geo_data_name="s2_rgb",
    ):
        super().__init__()
        self.allowed_geo_data_names = list(wvs.keys())
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

        self.geo_encoder = geo_encoder

        self.register_buffer("wvs", wvs[geo_data_name])  # moves to gpu automatically

        size = self.geo_encoder.image_size
        self.resize_crop = T.Resize(
            size=size, interpolation=T.InterpolationMode.BICUBIC, antialias=True
        )

        self.output_dim = out_dim

    def _setup(self):
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward function through the Remote Clip Image Encoder."""

        # Get images
        img = batch["eo"]["s2"]

        if not torch.is_floating_point(img):
            img = img.float()

        img = self.resize_crop(img)

        C = img.shape[1]
        normalize = T.Normalize(mean=[0.5] * C, std=[0.5] * C)
        img = normalize(img)

        # Encode
        feats = self.geo_encoder.trunk(img, self.wvs)[0]

        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats
