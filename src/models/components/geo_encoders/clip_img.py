from typing import Dict

import torch
import torchvision.transforms as T
from transformers import CLIPVisionModelWithProjection

from models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# HF-hosted CLIP ViT checkpoints
HF_CLIP_MODELS = {
    "ViT-B-32": ("openai/clip-vit-base-patch32", 224, 512),
    "ViT-B-16": ("openai/clip-vit-base-patch16", 224, 512),
    "ViT-L-14": ("openai/clip-vit-large-patch14", 224, 768),
}

# OpenAI CLIP ResNet checkpoints -> not ported into `transformers`
OPEN_CLIP_RN_MODELS = {
    "RN50": ("RN50", 224, 1024),
}


class ClipImgEncoder(BaseGeoEncoder):
    """CLIP Image Encoder."""

    def __init__(
        self,
        geo_data_name: str = "s2",
        model_name: str = "ViT-B-32",
        hf_cache_dir: str = "../.cache",
    ):
        super().__init__()

        self.allowed_geo_data_names = ["s2"]
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name
        self.model_name = model_name

        if model_name in HF_CLIP_MODELS:
            self.backend = "hf"
            pretrained_name, crop_size, self.output_dim = HF_CLIP_MODELS[model_name]
            self.geo_encoder = CLIPVisionModelWithProjection.from_pretrained(
                pretrained_name, cache_dir=hf_cache_dir
            )

        elif model_name in OPEN_CLIP_RN_MODELS:
            import open_clip

            self.backend = "open_clip"
            oc_name, crop_size, self.output_dim = OPEN_CLIP_RN_MODELS[model_name]
            clip_model, _, _ = open_clip.create_model_and_transforms(
                oc_name, pretrained="openai", cache_dir=hf_cache_dir
            )
            self.geo_encoder = clip_model.visual
        else:
            allowed = list(HF_CLIP_MODELS) + list(OPEN_CLIP_RN_MODELS)
            raise ValueError(f"Model {model_name} is not supported. Choose from {allowed}")

        self.resize_crop = T.Compose(
            [
                T.Resize(crop_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(crop_size),
            ]
        )
        self.normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

    def _setup(self):
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward function through the CLIP Image Encoder."""

        img = batch["eo"][self.geo_data_name]

        img = self.resize_crop(img)
        img = self.normalize(img)

        if self.backend == "hf":
            feats = self.geo_encoder(pixel_values=img).image_embeds
        else:
            feats = self.geo_encoder(img)

        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats
