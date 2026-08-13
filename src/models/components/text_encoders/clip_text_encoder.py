from typing import Dict, override

import open_clip
import torch
from geoclip import GeoCLIP
from transformers import CLIPModel, CLIPProcessor

from src.models.components.text_encoders.base_text_encoder import (
    BaseTextEncoder,
)
from utils.errors import MissingConfigurationError

# Keep these keys in sync with clip_img.py so an (image, text) pair can be
# configured with the same model_name and land in the same embedding space.
HF_CLIP_MODELS = {
    "ViT-B-32": ("openai/clip-vit-base-patch32", 512),
    "ViT-B-16": ("openai/clip-vit-base-patch16", 512),
    "ViT-L-14": ("openai/clip-vit-large-patch14", 768),
}

OPEN_CLIP_RN_MODELS = {"RN50": ("RN50", 1024)}


class ClipTextEncoder(BaseTextEncoder):
    """CLIP Text Encoder."""

    def __init__(
        self,
        hf_cache_dir: str = "../.cache",
        use_geoclip_projector: bool = True,
        model_name: str = "ViT-L-14",  # default GeoCLIP
    ) -> None:
        super().__init__()

        if model_name in HF_CLIP_MODELS:
            self.backend = "hf"
            pretrained_name, embed_dim = HF_CLIP_MODELS[model_name]

            self.processor = CLIPProcessor.from_pretrained(
                pretrained_name,
                use_fast=True,
                cache_dir=hf_cache_dir,
            )
            self.model = CLIPModel.from_pretrained(
                pretrained_name,
                cache_dir=hf_cache_dir,
            )
            self.model.vision_model = None
            self.model.visual_projection = None

        elif model_name in OPEN_CLIP_RN_MODELS:
            self.backend = "open_clip"
            oc_name, embed_dim = OPEN_CLIP_RN_MODELS[model_name]

            self.tokenizer = open_clip.get_tokenizer(oc_name)
            clip_model, _, _ = open_clip.create_model_and_transforms(
                oc_name, pretrained="openai", cache_dir=hf_cache_dir
            )
            clip_model.visual = None  # drop the vision tower, text-only
            self.model = clip_model

        else:
            allowed = list(HF_CLIP_MODELS) + list(OPEN_CLIP_RN_MODELS)
            raise ValueError(f"Model {model_name} is not supported. Choose from {allowed}")

        self.model_name = model_name

        if use_geoclip_projector:
            assert model_name == "ViT-L-14", MissingConfigurationError(
                "GeoCLIP projector needs a ViT-L-14 text encoder"
            )
            self.projector = GeoCLIP().image_encoder.mlp
            self.output_dim = 512
        else:
            self.projector = None
            self.output_dim = embed_dim

    @override
    def forward(self, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        """Forward pass through the text encoder."""
        # Get text inputs
        text_input = batch.get("text")

        if mode == "train":
            text_input = [text_input]
        grad_enabled = any(p.requires_grad for p in self.model.parameters())
        device = next(self.model.parameters()).device

        # Embed text and if not training loop average all templates
        avr_embeds = []
        for captions_per_row in text_input:
            # Tokenize and embed
            if self.backend == "hf":
                text_tokens = self.processor(
                    text=captions_per_row,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
                if grad_enabled:
                    text_embeds = self.model.get_text_features(**text_tokens)
                else:
                    with torch.no_grad():
                        text_embeds = self.model.get_text_features(**text_tokens)

            else:
                text_tokens = self.tokenizer(captions_per_row).to(device)
                if grad_enabled:
                    text_embeds = self.model.encode_text(text_tokens)
                else:
                    with torch.no_grad():
                        text_embeds = self.model.encode_text(text_tokens)

            # Project
            if self.projector is not None:
                text_embeds = self.projector(text_embeds)

            if self.extra_projector is not None:
                text_embeds = self.extra_projector(text_embeds)

            if mode != "train":
                avr_embeds.append(text_embeds.mean(dim=0))

        if mode != "train":
            text_embeds = torch.stack(avr_embeds, dim=0)

        return text_embeds
