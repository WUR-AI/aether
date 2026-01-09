from typing import Dict, override

import torch
from geoclip import GeoCLIP
from transformers import CLIPProcessor, CLIPModel

from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder


class ClipTextEncoder(BaseTextEncoder):
    def __init__(
            self,
            hf_cache_dir: str = '../.cache'
    ) -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-large-patch14',
            use_fast=True,
            cache_dir=hf_cache_dir,
        )
        self.model = CLIPModel.from_pretrained(
            'openai/clip-vit-large-patch14',
            cache_dir=hf_cache_dir,
        )

        self.projector = GeoCLIP().image_encoder.mlp
        self.output_dim = 512

    @override
    def forward(
            self,
            batch: Dict[str, torch.Tensor],
            mode: str
    ) -> torch.Tensor:
        # Get text inputs
        text_input = batch.get('text')

        if mode == 'train':
            text_input = [text_input]
        # Embed text and if not training loop average all templates
        avr_embeds = []
        for captions_per_row in text_input:
            # Tokenize and embed
            text_tokens = self.processor(text=captions_per_row, return_tensors='pt', padding=True)
            device = next(self.model.parameters()).device
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

            text_embeds = self.model.get_text_features(**text_tokens)

            # Project
            if self.projector is not None:
                text_embeds = self.projector(text_embeds)

            if self.extra_projector is not None:
                text_embeds = self.extra_projector(text_embeds)

            if mode != 'train':
                avr_embeds.append(text_embeds.mean(dim=0))

        if mode != 'train':
            text_embeds = torch.stack(avr_embeds, dim=0)

        return text_embeds