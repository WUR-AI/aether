from typing import Dict

import torch
from torch import nn

from models.components.text_encoders.base_text_encoder import BaseTextEncoder


class DOFACLIPTextEncoder(BaseTextEncoder):
    """Remote Clip Text Encoder."""

    def __init__(self, model: nn.Module, tokenizer, out_dim: int, context_len: int):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model
        self.context_len = context_len

        self.output_dim = out_dim

    def _setup(self):
        pass

    def forward(self, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        """Forward function through the Remote Clip Text Encoder."""
        text_input = batch["text"]

        if mode == "train":
            text_input = [text_input]

        avr_embeds = []
        for captions_per_row in text_input:

            # Tokenize and embed
            text = self.tokenizer(captions_per_row, context_length=self.context_len).to(
                self.device
            )
            text_embeds = self.model.encode_text(text)

            # Project
            if self.extra_projector is not None:
                text_embeds = self.extra_projector(text_embeds)

            if mode != "train":
                avr_embeds.append(text_embeds.mean(dim=0))

        if mode != "train":
            text_embeds = torch.stack(avr_embeds, dim=0)

        return text_embeds
