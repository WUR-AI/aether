from typing import Dict

import torch
from torch import nn

from models.components.text_encoders.base_text_encoder import BaseTextEncoder


class RemoteCLIPTextEncoder(BaseTextEncoder):
    """Remote Clip Text Encoder."""

    def __init__(self, model: nn.Module, tokenizer, out_dim: int):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model

        self.output_dim = out_dim

    def _setup(self):
        pass

    def forward(self, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        """Forward function through the Remote Clip Text Encoder."""
        text_input = batch["text"]

        if isinstance(text_input[0], str):
            text_input = [text_input]
            mode = "singular"
        else:
            mode = "multiple"

        avr_embeds = []
        for captions_per_row in text_input:

            # Tokenize and embed
            text = self.tokenizer(captions_per_row).to(self.device)
            text_embeds = self.model.encode_text(text)

            # Project
            if self.extra_projector is not None:
                text_embeds = self.extra_projector(text_embeds)

            if mode == "multiple":
                avr_embeds.append(text_embeds.mean(dim=0))

        if mode == "multiple":
            text_embeds = torch.stack(avr_embeds, dim=0)

        return text_embeds
