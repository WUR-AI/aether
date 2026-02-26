from typing import Dict, override

import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class ClipLoss(BaseLossFn):
    def __init__(
        self,
        temperature: float,
    ) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.name = "CLIPLoss"

    @override
    def forward(
        self,
        eo_mod: torch.Tensor,
        text_mod: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        # Normalise inputs
        eo_mod = F.normalize(eo_mod, dim=-1)
        text_mod = F.normalize(text_mod, dim=-1)

        # Clip temperature to not exceed 100
        temperature = torch.clamp(self.log_temp.exp(), max=100)

        # Get cosine similarity
        dot_product = (eo_mod @ text_mod.T) / temperature

        # Handle targets for contrastive loss
        targets = torch.arange(eo_mod.shape[0], device=eo_mod.device)
        loss1 = F.cross_entropy(dot_product, targets)
        loss2 = F.cross_entropy(dot_product.T, targets)

        loss = (loss1 + loss2) / 2
        if "return_label" in kwargs:
            return {self.name: loss}
        else:
            return loss


if __name__ == "__main__":
    _ = ClipLoss(None)
