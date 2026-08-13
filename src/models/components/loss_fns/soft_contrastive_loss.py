import json
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from src.data.base_datamodule import BaseDataModule
from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class SoftContrastiveLoss(BaseLossFn):
    def __init__(self, stats_file: str, temperature: float = 0.07, sigma: float = 1.0):
        """Soft-Contrastive Loss Function which allows environmentally similar locations to be
        treated as soft positives.

        :param temperature: Soft-Contrastive Loss Function Temperature
        :param sigma: Soft-Contrastive Loss Function Strength
        :param stats_file: path to a json file with statistics of the aux cols (train split).
        """
        super().__init__()

        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
        self.sigma = sigma

        self.stats = json.load(open(stats_file))

        self.name = "SoftContrastiveLoss"

    @override
    def setup(self, datamodule: BaseDataModule, device: torch.device):
        """Extract auxiliary value statistics into tensors with correct column id sequence."""
        max_id = len(datamodule.caption_builder.column_to_metadata_map["aux"])
        means = torch.zeros(max_id)
        stds = torch.ones(max_id)

        for name, stats in self.stats.items():
            # Synchronise aux col names into proper col ids
            idx = datamodule.caption_builder.column_to_metadata_map["aux"][name]["id"]
            means[idx] = stats["mean"]
            stds[idx] = stats["std"]

        self.means = means
        self.means = self.means.to(device)
        self.stds = stds + 1e-8
        self.stds = self.stds.to(device)

    @override
    def forward(
        self,
        eo_mod: torch.Tensor,
        text_mod: torch.Tensor,
        aux_values: torch.Tensor,
        aux_ids_per_caption: List[List[int]],
        mode: str | None = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward computation."""
        # Get target matrix
        T = self._get_soft_target_matrix(
            aux_values=aux_values, aux_ids_per_caption=aux_ids_per_caption
        )

        # Normalize targets
        T_eo2text = T / T.sum(dim=1, keepdim=True)
        T_text2eo = T / T.sum(dim=0, keepdim=True)

        # Normalise inputs
        eo_mod = F.normalize(eo_mod, dim=-1)
        text_mod = F.normalize(text_mod, dim=-1)

        # Clip temperature to not exceed 100
        temperature = torch.clamp(self.log_temp.exp(), max=100)

        # Get cosine similarity
        dot_product = (eo_mod @ text_mod.T) / temperature

        # Cross entropy: sum(-target * log_prob)
        loss_eo2text = -(T_eo2text * F.log_softmax(dot_product, dim=1)).sum(dim=1).mean()
        loss_text2eo = -(T_text2eo * F.log_softmax(dot_product, dim=0)).sum(dim=0).mean()

        loss = (loss_eo2text + loss_text2eo) / 2
        if "return_label" in kwargs:
            return {f"{mode}_{self.name}": loss}
        else:
            return loss

    def _get_soft_target_matrix(
        self,
        aux_values: torch.Tensor,
        aux_ids_per_caption: list[list[int]] | None,
    ) -> torch.Tensor:
        """Puts together a target matrix based on auxiliary value similarity (either all, or
        specific per caption template).

        :param aux_values: auxiliary column values (standardised).
        :param aux_ids_per_caption: list of ids of auxiliary columns used per caption template.
        :return: target matrix which tells how each row should be similar to column based on aux
            values (all or specific ones per caption template).
        """
        batch_size, n_aux_cols = aux_values.shape
        device = aux_values.device

        # Standardise aux values
        if self.stats is not None:
            aux_values = (aux_values - self.means) / self.stds

        # Soft targets based on the squared difference between location i and j for all aux values
        diffs = (aux_values.unsqueeze(1) - aux_values.unsqueeze(0)) ** 2

        # Create a mask based on aux_columns used per location caption
        if aux_ids_per_caption:
            # Create a mask
            mask = torch.zeros((batch_size, n_aux_cols), dtype=aux_values.dtype, device=device)
            for j, used_cols in enumerate(aux_ids_per_caption):
                if len(used_cols) > 0:
                    mask[j, used_cols] = 1.0
            mask = mask.unsqueeze(0)

            # Calculate the masked average distance for the selected aux cols
            dist = (diffs * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        else:
            # Distance based on all aux columns
            dist = diffs.mean(dim=-1)

        # Convert distances to similarities using a Gaussian kernel
        T = torch.exp(-dist / (2 * self.sigma**2))  # (N, N)
        return T
