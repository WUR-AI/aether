from typing import override

import torch
import torch.nn.functional as F

from src.models.components.metrics.base_metrics import BaseMetrics


class CosineSimilarities(BaseMetrics):
    def __init__(self, k_list=None) -> None:
        super().__init__()
        self.k_list = k_list or [1, 5, 10]

    @override
    def forward(
        self,
        mode: str,
        eo_feats: torch.Tensor,
        text_feats: torch.Tensor,
        local_batch_size: int,
        **kwargs,
    ):
        """Calculate cosine similarity between eo and text embeddings and logs it."""

        # Similarity matrix
        cos_sim_matrix = F.cosine_similarity(eo_feats[:, None, :], text_feats[None, :, :], dim=-1)

        # Average for positive and negative pairs
        # TODO change label option if we change what gets treated to be pos/neg
        id_matrix = torch.eye(cos_sim_matrix.shape[0], dtype=torch.bool)
        pos_sim = cos_sim_matrix[id_matrix]
        neg_sim = cos_sim_matrix[~id_matrix]

        # Average
        avr_sim = torch.mean(cos_sim_matrix)
        sub_neg_sim = neg_sim[
            torch.randperm(len(neg_sim))[: len(pos_sim)]
        ]  # pick same amount of negatives as positives
        balanced_sim = torch.cat([pos_sim, sub_neg_sim], dim=0)
        balanced_avr_sim = torch.mean(balanced_sim)

        return {
            f"{mode}_avr_sim": avr_sim,
            f"{mode}_avr_sim_balanced": balanced_avr_sim,
            f"{mode}_pos_sim": torch.mean(pos_sim),
            f"{mode}_neg_sim": torch.mean(neg_sim),
        }
