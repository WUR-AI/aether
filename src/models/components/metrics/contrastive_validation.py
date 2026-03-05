from typing import Any, Dict, List, override

import torch

from src.models.components.metrics.base_metrics import BaseMetrics


class RetrievalContrastiveValidation(BaseMetrics):
    def __init__(self, ks: List[Any], concept_configs: List[Any]) -> None:
        """Evaluates how many eo embeddings are retrieved in top-k metrics based the GT labels.

        :param ks: k values for top-k metrics
        :param concept_configs: concept configurations containing details about min/max mode, which
            aux_col to use as GT.
        """
        super().__init__()

        self.concept_configs = concept_configs

        self.ks = ks
        if any("theta_k" in c for c in self.concept_configs):
            self.ks.append("dynamic_k")

    @override
    def forward(
        self,
        similarity_matrix: torch.Tensor,
        aux_values: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Calculates top-k metrics based the GT (aux-derived) labels."""

        aux_vals = aux_values.T

        concept_scores = {}
        for i, configs in enumerate(self.concept_configs):
            avr_scores = {k: [] for k in self.ks}

            idx = configs["id"]
            is_max = configs["is_max"]
            k_threshold = configs.get("theta_k")
            aux_val = aux_vals[idx]

            if k_threshold:
                k_threshold = (
                    sum(aux_val >= k_threshold).item()
                    if is_max
                    else sum(aux_val <= k_threshold).item()
                )

            score = self.topk_rank_agreement(
                aux_val, similarity_matrix[i], self.ks, is_max, k_threshold
            )

            for k, v in score.items():
                avr_scores[k] = v

            concept_scores[i] = avr_scores

        return concept_scores

    @staticmethod
    def topk_rank_agreement(gt_vals, pred_vals, ks, is_max=True, dynamic_k=None):
        """Get how much of top-k concept retrievals are predicted correctly."""
        num_candidates = len(gt_vals)

        gt_order = torch.argsort(gt_vals, descending=True)
        pred_order = torch.argsort(pred_vals, descending=True)

        gt_rank_pos = torch.empty_like(gt_order)
        gt_rank_pos[gt_order] = torch.arange(num_candidates, device=gt_order.device)

        pred_rank_pos = torch.empty_like(pred_order)
        pred_rank_pos[pred_order] = torch.arange(num_candidates, device=pred_order.device)

        results = {}

        for k in ks:
            k_key = k
            if k == "dynamic_k":
                if dynamic_k != 0:
                    k = dynamic_k
                else:
                    continue

            if is_max:
                gt_mask = gt_rank_pos < k
                pred_mask = pred_rank_pos < k
            else:
                k_inverted = num_candidates - k
                gt_mask = gt_rank_pos >= k_inverted
                pred_mask = pred_rank_pos >= k_inverted
            results[k_key] = (gt_mask & pred_mask).sum().item() / k * 100

        return results
