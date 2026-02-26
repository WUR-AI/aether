from typing import Dict, override

import torch

from src.models.components.metrics.base_metrics import BaseMetrics


class RSquared(BaseMetrics):
    def __init__(self) -> None:
        super().__init__()
        self.name = "r2"

    @override
    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        labels = labels if labels is not None else batch.get("target")

        ss_res = torch.sum((labels - pred) ** 2)
        ss_tot = torch.sum((labels - torch.mean(labels)) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        return {self.name: r2}
