from typing import Dict, List

import torch
from torch import nn

from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.metrics.base_metrics import BaseMetrics


class MetricsWrapper(nn.Module):
    def __init__(self, metrics: List[BaseMetrics | BaseLossFn]) -> None:
        super().__init__()
        self.metrics = nn.ModuleList(metrics)

    def forward(self, mode="train", **kwargs) -> Dict[str, torch.float]:
        """Calculates all metrics and adds all the results into one dictionary for logging."""
        compiled_dict = {}

        for metric in self.metrics:
            metric_results = metric(mode=mode, return_label=True, **kwargs)
            for k, v in metric_results.items():
                compiled_dict[k] = v

        return compiled_dict
