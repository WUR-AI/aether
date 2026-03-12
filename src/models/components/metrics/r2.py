from typing import Dict, override

import torch
from torch import nn
from torchmetrics.regression import R2Score

from src.models.components.metrics.base_metrics import BaseMetrics

_MODES = ("train", "val", "test")


class RSquared(BaseMetrics):
    """Epoch-level R² using torchmetrics.R2Score.

    A separate R2Score accumulator is kept per mode so that train, val, and test statistics never
    mix.  Lightning detects the returned torchmetrics Metric objects and calls .compute()/.reset()
    at epoch boundaries, giving a correct epoch-wide R² instead of an average of per-batch R²
    values.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "r2"
        # Keys are prefixed to avoid clashing with nn.Module attribute names
        # (e.g. "train" conflicts with nn.Module.train()).
        self._r2 = nn.ModuleDict({f"mode_{m}": R2Score() for m in _MODES})

    @override
    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        labels = labels if labels is not None else batch.get("target")
        mode = kwargs.get("mode", "train")

        metric = self._r2[f"mode_{mode}"]
        metric.update(pred.squeeze(-1), labels.squeeze(-1))
        return {self.name: metric}
