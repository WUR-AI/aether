from typing import Dict, override

import torch
from torch import nn
from torchmetrics.regression import MeanAbsolutePercentageError

from src.models.components.metrics.base_metrics import BaseMetrics

_MODES = ("train", "val", "test")


class MAPE(BaseMetrics):
    """Epoch-level MAPE using torchmetrics.MeanAbsolutePercentageError.

    A separate MeanAbsolutePercentageError accumulator is kept per mode so that train, val, and
    test statistics never mix.  Lightning detects the returned torchmetrics Metric objects and
    calls .compute()/.reset() at epoch boundaries, giving a correct epoch-wide MAPE instead of an
    average of per-batch MAPE values.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "mape"
        # Keys are prefixed to avoid clashing with nn.Module attribute names
        # (e.g. "train" conflicts with nn.Module.train()).
        self._mape = nn.ModuleDict({f"mode_{m}": MeanAbsolutePercentageError() for m in _MODES})

    @override
    def forward(
        self,
        pred: torch.Tensor,
        mode: str | None = None,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if mode not in _MODES:
            raise ValueError(f"MAPE.forward: mode must be one of {_MODES}, got '{mode}'")
        if labels is None:
            labels = batch.get("target") if batch is not None else None
        if labels is None:
            raise ValueError(
                "MAPE.forward: labels must be provided via `labels` or `batch['target']`"
            )

        metric = self._mape[f"mode_{mode}"]
        metric.update(pred.squeeze(-1), labels.squeeze(-1))
        return {f"{mode}_{self.name}": metric}
