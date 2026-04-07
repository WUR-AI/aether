from typing import Dict, override

import torch

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class RRMSELoss(BaseLossFn):
    """Relative Root Mean Squared Error (RRMSE).

    RRMSE = RMSE / mean(labels)

    Normalises RMSE by the mean absolute value of the target, giving a unit-free percentage error.
    This makes results comparable across crops and regions with different absolute yield scales
    (e.g. t/ha ranges differ significantly between maize in Zambia and rice in Rwanda).

    Returns a fraction (e.g. 0.15 = 15 % error). Multiply by 100 for percentage when reporting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.MSELoss()
        self.name = "rrmse_loss"

    @override
    def forward(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor | None = None,
        batch: Dict[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:

        labels = labels if labels is not None else batch.get("target")
        rmse = torch.sqrt(self.criterion(pred, labels))
        mean_abs = torch.mean(torch.abs(labels))
        loss = rmse / (mean_abs + 1e-8)

        if "return_label" in kwargs:
            return {self.name: loss}
        else:
            return loss
