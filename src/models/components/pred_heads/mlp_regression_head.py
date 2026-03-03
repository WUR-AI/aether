"""MLP regression prediction head.

Renamed from: mlp_reg_pred_head.py  →  mlp_regression_head.py
Location:     src/models/components/pred_heads/mlp_regression_head.py

Changes vs original:
  - File/class name made more readable (mlp_reg_pred_head → mlp_regression_head)
  - No logic changes; class name kept as MLPRegressionPredictionHead for clarity
"""

from typing import override

import torch
from torch import nn

from src.models.components.pred_heads.base_pred_head import BasePredictionHead


class MLPRegressionPredictionHead(BasePredictionHead):
    """MLP prediction head for regression tasks (outputs a continuous value)."""

    def __init__(self, nn_layers: int = 2, hidden_dim: int = 256) -> None:
        super().__init__()
        self.nn_layers = nn_layers
        self.hidden_dim = hidden_dim

    @override
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

    @override
    def configure_nn(self) -> None:
        assert isinstance(self.input_dim, int), self.input_dim
        assert isinstance(self.output_dim, int), self.output_dim

        layers = []
        in_dim = self.input_dim

        for _ in range(self.nn_layers - 1):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = self.hidden_dim

        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
