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

    def __init__(
        self,
        nn_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> None:
        """MLP prediction head for regression tasks.

        :param nn_layers: number of layers in MLP
        :param hidden_dim: the size of hidden dimensions
        :param dropout: the dropout rate
        :param input_dim: the size of input dimension
        :param output_dim: the size of output dimension
        """
        super().__init__()
        self.nn_layers = nn_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        if input_dim and output_dim:
            self.set_dim(input_dim, output_dim)

    @override
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass through the prediction head."""
        return self.net(feats)

    @override
    def _setup(self) -> None:
        """Configures specific prediction head."""
        assert isinstance(self.input_dim, int), self.input_dim
        assert isinstance(self.output_dim, int), self.output_dim

        layers = []
        in_dim = self.input_dim

        for _ in range(self.nn_layers - 1):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim

        layers.append(nn.Linear(in_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
        return
