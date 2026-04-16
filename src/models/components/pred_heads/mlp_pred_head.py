from typing import override

import torch
from torch import nn

from src.models.components.pred_heads.base_pred_head import BasePredictionHead


class MLPPredictionHead(BasePredictionHead):
    def __init__(
        self,
        nn_layers: int = 2,
        hidden_dim: int = 256,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> None:
        """MLP prediction head for classification.

        :param nn_layers: number of layers in MLP
        :param hidden_dim: the size of hidden dimensions
        :param input_dim: the size of input dimension
        :param output_dim: the size of output dimension
        """
        super().__init__()
        self.nn_layers = nn_layers
        self.hidden_dim = hidden_dim

        if input_dim and output_dim:
            self.set_dim(input_dim, output_dim)

    @override
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass through the prediction head."""
        return torch.sigmoid(self.net(feats))

    @override
    def _setup(self) -> None:
        """Configures specific prediction head."""
        assert type(self.input_dim) is int, self.input_dim
        assert type(self.output_dim) is int, self.output_dim
        layers = []
        input_dim = self.input_dim
        for i in range(self.nn_layers - 1):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = self.hidden_dim
        layers.append(nn.Linear(input_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
        return


if __name__ == "__main__":
    _ = MLPPredictionHead()
