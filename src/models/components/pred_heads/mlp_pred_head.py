from typing import override

import torch
from torch import nn

from src.models.components.pred_heads.base_pred_head import BasePredictionHead


class MLPPredictionHead(BasePredictionHead):
    def __init__(self, nn_layers: int = 2, hidden_dim: int = 256) -> None:
        super().__init__()
        self.nn_layers = nn_layers
        self.hidden_dim = hidden_dim

    @override
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feats))

    @override
    def configure_nn(self) -> None:
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
