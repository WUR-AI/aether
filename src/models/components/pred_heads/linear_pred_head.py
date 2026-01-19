from typing import override

import torch
from torch import nn

from src.models.components.pred_heads.base_pred_head import BasePredictionHead


class LinearPredictionHead(BasePredictionHead):
    def __init__(self):
        super().__init__()

    @override
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feats))

    @override
    def configure_nn(self) -> None:
        assert type(self.input_dim) is int, self.input_dim
        assert type(self.output_dim) is int, self.output_dim
        self.net = nn.Linear(self.input_dim, self.output_dim)
        return


if __name__ == "__main__":
    _ = LinearPredictionHead()
