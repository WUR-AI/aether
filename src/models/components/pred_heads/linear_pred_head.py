from typing import List, override

import torch
from torch import nn

from src.models.components.pred_heads.base_pred_head import BasePredictionHead


class LinearPredictionHead(BasePredictionHead):
    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
    ) -> None:
        """Linear prediction head for classification.

        :param input_dim: the size of input dimension
        :param output_dim: the size of output dimension
        """
        super().__init__()
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
        self.net = nn.Linear(self.input_dim, self.output_dim)
        return


if __name__ == "__main__":
    _ = LinearPredictionHead()
