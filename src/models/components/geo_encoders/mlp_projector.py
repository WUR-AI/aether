from typing import List, override

import torch
from torch import nn

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder


class MLPProjector(BaseGeoEncoder):
    def __init__(
        self,
        output_dim: int,
        input_dim: int | None = None,
        nn_layers: int = 2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.nn_layers = nn_layers
        self.hidden_dim = hidden_dim

        # Placeholder
        self.net: nn.Module | None = None

    @override
    def setup(self) -> List[str]:
        self.configure_nn()
        return ["net"]

    def set_input_dim(self, input_dim: int) -> None:
        self.input_dim = input_dim

    def configure_nn(self) -> None:
        """Configure the MLP network."""
        assert self.input_dim is not None, "input_dim must be defined"
        assert self.output_dim is not None, "output_dim must be defined"
        layers = []
        input_dim = self.input_dim

        for i in range(self.nn_layers - 1):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = self.hidden_dim

        layers.append(nn.Linear(input_dim, self.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
