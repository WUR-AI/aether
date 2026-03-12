from abc import ABC, abstractmethod
from typing import final

import torch
from torch import nn


class BasePredictionHead(nn.Module, ABC):
    def __init__(self) -> None:
        """Base prediction head interface class."""
        super().__init__()
        self.net: nn.Module | None = None
        self.input_dim: int | None = None
        self.output_dim: int | None = None

    @abstractmethod
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass through the prediction head."""
        pass

    @final
    def set_dim(self, input_dim: int, output_dim: int) -> None:
        """Set dimensions for the prediction head configuration.

        :param input_dim: input dimension
        :param output_dim: output dimension
        """
        assert isinstance(self.input_dim, int), TypeError(
            "Input dimension must be specified as integer"
        )
        assert isinstance(self.output_dim, int), TypeError(
            "Output dimension must be specified as integer"
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def setup(self) -> None:
        """Configures networks, data-dependent parts.

        Gets called in model.setup() method. Returns names of any new module configured to be added
        to the trainable modules list.
        """
        pass
