from abc import ABC, abstractmethod
from typing import List, final

import torch
from torch import nn


class BasePredictionHead(nn.Module, ABC):
    def __init__(self) -> None:
        """Base prediction head interface class."""
        super().__init__()

        # Modules
        self.net: nn.Module | None = None

        self.input_dim: int | None = None
        self.output_dim: int | None = None
        self.setup_flag: bool = False
        self.cfg_dict = {}

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
        assert isinstance(input_dim, int), TypeError(
            "Input dimension must be specified as integer"
        )
        assert isinstance(output_dim, int), TypeError(
            "Output dimension must be specified as integer"
        )
        self.input_dim = input_dim
        self.output_dim = output_dim

    @final
    def setup(self) -> List[str]:
        """Configures modules.

        Gets called in model.setup() method. Returns names of any new module configured to be added
        to the trainable modules list.
        """
        if self.setup_flag:
            print(f"Module {self.__str__()} is already set up.")
            return []
        else:
            self._setup()
            print(f"Model set up with {self.__str__()}")
            self.setup_flag = True
            return ["prediction_head"]

    @abstractmethod
    def _setup(self) -> None:
        """Configures specific prediction head."""
        pass
