from abc import ABC, abstractmethod
from typing import Dict, List, final

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

        # Modules
        self.output_dim: int | None = None
        self.setup_flag: bool = False
        self.cfg_dict: Dict = {}
        self.input_dim: int | None = None
        self.cfg_dict: Dict = {}

    @final
    def set_input_dim(self, input_dim: int) -> None:
        self.input_dim = input_dim

    @final
    def setup(self, verbose=1) -> List[str]:
        """Configures modules.

        Gets called in model.setup() method. Returns names of any new module configured to be added
        to the trainable modules list.
        """
        if self.setup_flag:
            print(f"Module {self.__str__()} is already set up.")
            return []
        else:
            trainable_modules = self._setup()
            if verbose > 0:
                print(f"Model set up with {self.__str__()}")
            self.setup_flag = True
            return trainable_modules

    @abstractmethod
    def _setup(self) -> List[str]:
        """Configures modules and returns newly initialised, trainable module names."""
        pass

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device | None:
        devices = {p.device for p in self.parameters()}
        if len(devices) > 1:
            raise RuntimeError("Encoder is on multiple devices")
        elif len(devices) == 0:
            return None
        return devices.pop()

    @property
    def dtype(self) -> torch.dtype | None:
        dtypes = {p.dtype for p in self.parameters()}
        if len(dtypes) > 1:
            raise RuntimeError("Encoder has multiple dtypes")
        elif len(dtypes) == 0:
            return None
        return dtypes.pop()
