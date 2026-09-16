import torch
import torch.nn as nn


class LinearAdapter(nn.Module):

    def __init__(self, input_dim: int = 128, output_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def set_input_dim(self, input_dim: int) -> None:
        """Dynamically updates input dimensions if required by the model setup."""
        if self.input_dim != input_dim:
            self.input_dim = input_dim
            self.proj = nn.Linear(input_dim, self.output_dim)

    def setup(self, stage: str = None) -> None:
        """Lifecycle hook called during InferenceModel._setup()."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
