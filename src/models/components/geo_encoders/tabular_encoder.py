from typing import Dict, override

import torch
from torch import nn

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder


class TabularEncoder(BaseGeoEncoder):
    """Tabular data encoder."""

    def __init__(
        self,
        output_dim: int,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout_prob: float = 0.0,
        geo_data_name: str = "tabular",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        self.geo_encoder: nn.Module | None = None

        self.allowed_geo_data_names = ["tabular"]
        assert (
            geo_data_name in self.allowed_geo_data_names
        ), f"geo_data_name must be one of {self.allowed_geo_data_names}, got {geo_data_name}"
        self.geo_data_name = geo_data_name

        # Normalisation statistics fitted on the training split by BaseDataModule.
        # Registered as buffers so they move to the correct device automatically.
        self.register_buffer("feat_mean", None)
        self.register_buffer("feat_std", None)

    @override
    def _setup(self, input_dim: int = None) -> list[str]:
        self.configure_nn(input_dim)
        return ["tabular_encoder"]

    def set_tabular_input_dim(self, input_dim: int) -> None:
        self.input_dim = input_dim

    def set_normalisation_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set per-feature normalisation statistics fitted on the training split."""
        self.feat_mean = mean
        self.feat_std = std

    def configure_nn(self, input_dim: int = None) -> None:
        input_dim = input_dim or self.input_dim
        assert input_dim is not None, "input_dim must be defined"

        if self.hidden_dim is None:
            self.hidden_dim = max(self.input_dim * 2, 128)

        self.geo_encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        tab_data = batch.get("eo", {}).get("tabular")

        dtype = self.dtype
        if tab_data.dtype != dtype:
            tab_data = tab_data.to(dtype)

        if self.feat_mean is not None:
            tab_data = (tab_data - self.feat_mean) / self.feat_std

        feats = self.geo_encoder(tab_data)

        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats.to(dtype)
