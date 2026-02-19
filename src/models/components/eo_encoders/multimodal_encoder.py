"""
Unified multimodal encoder for EO data.

Replaces the three separate encoders:
  - GeoClipCoordinateEncoder         (coords only)
  - TabularOnlyEncoder               (tabular only)
  - GeoClipCoordsTabularFusionEncoder (both)

Controlled entirely via constructor flags:
  - use_coords:   encode lat/lon with GeoClip
  - use_tabular:  encode feat_* tabular columns

tabular_dim is NOT required at construction time when use_tabular=True.
Call encoder.build_tabular_branch(tabular_dim) before the first forward
pass (done automatically by PredictiveRegressionModel.setup()).

Location: src/models/components/eo_encoders/multimodal_encoder.py
"""

from typing import Dict, override

import torch
from torch import nn

from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.eo_encoders.geoclip import GeoClipCoordinateEncoder


class MultiModalEncoder(BaseEOEncoder):
    """
    Flexible encoder that supports:
      - coords only         (use_coords=True,  use_tabular=False)
      - tabular only        (use_coords=False, use_tabular=True)
      - coords + tabular    (use_coords=True,  use_tabular=True)
    """

    def __init__(
        self,
        use_coords: bool = True,
        use_tabular: bool = False,
        tab_embed_dim: int = 64,
        output_normalization: str = "l2",
        tabular_dim: int = None,       # set here OR via build_tabular_branch()
    ) -> None:
        super().__init__()

        assert use_coords or use_tabular, (
            "At least one of use_coords or use_tabular must be True."
        )

        self.use_coords = use_coords
        self.use_tabular = use_tabular
        self.tab_embed_dim = tab_embed_dim
        self.output_normalization = output_normalization
        self._tabular_ready = False

        coords_dim = 0
        if use_coords:
            self.coords_encoder = GeoClipCoordinateEncoder(
                output_normalization=output_normalization
            )
            coords_dim = self.coords_encoder.output_dim   # 512

        self._coords_dim = coords_dim

        # Build tabular branch now only if dim is already known
        if use_tabular and tabular_dim is not None:
            self.build_tabular_branch(tabular_dim)
        elif use_tabular:
            # Will be built later by PredictiveRegressionModel.setup()
            self.tabular_proj = None
        else:
            self.output_dim = coords_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_tabular_branch(self, tabular_dim: int) -> None:
        """
        Build (or rebuild) the tabular projection layer.
        Safe to call multiple times — idempotent if dim is the same.

        Called automatically by PredictiveRegressionModel.setup()
        once the datamodule is ready.
        """
        if self._tabular_ready and hasattr(self, "_last_tabular_dim"):
            if self._last_tabular_dim == tabular_dim:
                return   # already built with correct dim

        self.tabular_proj = nn.Sequential(
            nn.LayerNorm(tabular_dim),
            nn.Linear(tabular_dim, self.tab_embed_dim),
            nn.ReLU(),
        )
        self._last_tabular_dim = tabular_dim
        self._tabular_ready = True
        self.output_dim = self._coords_dim + self.tab_embed_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []

        if self.use_coords:
            parts.append(self.coords_encoder(batch))            # (B, 512)

        if self.use_tabular:
            assert self._tabular_ready, (
                "Tabular branch not built yet. Call build_tabular_branch(tabular_dim) first, "
                "or pass tabular_dim to the constructor."
            )
            tab = batch["eo"]["tabular"].float()               # (B, tabular_dim)
            parts.append(self.tabular_proj(tab))               # (B, tab_embed_dim)

        return torch.cat(parts, dim=-1)
