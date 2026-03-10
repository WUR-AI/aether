"""Unified multimodal encoder for EO data.

Controlled entirely via constructor flags:
  - use_coords:       activate the spatial/geo encoder branch
  - use_tabular:      encode feat_* tabular columns
  - geo_encoder_cfg:  pluggable geo encoder (any BaseGeoEncoder subclass);
                      when None and use_coords=True, defaults to GeoClipCoordinateEncoder
"""

from typing import Dict, override

import torch
from torch import nn

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.geo_encoders.geoclip import GeoClipCoordinateEncoder


class MultiModalEncoder(BaseGeoEncoder):
    """
    Modes (controlled by use_coords / use_tabular flags):

      - geo only          (use_coords=True,  use_tabular=False)
      - tabular only      (use_coords=False, use_tabular=True)
      - geo + tabular     (use_coords=True,  use_tabular=True)

    The geo encoder defaults to GeoClipCoordinateEncoder but can be replaced
    with any BaseGeoEncoder via the geo_encoder_cfg parameter.  Hydra
    instantiates geo_encoder_cfg before passing it here, so it arrives as a
    ready-to-use nn.Module (e.g. AverageEncoder for TESSERA tiles).

    Example config (TESSERA + tabular fusion):
        geo_encoder:
          _target_: ...MultiModalEncoder
          use_coords: true
          use_tabular: true
          geo_encoder_cfg:
            _target_: ...AverageEncoder
            geo_data_name: tessera
    """

    def __init__(
        self,
        use_coords: bool = True,
        use_tabular: bool = False,
        tab_embed_dim: int = 64,
        tabular_dropout: float = 0.0,
        tabular_dim: int = None,
        geo_encoder_cfg: BaseGeoEncoder | None = None,
    ) -> None:
        super().__init__()

        assert use_coords or use_tabular, "At least one of use_coords or use_tabular must be True."

        self.use_coords = use_coords
        self.use_tabular = use_tabular
        self.tab_embed_dim = tab_embed_dim
        self.tabular_dropout = tabular_dropout
        self._tabular_ready = False
        self.fusion_norm = None  # set in build_tabular_branch when both branches active

        coords_dim = 0
        if use_coords:
            if geo_encoder_cfg is not None:
                self.coords_encoder = geo_encoder_cfg
            else:
                self.coords_encoder = GeoClipCoordinateEncoder()
            coords_dim = self.coords_encoder.output_dim

        self._coords_dim = coords_dim

        # Built only if dim is already known
        if use_tabular and tabular_dim is not None:
            self.build_tabular_branch(tabular_dim)
        elif use_tabular:
            self.tabular_proj = None
        else:
            self.output_dim = coords_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_tabular_branch(self, tabular_dim: int) -> None:
        """Build (or rebuild) the tabular projection MLP.

        Architecture: LayerNorm → Linear(in, h) → ReLU → Dropout →
                      Linear(h, h//2) → ReLU → Dropout → Linear(h//2, out)
        where h = max(tab_embed_dim * 2, 128).
        """
        if self._tabular_ready and hasattr(self, "_last_tabular_dim"):
            if self._last_tabular_dim == tabular_dim:
                return  # already built with correct dim

        hidden = max(self.tab_embed_dim * 2, 128)
        drop = self.tabular_dropout
        self.tabular_proj = nn.Sequential(
            nn.LayerNorm(tabular_dim),
            nn.Linear(tabular_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, self.tab_embed_dim),
        )
        self._last_tabular_dim = tabular_dim
        self._tabular_ready = True
        self.output_dim = self._coords_dim + self.tab_embed_dim

        # Normalise the fused representation when both branches are active.
        # The geo encoder output and the tabular projection may have different
        # scales, so a LayerNorm stabilises training after concat.
        if self.use_coords:
            self.fusion_norm = nn.LayerNorm(self.output_dim)
        else:
            self.fusion_norm = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []

        if self.use_coords:
            parts.append(self.coords_encoder(batch))  # (B, coords_encoder.output_dim)

        if self.use_tabular:
            assert self._tabular_ready, (
                "Tabular branch not built yet. Call build_tabular_branch(tabular_dim) first, "
                "or pass tabular_dim to the constructor."
            )
            tab = batch["eo"]["tabular"].float()  # (B, tabular_dim)
            parts.append(self.tabular_proj(tab))  # (B, tab_embed_dim)

        fused = torch.cat(parts, dim=-1)
        if self.fusion_norm is not None:
            fused = self.fusion_norm(fused)
        return fused
