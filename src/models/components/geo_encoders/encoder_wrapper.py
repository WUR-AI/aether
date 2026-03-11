from typing import Any, Dict, List, override

import torch

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.geo_encoders.tabular_encoder import TabularEncoder


class EncoderWrapper(BaseGeoEncoder):
    """Wrapper class for multi-modal encoders."""

    def __init__(
        self,
        encoder_branches: List[Dict[str, Any]],
        fusion_strategy: str,
    ):
        super().__init__()

        self.encoder_branches = encoder_branches
        assert fusion_strategy in ["mean", "concat", "none"], ValueError(
            f'Unsupported fusion strategy "{fusion_strategy}"'
        )
        self.fusion_strategy = fusion_strategy
        self.output_dim = None

        # Configure/initialise missing/conditional parts
        for branch in self.encoder_branches:
            intermediate_dim = branch.get("encoder").output_dim
            projector = branch.get("projector", None)
            if projector is not None:
                projector.set_input_dim(input_dim=intermediate_dim)
                projector.configure_nn()

    def configure_nn(self, tabular_dim: int) -> None:
        output_dims = []
        new_parts = set()
        for branch in self.encoder_branches:
            if isinstance(branch["encoder"], TabularEncoder):
                branch["encoder"].configure_nn(tabular_dim)
                new_parts.add("ta")
            if branch.get("projector"):
                output_dims.append(branch["projector"].output_dim)
            else:
                output_dims.append(branch["encoder"].output_dim)

        if self.fusion_strategy == "concat":
            self.output_dim = sum(output_dims)
        elif self.fusion_strategy == "mean":
            assert set(output_dims) == 1, ValueError(
                f"Encoder branches produces different output dimensions {output_dims} and cannot be averaged."
            )
            self.output_dim = output_dims[0]

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        branch_feats = []
        for branch in self.encoder_branches:
            feats = branch["encoder"].forward(batch)  # each encoder knows what modality it needs

            if branch.get("projector", None):
                feats = branch["projector"].forward(feats)

            branch_feats.append(feats)

        if self.fusion_strategy == "concat":
            return torch.cat(branch_feats, dim=1)
        return torch.mean(branch_feats, dim=1)

    @property
    def device(self):
        devices = set()
        for branch in self.encoder_branches:
            encoder = branch["encoder"]
            devices.update({p.device for p in encoder.parameters()})
            projector = branch.get("projector")
            if projector is not None:
                devices.update({p.device for p in projector.parameters()})

        if len(devices) != 1:
            raise RuntimeError("GEO encoder is on multiple devices")
        return devices.pop()

    @property
    def dtype(self) -> torch.dtype:
        dtypes = set()
        for branch in self.encoder_branches:
            encoder = branch["encoder"]
            dtypes.update({p.dtype for p in encoder.parameters()})
            projector = branch.get("projector")
            if projector is not None:
                dtypes.update({p.dtype for p in projector.parameters()})

        if len(dtypes) != 1:
            raise RuntimeError("GEO encoder is on multiple devices")
        return dtypes.pop()
