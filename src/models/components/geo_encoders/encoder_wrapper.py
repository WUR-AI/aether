from typing import Any, Dict, List, override

import torch
import torch.nn as nn

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.geo_encoders.identity_encoder import IdentityEncoder
from src.models.components.geo_encoders.tabular_encoder import TabularEncoder
from src.utils.errors import IllegalArgumentCombination


class EncoderWrapper(BaseGeoEncoder):
    """Wrapper class for multi-modal encoders."""

    def __init__(
        self,
        encoder_branches: List[Dict[str, Any]],
        fusion_strategy: str = "concat",
    ):
        super().__init__()

        self.output_dim = None

        self._reformat_set_branches(encoder_branches)

        # Populated by setup() — one norm per branch (Identity for TabularEncoder branches,
        # LayerNorm for all others that do not have a built-in final normalisation).
        self.branch_norms = nn.ModuleList()

        assert fusion_strategy in ["mean", "concat", "none", "gated"], ValueError(
            f'Unsupported fusion strategy "{fusion_strategy}"'
        )
        self.fusion_strategy = fusion_strategy

        # Populated by setup() for gated fusion only.
        self.gate_logits: nn.Parameter | None = None

    def _reformat_set_branches(self, encoder_branches: List[Dict[str, Any]]):
        """Reformatting to allow registering modules properly."""
        self.encoder_branches = nn.ModuleList()

        for branch in encoder_branches:
            module_dict = nn.ModuleDict({"encoder": branch["encoder"]})

            if branch.get("projector") is not None:
                module_dict["projector"] = branch["projector"]

            self.encoder_branches.append(module_dict)

    @override
    def update_configs(self, cfg):
        """Update model configurations."""
        # If adopted encoder -> it should already saved the configs
        if (
            cfg["_target_"] == "src.models.components.geo_encoders.adopt_encoder.adopt_encoder"
            and len(self.cfg_dict) != 0
        ):
            return

        for i, branch in enumerate(cfg["encoder_branches"]):
            if (
                branch["encoder"]["_target_"]
                == "src.models.components.geo_encoders.adopt_encoder.adopt_encoder"
            ):
                branch["encoder"] = self.encoder_branches[i]["encoder"].cfg_dict

        self.cfg_dict = cfg

    @override
    def _setup(self) -> List[str]:
        new_modules = []
        branch_output_dims = []

        # Configure/initialise missing/conditional parts
        for i, branch in enumerate(self.encoder_branches):
            # Setup encoder
            encoder = branch["encoder"]

            # Configure tabular encoder
            if isinstance(encoder, TabularEncoder):
                if self.tabular_dim is None:
                    raise ValueError("TabularEncoder requires tabular_dim")
                encoder.set_tabular_input_dim(self.tabular_dim)

            new_parts = encoder.setup()
            new_modules.extend(
                [f"encoder_branches.{str(i)}.encoder.{p}" for p in new_parts]
                if len(new_parts) != 0
                else []
            )

            branch_dim = encoder.output_dim

            # Configure adapter/projector if requested
            if "projector" in branch:
                if isinstance(encoder, IdentityEncoder):
                    raise IllegalArgumentCombination(
                        "Identity encoder cannot have linear projector"
                    )
                projector = branch["projector"]

                projector.set_input_dim(input_dim=branch_dim)
                new_parts = projector.setup()
                new_modules.extend(
                    [f"encoder_branches.{str(i)}.projector.{p}" for p in new_parts]
                    if len(new_parts) != 0
                    else []
                )
                branch_dim = projector.output_dim

            branch_output_dims.append(branch_dim)

        # Per-branch LayerNorm applied after encoder (+projector) output, before fusion.
        # TabularEncoder already ends with LayerNorm internally, so skip it there.
        self.branch_norms = nn.ModuleList()
        for i, (branch, dim) in enumerate(zip(self.encoder_branches, branch_output_dims)):
            if isinstance(branch["encoder"], TabularEncoder):
                self.branch_norms.append(nn.Identity())
            else:
                self.branch_norms.append(nn.LayerNorm(dim))
                new_modules.append(f"branch_norms.{i}")

        # Gated fusion: learnable scalar gate logit per branch.
        # All branches must have equal output dims; use per-branch projectors to align if needed.
        if self.fusion_strategy == "gated":
            self.gate_logits = nn.Parameter(torch.zeros(len(branch_output_dims)))
            new_modules.append("gate_logits")

        self.set_output_dim()
        return new_modules

    def set_tabular_input_dim(self, tabular_dim=None):
        """Set tabular dimension if there is tabular encoder."""
        self.tabular_dim = None

        for branch in self.encoder_branches:
            branch_out_dim = branch["encoder"]
            if isinstance(branch_out_dim, TabularEncoder):
                self.tabular_dim = tabular_dim
                return

    def set_tabular_normalisation_stats(
        self, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        """Propagate normalisation statistics to the TabularEncoder branch, if present."""
        for branch in self.encoder_branches:
            if isinstance(branch["encoder"], TabularEncoder):
                branch["encoder"].set_normalisation_stats(mean, std)
                return

    def set_output_dim(self):
        """Calculates the output dimension."""

        # Collect all output dimensions
        output_dims = []
        for branch in self.encoder_branches:
            branch_out_dim = branch["encoder"].output_dim

            if "projector" in branch:
                projector = branch["projector"]
                branch_out_dim = projector.output_dim

            output_dims.append(branch_out_dim)

        # Combine output dimensions
        if self.fusion_strategy == "concat":
            self.output_dim = sum(output_dims)
        elif self.fusion_strategy == "mean":
            if len(set(output_dims)) != 1:
                raise ValueError(
                    f"Encoder branches produces different output dimensions {output_dims} and cannot be averaged."
                )
            self.output_dim = output_dims[0]
        elif self.fusion_strategy == "gated":
            if len(set(output_dims)) != 1:
                raise ValueError(
                    f"Gated fusion requires all branches to have the same output dimension, "
                    f"got {output_dims}. Use per-branch projectors to align dimensions."
                )
            self.output_dim = output_dims[0]

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        branch_feats = []
        for i, branch in enumerate(self.encoder_branches):
            feats = branch["encoder"](batch)  # each encoder knows what modality it needs

            if "projector" in branch:
                feats = branch["projector"](feats)

            if self.branch_norms is not None:
                feats = self.branch_norms[i](feats)

            branch_feats.append(feats)

        if self.fusion_strategy == "concat":
            feats = torch.cat(branch_feats, dim=1)

        elif self.fusion_strategy == "gated":
            weights = torch.softmax(self.gate_logits, dim=0)
            stacked = torch.stack(branch_feats, dim=0)
            view_shape = [stacked.shape[0]] + [1] * (stacked.ndim - 1)
            feats = (stacked * weights.view(*view_shape)).sum(dim=0)

        else:
            feats = torch.stack(branch_feats, dim=0).mean(dim=0)

        # final (linear) project can be set in base class
        if self.extra_projector is not None:
            feats = self.extra_projector(feats)

        return feats

    @property
    def device(self):
        devices = set()
        for branch in self.encoder_branches:
            encoder = branch["encoder"]
            devices.update({p.device for p in encoder.parameters()})
            if "projector" in branch:
                projector = branch["projector"]
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
            if "projector" in branch:
                projector = branch["projector"]
                dtypes.update({p.dtype for p in projector.parameters()})

        if len(dtypes) != 1:
            raise RuntimeError("GEO encoder is on multiple devices")
        return dtypes.pop()
