"""Tests for the dynamic_gate fusion strategy in EncoderWrapper.

Dynamic gate fusion uses a small MLP to predict per-sample branch weights from the concatenated
branch outputs.  Unlike static gated fusion (one global scalar per branch), the weights vary with
each input — samples can rely more heavily on one modality depending on the data.  The MLP takes a
vector of shape [n_branches * dim] and outputs [n_branches] logits, which are passed through
softmax to form a convex combination of branch embeddings. All branches must share the same output
dim.
"""

from typing import Dict, List, override

import pytest
import torch

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.geo_encoders.encoder_wrapper import EncoderWrapper
from src.models.components.geo_encoders.tabular_encoder import TabularEncoder

# ---------------------------------------------------------------------------
# Minimal stub encoder for testing — outputs a fixed-size embedding.
# ---------------------------------------------------------------------------


class _StubEncoder(BaseGeoEncoder):
    """Stub encoder that outputs a fixed dimension via a linear layer."""

    def __init__(self, output_dim: int, key: str = "coords") -> None:
        super().__init__()
        self._out_dim = output_dim
        self._key = key
        self._linear = torch.nn.Linear(output_dim, output_dim)

    @override
    def _setup(self) -> List[str]:
        self.output_dim = self._out_dim
        return ["_linear"]

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["eo"][self._key]
        return self._linear(x[:, : self._out_dim])


def _make_wrapper(branch_dims=(32, 32), keys=None) -> EncoderWrapper:
    if keys is None:
        keys = [f"b{i}" for i in range(len(branch_dims))]
    branches = [{"encoder": _StubEncoder(dim, key=key)} for dim, key in zip(branch_dims, keys)]
    wrapper = EncoderWrapper(encoder_branches=branches, fusion_strategy="dynamic_gate")
    wrapper.set_tabular_input_dim(None)
    wrapper.setup()
    return wrapper


def _make_batch(branch_dims=(32, 32), batch_size=4, keys=None) -> Dict:
    if keys is None:
        keys = [f"b{i}" for i in range(len(branch_dims))]
    eo = {key: torch.randn(batch_size, dim) for key, dim in zip(keys, branch_dims)}
    return {"eo": eo}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_dim():
    # Output dim equals the shared branch dim (not summed, as with concat).
    wrapper = _make_wrapper(branch_dims=(32, 32))
    assert wrapper.output_dim == 32


def test_forward_shape():
    wrapper = _make_wrapper(branch_dims=(32, 32))
    batch = _make_batch(branch_dims=(32, 32))
    out = wrapper(batch)
    assert out.shape == (4, 32)


def test_mlp_is_module():
    # dynamic_gate_mlp must be a proper nn.Module, not a plain callable.
    wrapper = _make_wrapper()
    assert wrapper.dynamic_gate_mlp is not None
    assert isinstance(wrapper.dynamic_gate_mlp, torch.nn.Module)


def test_mlp_in_parameters():
    # MLP parameters must be reachable via wrapper.parameters() so they are
    # picked up by the optimiser and saved in checkpoints.
    wrapper = _make_wrapper()
    wrapper_param_ids = {id(p) for p in wrapper.parameters()}
    mlp_param_ids = {id(p) for p in wrapper.dynamic_gate_mlp.parameters()}
    assert mlp_param_ids.issubset(wrapper_param_ids)


def test_mlp_parameters_require_grad():
    wrapper = _make_wrapper()
    assert all(p.requires_grad for p in wrapper.dynamic_gate_mlp.parameters())


def test_weights_sum_to_one_per_sample():
    """Softmax over branches must produce weights that sum to 1.0 for every sample."""
    wrapper = _make_wrapper(branch_dims=(32, 32, 32))
    batch = _make_batch(branch_dims=(32, 32, 32), batch_size=8)

    # Replicate the gating arithmetic from forward() to inspect the raw weights.
    branch_feats = []
    with torch.no_grad():
        for i, branch in enumerate(wrapper.encoder_branches):
            feats = branch["encoder"](batch)
            feats = wrapper.branch_norms[i](feats)
            branch_feats.append(feats)

        stacked = torch.stack(branch_feats, dim=1)
        gate_input = stacked.flatten(start_dim=1)
        weights = torch.softmax(wrapper.dynamic_gate_mlp(gate_input), dim=1)

    row_sums = weights.sum(dim=1)
    assert row_sums.allclose(torch.ones(8), atol=1e-6)


def test_weights_vary_per_sample():
    """Weights must differ across samples — this is the key property that distinguishes
    dynamic_gate from static gated fusion, where weights are the same for every sample."""
    wrapper = _make_wrapper(branch_dims=(32, 32))
    batch = _make_batch(branch_dims=(32, 32), batch_size=8)

    with torch.no_grad():
        branch_feats = []
        for i, branch in enumerate(wrapper.encoder_branches):
            feats = branch["encoder"](batch)
            feats = wrapper.branch_norms[i](feats)
            branch_feats.append(feats)

        stacked = torch.stack(branch_feats, dim=1)
        gate_input = stacked.flatten(start_dim=1)
        weights = torch.softmax(wrapper.dynamic_gate_mlp(gate_input), dim=1)

    assert not weights[0].allclose(weights[1], atol=1e-6)


def test_single_branch():
    """With one branch, output must equal the branch embedding (weight collapses to 1.0)."""
    wrapper = _make_wrapper(branch_dims=(32,), keys=["b0"])
    batch = _make_batch(branch_dims=(32,), keys=["b0"])

    out = wrapper(batch)

    encoder = wrapper.encoder_branches[0]["encoder"]
    expected = wrapper.branch_norms[0](encoder(batch))

    assert out.allclose(expected, atol=1e-6)


def test_mismatched_dims_raises():
    """Branches with different output dims must raise at setup time.

    Use per-branch projectors to align dims before fusion.
    """
    branches = [{"encoder": _StubEncoder(16, "b0")}, {"encoder": _StubEncoder(32, "b1")}]
    wrapper = EncoderWrapper(encoder_branches=branches, fusion_strategy="dynamic_gate")
    wrapper.set_tabular_input_dim(None)
    with pytest.raises(ValueError, match="same output dimension"):
        wrapper.setup()


def test_with_tabular_encoder():
    """TabularEncoder as one branch; set_tabular_input_dim must flow through.

    Both branches projected to the same output_dim (32) via per-branch projectors.
    """
    from src.models.components.geo_encoders.mlp_projector import MLPProjector

    tabular_dim = 23
    tab_enc = TabularEncoder(output_dim=32)
    stub_enc = _StubEncoder(output_dim=16, key="coords")

    wrapper = EncoderWrapper(
        encoder_branches=[
            {"encoder": tab_enc},
            {"encoder": stub_enc, "projector": MLPProjector(output_dim=32)},
        ],
        fusion_strategy="dynamic_gate",
    )
    wrapper.set_tabular_input_dim(tabular_dim)
    wrapper.setup()

    batch = {
        "eo": {
            "tabular": torch.randn(4, tabular_dim),
            "coords": torch.randn(4, 32),  # stub slices first 16
        }
    }
    out = wrapper(batch)
    assert out.shape == (4, 32)


def test_three_branch_dual_tessera():
    """Three-branch EncoderWrapper matching the dual-tessera model config.

    Branches: tessera (year Y), tessera_prev (year Y-1), tabular — all projected to dim=256.
    Verifies: output shape [batch, 256], gate weights sum to 1.0 per sample, all weights > 0.
    """
    from src.models.components.geo_encoders.mlp_projector import MLPProjector

    tabular_dim = 20
    stub_dim = 128  # TESSERA embedding width

    wrapper = EncoderWrapper(
        encoder_branches=[
            {
                "encoder": _StubEncoder(stub_dim, key="tessera"),
                "projector": MLPProjector(nn_layers=1, output_dim=256),
            },
            {
                "encoder": _StubEncoder(stub_dim, key="tessera_prev"),
                "projector": MLPProjector(nn_layers=1, output_dim=256),
            },
            {"encoder": TabularEncoder(output_dim=256, dropout_prob=0.2)},
        ],
        fusion_strategy="dynamic_gate",
    )
    wrapper.set_tabular_input_dim(tabular_dim)
    wrapper.setup()

    batch_size = 4
    batch = {
        "eo": {
            "tessera": torch.randn(batch_size, stub_dim),
            "tessera_prev": torch.randn(batch_size, stub_dim),
            "tabular": torch.randn(batch_size, tabular_dim),
        }
    }

    out = wrapper(batch)
    assert out.shape == (batch_size, 256)

    # Replicate the gating arithmetic to inspect branch weights.
    with torch.no_grad():
        branch_feats = []
        for i, branch in enumerate(wrapper.encoder_branches):
            feats = branch["encoder"](batch)
            if "projector" in branch:
                feats = branch["projector"](feats)
            feats = wrapper.branch_norms[i](feats)
            branch_feats.append(feats)

        stacked = torch.stack(branch_feats, dim=1)
        gate_input = stacked.flatten(start_dim=1)
        weights = torch.softmax(wrapper.dynamic_gate_mlp(gate_input), dim=1)

    # Weights must sum to 1.0 per sample across all 3 branches.
    assert weights.shape == (batch_size, 3)
    assert weights.sum(dim=1).allclose(torch.ones(batch_size), atol=1e-6)
    # Softmax is strictly positive — all branches have non-zero weight.
    assert (weights > 0).all()


def test_existing_gated_unaffected():
    """Ensure static gated strategy still works after the dynamic_gate additions."""
    wrapper = EncoderWrapper(
        encoder_branches=[
            {"encoder": _StubEncoder(32, "b0")},
            {"encoder": _StubEncoder(32, "b1")},
        ],
        fusion_strategy="gated",
    )
    wrapper.set_tabular_input_dim(None)
    wrapper.setup()
    batch = _make_batch(branch_dims=(32, 32))
    out = wrapper(batch)
    assert out.shape == (4, 32)
