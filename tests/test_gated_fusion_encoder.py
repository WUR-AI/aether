"""Tests for the gated fusion strategy in EncoderWrapper."""

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
        self._linear = torch.nn.Linear(output_dim, output_dim)  # gives it parameters

    @override
    def setup(self) -> List[str]:
        self.output_dim = self._out_dim
        return ["_linear"]

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["eo"][self._key]
        return self._linear(x[:, : self._out_dim])


def _make_wrapper(branch_dims=(32, 32), keys=None) -> EncoderWrapper:
    if keys is None:
        keys = [f"b{i}" for i in range(len(branch_dims))]
    branches = [
        {"encoder": _StubEncoder(dim, key=key)}
        for dim, key in zip(branch_dims, keys)
    ]
    wrapper = EncoderWrapper(encoder_branches=branches, fusion_strategy="gated")
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
    wrapper = _make_wrapper(branch_dims=(32, 32))
    assert wrapper.output_dim == 32


def test_forward_shape():
    wrapper = _make_wrapper(branch_dims=(32, 32))
    batch = _make_batch(branch_dims=(32, 32))
    out = wrapper(batch)
    assert out.shape == (4, 32)


def test_gate_logits_learnable():
    wrapper = _make_wrapper()
    assert wrapper.gate_logits is not None
    assert isinstance(wrapper.gate_logits, torch.nn.Parameter)
    assert wrapper.gate_logits.requires_grad


def test_gate_logits_in_parameters():
    wrapper = _make_wrapper()
    param_ids = {id(p) for p in wrapper.parameters()}
    assert id(wrapper.gate_logits) in param_ids


def test_uniform_init_weights():
    wrapper = _make_wrapper(branch_dims=(32, 32))
    weights = torch.softmax(wrapper.gate_logits, dim=0)
    expected = 1.0 / len(wrapper.encoder_branches)
    assert weights.allclose(torch.full_like(weights, expected))


def test_gate_weights_sum_to_one():
    wrapper = _make_wrapper(branch_dims=(32, 32, 32))
    with torch.no_grad():
        wrapper.gate_logits.copy_(torch.tensor([1.0, -0.5, 2.0]))
    weights = torch.softmax(wrapper.gate_logits, dim=0)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_single_branch():
    """With one branch, the output must equal the branch embedding (weight is 1.0)."""
    wrapper = _make_wrapper(branch_dims=(32,), keys=["b0"])
    batch = _make_batch(branch_dims=(32,), keys=["b0"])

    out = wrapper(batch)

    encoder = wrapper.encoder_branches[0]["encoder"]
    expected = wrapper.branch_norms[0](encoder(batch))

    assert out.allclose(expected, atol=1e-6)


def test_mismatched_dims_raises():
    """Branches with different output dims must raise at setup time."""
    branches = [{"encoder": _StubEncoder(16, "b0")}, {"encoder": _StubEncoder(32, "b1")}]
    wrapper = EncoderWrapper(encoder_branches=branches, fusion_strategy="gated")
    wrapper.set_tabular_input_dim(None)
    with pytest.raises(ValueError, match="same output dimension"):
        wrapper.setup()


def test_with_tabular_encoder():
    """TabularEncoder as one branch; set_tabular_input_dim must flow through.
    Both branches projected to the same output_dim (32) via per-branch projectors."""
    from src.models.components.geo_encoders.mlp_projector import MLPProjector

    tabular_dim = 23
    tab_enc = TabularEncoder(output_dim=32)
    stub_enc = _StubEncoder(output_dim=16, key="coords")

    wrapper = EncoderWrapper(
        encoder_branches=[
            {"encoder": tab_enc},
            {"encoder": stub_enc, "projector": MLPProjector(output_dim=32)},
        ],
        fusion_strategy="gated",
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


def test_existing_concat_unaffected():
    """Ensure concat strategy still works after the gated additions."""
    branches = [{"encoder": _StubEncoder(16, "b0")}, {"encoder": _StubEncoder(32, "b1")}]
    wrapper = EncoderWrapper(encoder_branches=branches, fusion_strategy="concat")
    wrapper.set_tabular_input_dim(None)
    wrapper.setup()
    batch = _make_batch(branch_dims=(16, 32))
    out = wrapper(batch)
    assert out.shape == (4, 48)  # 16 + 32