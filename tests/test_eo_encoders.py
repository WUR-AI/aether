import json
import os

import pandas as pd
import pytest
import torch

from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.eo_encoders.geoclip import GeoClipCoordinateEncoder


# @pytest.mark.slow
def test_eo_encoder_generic_properties(create_butterfly_dataset):
    """This test checks that all EO encoders implement the basic properties and methods."""
    list_eo_encoders = [GeoClipCoordinateEncoder]
    ds, dm = create_butterfly_dataset
    batch = next(iter(dm.train_dataloader()))

    for eo_encoder_class in list_eo_encoders:
        eo_encoder = eo_encoder_class()

        assert hasattr(
            eo_encoder, "eo_encoder"
        ), f"'eo_encoder' attribute missing in {eo_encoder_class.__name__}."
        assert hasattr(
            eo_encoder, "output_dim"
        ), f"'output_dim' attribute missing in {eo_encoder_class.__name__}."
        assert hasattr(
            eo_encoder, "forward"
        ), f"'forward' method missing in {eo_encoder_class.__name__}."
        assert callable(
            getattr(eo_encoder, "forward")
        ), f"'forward' is not callable in {eo_encoder_class.__name__}."
        feats = eo_encoder.forward(batch)
        assert isinstance(
            feats, torch.Tensor
        ), f"'forward' method of {eo_encoder_class.__name__} does not return a torch.Tensor."
        assert (
            feats.shape[0] == dm.batch_size_per_device
        ), f"Output batch size mismatch in {eo_encoder_class.__name__}."
