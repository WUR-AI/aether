import json
import os

import pandas as pd
import pytest
import torch

from src.models.components.eo_encoders.average_encoder import AverageEncoder
from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.eo_encoders.cnn_encoder import CNNEncoder
from src.models.components.eo_encoders.geoclip import GeoClipCoordinateEncoder


# @pytest.mark.slow
def test_eo_encoder_generic_properties(create_butterfly_dataset):
    """This test checks that all EO encoders implement the basic properties and methods."""
    dict_eo_encoders = {
        "geoclip_coords": GeoClipCoordinateEncoder,
        "cnn": CNNEncoder,
        "average": AverageEncoder,
    }
    ds, dm = create_butterfly_dataset
    batch = next(iter(dm.train_dataloader()))

    for eo_encoder_name, eo_encoder_class in dict_eo_encoders.items():
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

        if eo_encoder_name == "geoclip_coords":
            # TODO: try more EO encoders when (mock) test data also includes images.
            feats = eo_encoder.forward(batch)
            assert isinstance(
                feats, torch.Tensor
            ), f"'forward' method of {eo_encoder_class.__name__} does not return a torch.Tensor."
            assert (
                feats.shape[0] == dm.batch_size_per_device
            ), f"Output batch size mismatch in {eo_encoder_class.__name__}."
