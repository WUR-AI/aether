import json
import os

import pandas as pd
import pytest
import torch

from src.models.components.geo_encoders.average_encoder import AverageEncoder
from src.models.components.geo_encoders.cnn_encoder import CNNEncoder
from src.models.components.geo_encoders.geoclip import GeoClipCoordinateEncoder
from src.models.components.geo_encoders.mlp_projector import MLPProjector
from src.models.components.geo_encoders.tabular_encoder import TabularEncoder


# @pytest.mark.slow
def test_geo_encoder_generic_properties(create_butterfly_dataset):
    """This test checks that all GEO encoders implement the basic properties and methods."""
    dict_geo_encoders = {
        "geoclip_coords": GeoClipCoordinateEncoder,
        "cnn": CNNEncoder,
        "average": AverageEncoder,
        "tabular": TabularEncoder,
        "mlp_projector": MLPProjector,
    }
    ds, dm = create_butterfly_dataset
    batch = next(iter(dm.train_dataloader()))

    for geo_encoder_name, geo_encoder_class in dict_geo_encoders.items():
        if geo_encoder_class is MLPProjector:
            geo_encoder = geo_encoder_class(output_dim=64, input_dim=128)
        elif geo_encoder_class is TabularEncoder:
            geo_encoder = geo_encoder_class(output_dim=64, input_dim=128, hidden_dim=128)
        else:
            geo_encoder = geo_encoder_class()

        geo_encoder.setup()

        assert hasattr(
            geo_encoder, "geo_encoder"
        ), f"'geo_encoder' attribute missing in {geo_encoder_class.__name__}."
        assert hasattr(
            geo_encoder, "output_dim"
        ), f"'output_dim' attribute missing in {geo_encoder_class.__name__}."
        assert hasattr(
            geo_encoder, "forward"
        ), f"'forward' method missing in {geo_encoder_class.__name__}."
        assert callable(
            getattr(geo_encoder, "forward")
        ), f"'forward' is not callable in {geo_encoder_class.__name__}."

        if geo_encoder_name == "geoclip_coords":
            # TODO: try more GEO encoders when (mock) test data also includes images.
            feats = geo_encoder.forward(batch)
            assert isinstance(
                feats, torch.Tensor
            ), f"'forward' method of {geo_encoder_class.__name__} does not return a torch.Tensor."
            assert (
                feats.shape[0] == dm.batch_size_per_device
            ), f"Output batch size mismatch in {geo_encoder_class.__name__}."
