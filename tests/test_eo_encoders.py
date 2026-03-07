import json
import os

import pandas as pd
import pytest
import torch

from src.models.components.geo_encoders.average_encoder import AverageEncoder
from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.geo_encoders.cnn_encoder import CNNEncoder
from src.models.components.geo_encoders.geoclip import GeoClipCoordinateEncoder
from src.models.components.geo_encoders.multimodal_encoder import MultiModalEncoder


# @pytest.mark.slow
def test_geo_encoder_generic_properties(create_butterfly_dataset):
    """This test checks that all EO encoders implement the basic properties and methods."""
    dict_geo_encoders = {
        "geoclip_coords": GeoClipCoordinateEncoder,
        "cnn": CNNEncoder,
        "average": AverageEncoder,
        "multimodal_coords": MultiModalEncoder,
    }
    ds, dm = create_butterfly_dataset
    batch = next(iter(dm.train_dataloader()))

    for geo_encoder_name, geo_encoder_class in dict_geo_encoders.items():
        geo_encoder = geo_encoder_class()

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
            # TODO: try more EO encoders when (mock) test data also includes images.
            feats = geo_encoder.forward(batch)
            assert isinstance(
                feats, torch.Tensor
            ), f"'forward' method of {geo_encoder_class.__name__} does not return a torch.Tensor."
            assert (
                feats.shape[0] == dm.batch_size_per_device
            ), f"Output batch size mismatch in {geo_encoder_class.__name__}."
