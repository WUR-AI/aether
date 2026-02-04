import json
import os

import pandas as pd
import pytest
import torch

from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder
from src.models.components.text_encoders.clip_text_encoder import ClipTextEncoder
from src.models.components.text_encoders.llm2clip_text_encoder import (
    LLM2CLIPTextEncoder,
)


# Initialisation of text encoders involve downloading the large models
@pytest.mark.slow
def test_text_encoder_generic_properties(create_butterfly_dataset):
    """This test checks that all text encoders implement the basic properties and methods."""
    list_text_encoders = [ClipTextEncoder, LLM2CLIPTextEncoder]
    ds, dm = create_butterfly_dataset
    batch = next(iter(dm.train_dataloader()))
    text_input = batch.get("text")
    assert text_input is not None, f"text input is None in the batch from datamodule. {batch}"

    for text_encoder_class in list_text_encoders:
        text_encoder = text_encoder_class()
        assert hasattr(
            text_encoder, "processor"
        ), f"'processor' attribute missing in {text_encoder_class.__name__}."
        assert hasattr(
            text_encoder, "model"
        ), f"'model' attribute missing in {text_encoder_class.__name__}."
        assert hasattr(
            text_encoder, "projector"
        ), f"'projector' attribute missing in {text_encoder_class.__name__}."
        assert hasattr(
            text_encoder, "output_dim"
        ), f"'output_dim' attribute missing in {text_encoder_class.__name__}."
        assert hasattr(
            text_encoder, "extra_projector"
        ), f"'extra_projector' attribute missing in {text_encoder_class.__name__}."
        assert hasattr(
            text_encoder, "forward"
        ), f"'forward' method missing in {text_encoder_class.__name__}."
        assert callable(
            getattr(text_encoder, "forward")
        ), f"'forward' is not callable in {text_encoder_class.__name__}."
        for mode in ["train", "val", "test"]:
            feats = text_encoder.forward(batch, mode=mode)
            assert isinstance(
                feats, torch.Tensor
            ), f"'forward' method of {text_encoder_class.__name__} does not return a torch.Tensor in {mode} mode."
            assert (
                feats.shape[0] == dm.batch_size_per_device
            ), f"Output batch size mismatch in {text_encoder_class.__name__} in {mode} mode."
