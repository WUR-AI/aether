import json, os
import pandas as pd
import pytest
import torch

from src.models.components.pred_heads.base_pred_head import BasePredictionHead
from src.models.components.pred_heads.linear_pred_head import LinearPredictionHead
from src.models.components.pred_heads.mlp_pred_head import MLPPredictionHead
from src.models.components.eo_encoders.geoclip import GeoClipCoordinateEncoder

@pytest.mark.slow
def test_pred_head_generic_properties(create_butterfly_dataset):
    ds, dm = create_butterfly_dataset
    batch = next(iter(dm.train_dataloader()))
    eo_encoder = GeoClipCoordinateEncoder()
    feats = eo_encoder.forward(batch)

    list_pred_heads = [LinearPredictionHead, MLPPredictionHead]
    for pred_head_class in list_pred_heads:
        pred_head = pred_head_class()
        assert hasattr(pred_head, "set_dim"), f"'set_dim' method missing in {pred_head_class.__name__}."
        assert callable(getattr(pred_head, "set_dim")), f"'set_dim' is not callable in {pred_head_class.__name__}."
        pred_head.set_dim(eo_encoder.output_dim, ds.num_classes)
        assert hasattr(pred_head, "input_dim"), f"'input_dim' attribute missing in {pred_head_class.__name__}."
        assert hasattr(pred_head, "output_dim"), f"'output_dim' attribute missing in {pred_head_class.__name__}."
        assert hasattr(pred_head, "configure_nn"), f"'configure_nn' method missing in {pred_head_class.__name__}."
        assert callable(getattr(pred_head, "configure_nn")), f"'configure_nn' is not callable in {pred_head_class.__name__}."
        pred_head.configure_nn()
        assert hasattr(pred_head, "net"), f"'net' attribute missing in {pred_head_class.__name__}."
        assert hasattr(pred_head, "forward"), f"'forward' method missing in {pred_head_class.__name__}."
        assert callable(getattr(pred_head, "forward")), f"'forward' is not callable in {pred_head_class.__name__}."
        out = pred_head.forward(feats)
        assert isinstance(out, torch.Tensor), f"'forward' method of {pred_head_class.__name__} does not return a torch.Tensor."
        assert out.shape == (dm.batch_size_per_device, ds.num_classes), f"Output shape mismatch in {pred_head_class.__name__}."

