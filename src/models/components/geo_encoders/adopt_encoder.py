from typing import Dict, List, override

import hydra
import torch

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.utils.logging_utils import log_model_loading


def adopt_encoder(ckpt_path: str) -> BaseGeoEncoder:
    """Return geo_encoder from a provided checkpoint.

    :param ckpt_path: path to checkpoint file
    :return: trained geo encoder
    """
    # Get checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Get skeleton
    geo_config = ckpt["hyper_parameters"].get("geo_encoder")
    encoder: BaseGeoEncoder = hydra.utils.instantiate(geo_config)
    print("---Adopted encoder------")
    encoder.setup()
    print("------------------------")

    # Load in the weights
    state_dict = {
        k.replace("geo_encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if "geo_encoder." in k
    }
    res = encoder.load_state_dict(state_dict, strict=False)
    log_model_loading("geo_encoder_ckpt", res)

    encoder.setup = lambda *args, **kwargs: None  # TODO: switch to maybe self.setup flag

    return encoder
