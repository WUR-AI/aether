import logging
import os

import open_clip
import torch
from huggingface_hub import hf_hub_download

from models.components.geo_encoders.remote_clip_img_encoder import RemoteClipImgEncoder
from models.components.text_encoders.remote_clip_text_encoder import (
    RemoteCLIPTextEncoder,
)
from models.text_alignment_model import TextAlignmentModel

log = logging.getLogger(__name__)


def build_RemoteCLIP_model(
    model_name: str = "ViT-L-14",
    hf_cache_dir: str = "../.cache",
    return_geo_encoder: bool = True,
    return_text_encoder: bool = True,
    preprocessing: str | None = None,
    **kwargs,
):
    """Implements RemoteCLIP model as TextAlignmentModel."""

    models = {"RN50": 1024, "ViT-B-32": 512, "ViT-L-14": 768}

    assert model_name in models.keys()

    checkpoint_path = hf_hub_download(
        "chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir=f"{hf_cache_dir}"
    )

    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)

    ckpt = torch.load(
        f"{os.path.dirname(checkpoint_path)}/RemoteCLIP-{model_name}.pt", map_location="cpu"
    )
    log.info(model.load_state_dict(ckpt))

    out_dim = models[model_name]

    if return_geo_encoder:
        assert (
            preprocessing == "div_10000"
        ), "S2 must be preprocessed with preprocessing set to 'div_10000'"
        geo_encoder = RemoteClipImgEncoder(geo_encoder=model.visual, out_dim=out_dim)
    if return_text_encoder:
        model.visual = None
        text_encoder = RemoteCLIPTextEncoder(model=model, tokenizer=tokenizer, out_dim=out_dim)
        if return_geo_encoder:
            return TextAlignmentModel(
                geo_encoder=geo_encoder,
                text_encoder=text_encoder,
                **kwargs,
            )
        else:
            return text_encoder
    elif return_geo_encoder:
        return geo_encoder
