import logging

import open_clip
import torch
from huggingface_hub import hf_hub_download, list_repo_files

from models.components.geo_encoders.remote_clip_img_encoder import RemoteClipImgEncoder
from models.components.text_encoders.remote_clip_text_encoder import (
    RemoteCLIPTextEncoder,
)
from models.text_alignment_model import TextAlignmentModel

log = logging.getLogger(__name__)

HF_REPO = "Zilun/GeoRSCLIP"

# model_name -> (open_clip arch, base init used for GeoRSCLIP continual pretraining, embed dim)
GEORSCLIP_MODELS = {
    "ViT-B-32": ("ViT-B-32", "openai", 512),
    "ViT-H-14": ("ViT-H-14", "laion2b_s32b_b79k", 1024),
}


def _resolve_georsclip_ckpt_filename(model_name: str) -> str:
    """The GeoRSCLIP .pt files live directly in the HF repo but their exact
    subpath isn't guaranteed - resolve by listing repo files rather than
    hardcoding a path, matching on the model_name substring."""
    files = list_repo_files(HF_REPO)
    matches = [f for f in files if model_name in f and f.endswith(".pt")]
    assert matches, (
        f"No checkpoint matching '{model_name}' found in {HF_REPO}. " f"Available files: {files}"
    )
    if len(matches) > 1:
        log.warning(
            f"Multiple candidate checkpoints for {model_name} in {HF_REPO}: {matches}, using {matches[0]}"
        )
    return matches[0]


def build_GeoRSCLIP_model(
    model_name: str = "ViT-B-32",
    hf_cache_dir: str = "../.cache",
    return_geo_encoder: bool = True,
    return_text_encoder: bool = True,
    preprocessing: str | None = None,
    **kwargs,
):
    """Implements GeoRSCLIP (trained on RS5M) as TextAlignmentModel."""
    assert (
        model_name in GEORSCLIP_MODELS.keys()
    ), f"model_name must be one of {list(GEORSCLIP_MODELS.keys())}, got {model_name}"
    arch, base_init, out_dim = GEORSCLIP_MODELS[model_name]

    ckpt_filename = _resolve_georsclip_ckpt_filename(model_name)
    ckpt_path = hf_hub_download(HF_REPO, ckpt_filename, cache_dir=hf_cache_dir)

    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=base_init)
    tokenizer = open_clip.get_tokenizer(arch)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    log.info(model.load_state_dict(state_dict, strict=False))

    if return_geo_encoder:
        assert (
            preprocessing == "div_2000"
        ), "S2 must be preprocessed with preprocessing set to 'div_2000'"
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
