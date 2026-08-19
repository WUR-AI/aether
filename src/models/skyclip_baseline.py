import glob
import logging
import os
import urllib.request
import zipfile

import numpy as np
import open_clip
import torch

from models.components.geo_encoders.skyclip_encoder import SkyClipImgEncoder
from models.components.text_encoders.skyclip_text_encoder import SkyCLIPTextEncoder
from models.text_alignment_model import TextAlignmentModel

log = logging.getLogger(__name__)

SKYSCRIPT_S3_BASE = "https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt"

SKYCLIP_CHECKPOINTS = {
    "SkyCLIP_ViT_L14_top30pct": ("SkyCLIP_ViT_L14_top30pct.zip", "ViT-L-14", 768),
    "SkyCLIP_ViT_L14_top50pct": ("SkyCLIP_ViT_L14_top50pct.zip", "ViT-L-14", 768),
    "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS": (
        "SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS.zip",
        "ViT-L-14",
        768,
    ),
    "SkyCLIP_ViT_L14_top30pct_multi_objects": (
        "SkyCLIP_ViT_L14_top30pct_multi_objects.zip",
        "ViT-L-14",
        768,
    ),
    "SkyCLIP_ViT_B32_top50pct": ("SkyCLIP_ViT_B32_top50pct.zip", "ViT-B-32", 512),
    "CLIP_ViT_L14_LAION_RS": ("CLIP_ViT_L14_LAION_RS.zip", "ViT-L-14", 768),
}


def _download_and_extract_skyclip_ckpt(model_name: str, cache_dir: str) -> str:
    """Downloads + unzips a SkyCLIP checkpoint from S3 (not on the HF hub), caching by model_name,
    and returns the path to the extracted .pt file."""
    zip_name, _, _ = SKYCLIP_CHECKPOINTS[model_name]
    extract_dir = os.path.join(cache_dir, model_name)
    zip_path = os.path.join(cache_dir, zip_name)

    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.isdir(extract_dir) or not glob.glob(
        os.path.join(extract_dir, "**", "*.pt"), recursive=True
    ):
        if not os.path.isfile(zip_path):
            url = f"{SKYSCRIPT_S3_BASE}/{zip_name}"
            log.info(f"Downloading SkyCLIP checkpoint from {url}")
            urllib.request.urlretrieve(url, zip_path)  # nosec B310 - scheme validated above

        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    pt_files = glob.glob(os.path.join(extract_dir, "**", "*.pt"), recursive=True)
    assert pt_files, f"No .pt checkpoint found after extracting {zip_path} to {extract_dir}"
    if len(pt_files) > 1:
        log.warning(f"Multiple .pt files found in {extract_dir}, using {pt_files[0]}")
    return pt_files[0]


def _load_skyclip_state_dict(ckpt_path: str) -> dict:
    """Loads a SkyCLIP/open_clip training checkpoint."""
    safe_globals = [np.core.multiarray.scalar, np.dtype]
    for name in ("Float64DType", "Float32DType", "Int64DType"):
        dtype_cls = getattr(getattr(np, "dtypes", None), name, None)
        if dtype_cls is not None:
            safe_globals.append(dtype_cls)

    try:
        with torch.serialization.safe_globals(safe_globals):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        log.warning(
            f"weights_only=True load failed even with numpy globals allowlisted "
            f"({e}); falling back to weights_only=False for {ckpt_path}. "
            f"Only do this because this checkpoint is from a trusted source (SkyScript authors' S3 bucket)."
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def build_SkyCLIP_model(
    model_name: str = "SkyCLIP_ViT_L14_top50pct",
    hf_cache_dir: str = "../.cache",
    return_geo_encoder: bool = True,
    return_text_encoder: bool = True,
    preprocessing: str | None = None,
    **kwargs,
):
    """Implements SkyCLIP (SkyScript) model as TextAlignmentModel."""

    assert (
        model_name in SKYCLIP_CHECKPOINTS.keys()
    ), f"model_name must be one of {list(SKYCLIP_CHECKPOINTS.keys())}, got {model_name}"
    zip_name, base_arch, out_dim = SKYCLIP_CHECKPOINTS[model_name]

    ckpt_path = _download_and_extract_skyclip_ckpt(model_name, cache_dir=hf_cache_dir)

    model, _, preprocess = open_clip.create_model_and_transforms(base_arch)
    tokenizer = open_clip.get_tokenizer(base_arch)

    state_dict = _load_skyclip_state_dict(ckpt_path)

    log.info(model.load_state_dict(state_dict, strict=False))

    if return_geo_encoder:
        assert (
            preprocessing == "div_2000"
        ), "S2 must be preprocessed with preprocessing set to 'div_2000'"
        geo_encoder = SkyClipImgEncoder(geo_encoder=model.visual, out_dim=out_dim)
    if return_text_encoder:
        model.visual = None
        text_encoder = SkyCLIPTextEncoder(model=model, tokenizer=tokenizer, out_dim=out_dim)
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
