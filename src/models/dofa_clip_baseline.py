import logging

from models.components.geo_encoders.dofa_clip_img_encoder import DOFAClipImgEncoder
from models.components.text_encoders.dofa_clip_text_encoder import DOFACLIPTextEncoder
from models.text_alignment_model import TextAlignmentModel
from utils.dofa_clip.factory import create_model_from_pretrained, get_tokenizer

log = logging.getLogger(__name__)


def build_DOFACLIP_model(
    model_name: str = "ViT-L-14",
    hf_cache_dir: str = "../.cache",
    return_geo_encoder: bool = True,
    return_text_encoder: bool = True,
    geo_data_name: str | None = None,
    preprocessing: str | None = None,
    **kwargs,
):
    """Implements DOFACLIP model as TextAlignmentModel.

    https://github.com/xiong-zhitong/DOFA-CLIP
    """

    models = {
        "ViT-B-16": (768, "hf-hub:earthflow/GeoLB-ViT-B-16-SigLIP-All-EO"),
        "ViT-L-14": (1152, "hf-hub:earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"),
    }

    assert model_name in models.keys()

    hf_repo = models[model_name][1]

    model, preprocess = create_model_from_pretrained(hf_repo, cache_dir=hf_cache_dir)
    tokenizer = get_tokenizer(hf_repo)

    out_dim = models[model_name][0]

    if return_geo_encoder:
        assert (
            preprocessing == "div_2000"
        ), "S2 must be preprocessed with preprocessing set to 'div_2000'"
        geo_encoder = DOFAClipImgEncoder(
            geo_encoder=model.visual, out_dim=out_dim, geo_data_name=geo_data_name
        )
    if return_text_encoder:
        model.visual = None
        text_encoder = DOFACLIPTextEncoder(
            model=model, tokenizer=tokenizer, out_dim=out_dim, context_len=model.context_length
        )
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
