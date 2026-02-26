from typing import Dict, override

import torch
from llm2vec import LLM2Vec
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder
from src.models.components.text_encoders.llm2clip.llama import LlamaEncoderModel


class LLM2CLIPTextEncoder(BaseTextEncoder):
    def __init__(self, hf_cache_dir: str = "../.cache", output_normalization="l2") -> None:
        """LLM2CLIP text encoder implementation. Uses LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned as
        LLM and LLM2CLIP trained adapter.

        :param hf_cache_dir: huggingface cache directory
        :param output_normalization: output normalization type
        """
        super().__init__()

        # Adapter and image encoder
        self.projector = AutoModel.from_pretrained(
            "microsoft/LLM2CLIP-Openai-L-14-224",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            revision="50ed31c5248d8ff124893719e37829d59376be81",  # pin revision for full reproducibility
            cache_dir=hf_cache_dir,
        ).eval()

        # TODO: If we want to reuse the vision  part this is the fix place
        self.projector.vision_model = None
        self.projector.visual_projection = None

        # The LLM sentence encoder
        llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
        config = AutoConfig.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
        )
        config._attn_implementation = "eager"

        llm_model = LlamaEncoderModel.from_pretrained(
            llm_model_name,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=False,  # local code
            cache_dir=hf_cache_dir,
        )
        llm_model.config._name_or_path = (
            "meta-llama/Meta-Llama-3-8B-Instruct"  # Workaround for LLM2VEC
        )
        self.processor = AutoTokenizer.from_pretrained(llm_model_name)

        # Caption to vector with the llama LLM
        self.model = LLM2Vec(
            llm_model, self.processor, pooling_mode="mean", max_length=512, doc_max_length=512
        )

        self.output_dim = 1280

    @override
    def forward(self, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        """Forward pass through text encoder."""
        # Get text inputs
        text_input = batch.get("text")

        if mode == "train":
            text_input = [text_input]
        # Embed text and if not train average all templates
        avr_embeds = []
        for captions_per_row in text_input:
            # LLM is frozen, no gradients needed
            with torch.no_grad():
                # Embed
                text_embeds = self.model.encode(
                    captions_per_row, convert_to_tensor=True, device=self.device
                )

                # Change dtype
                text_embeds = text_embeds.to(
                    dtype=self.projector.dtype, device=self.projector.device
                )

            # Project to align with ViT in LLM2CLIP
            text_embeds = self.projector.get_text_features(text_embeds)

            if self.extra_projector is not None:
                text_embeds = self.extra_projector(text_embeds)

            if mode != "train":
                avr_embeds.append(text_embeds.mean(dim=0))

        if mode != "train":
            text_embeds = torch.stack(avr_embeds, dim=0)

        return text_embeds
