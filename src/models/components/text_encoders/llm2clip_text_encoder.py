from typing import Dict, override

import torch
from huggingface_hub import snapshot_download
from llm2vec import LLM2Vec
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
        self._download(hf_cache_dir)

        self.projector = AutoModel.from_pretrained(
            "microsoft/LLM2CLIP-Openai-L-14-224",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            revision="50ed31c5248d8ff124893719e37829d59376be81",
            cache_dir=hf_cache_dir,
        ).eval()

        self.projector.vision_model = None
        self.projector.visual_projection = None

        llm_model_name = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
        llm_revision = "main"

        config = AutoConfig.from_pretrained(
            llm_model_name,
            trust_remote_code=True,
            revision=llm_revision,
            cache_dir=hf_cache_dir,
        )
        config._attn_implementation = "eager"

        llm_model = LlamaEncoderModel.from_pretrained(
            llm_model_name,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=False,
            revision=llm_revision,
            cache_dir=hf_cache_dir,
        )
        llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.processor = AutoTokenizer.from_pretrained(
            llm_model_name,
            revision=llm_revision,
            cache_dir=hf_cache_dir,
        )

        # Caption to vector with the llama LLM
        self.model = LLM2Vec(
            llm_model, self.processor, pooling_mode="mean", max_length=512, doc_max_length=512
        )

        self.output_dim = 1280

    @staticmethod
    def _download(hf_cache_dir: str) -> None:
        llm_revision = "main"
        snapshot_download(
            "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",
            revision=llm_revision,
            cache_dir=hf_cache_dir,
        )
        snapshot_download(
            "microsoft/LLM2CLIP-Openai-L-14-224",
            revision="50ed31c5248d8ff124893719e37829d59376be81",
            cache_dir=hf_cache_dir,
        )

    @override
    def forward(self, batch: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        """Forward pass through text encoder."""
        # Get text inputs
        text_input = batch.get("text")

        if mode == "train":
            captions_flattened = text_input
            num_per_location = 1
        else:
            num_per_location = len(text_input[0])
            captions_flattened = [c for loc in text_input for c in loc]

        # LLM is frozen, no gradients needed
        with torch.no_grad():
            # Embed
            text_embeds = self.model.encode(
                captions_flattened, convert_to_tensor=True, device=self.device
            )
            # Change dtype
            text_embeds = text_embeds.to(dtype=self.projector.dtype, device=self.projector.device)

        # Project to align with ViT in LLM2CLIP
        text_embeds = self.projector.get_text_features(text_embeds)

        if self.extra_projector is not None:
            text_embeds = self.extra_projector(text_embeds)

        if mode != "train":
            text_embeds = text_embeds.reshape(-1, num_per_location, text_embeds.shape[-1])
            text_embeds = text_embeds.mean(dim=1)

        return text_embeds
