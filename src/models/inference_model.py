import os
from typing import Dict, Tuple, override

import hydra
import omegaconf
import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.metrics.metrics_wrapper import MetricsWrapper
from src.models.components.pred_heads.linear_pred_head import BasePredictionHead
from src.models.components.projectors.base_projector import BaseProjector
from src.models.components.text_encoders.base_text_encoder import (
    BaseTextEncoder,
)
from src.utils import RankedLogger
from src.utils.errors import FileNotSpecified
from src.utils.logging_utils import log_model_loading
from utils.errors import IllegalArgumentCombination

log = RankedLogger(__name__, rank_zero_only=True)


class InferenceModel(BaseModel):
    def __init__(
        self,
        geo_encoder: BaseGeoEncoder,
        text_encoder: BaseTextEncoder,
        prediction_head: BasePredictionHead,
        num_classes: int,
        geo_adapter: BaseProjector | None = None,
        text_adapter: BaseProjector | None = None,
        metrics: MetricsWrapper | None = None,
        ks: list[int] | None = None,
        match_to_geo: bool = True,
        **kwargs,
    ) -> None:
        """Inference model.

        :param geo_encoder: module for encoding geo data
        :param text_encoder: module for encoding text data
        :param prediction_head: module for making prediction from geo features
        :param num_classes: number of target classes
        :param metrics: metrics to track for model performance estimation
        :param ks: list of ks
        :param match_to_geo: whether to match dimensions of text encoder to geo_encoder or visa-
            versa
        """

        super().__init__(
            trainable_modules=[],
            geo_encoder=geo_encoder,
            text_encoder=text_encoder,
            prediction_head=prediction_head,
            optimizer=None,
            scheduler=None,
            loss_fn=None,
            metrics=metrics,
            num_classes=num_classes,
            tabular_dim=None,
        )

        self.geo_adapter = geo_adapter
        self.text_adapter = text_adapter

        # Params from alignment model
        self.match_to_geo = match_to_geo
        self.ks = ks or [5, 10, 15]

    @override
    def _setup(self, stage: str) -> None:
        """Set up the network."""
        if stage != "inference":
            raise ValueError(f"Trying to {stage} inference model")

        log.info("-------Model------------")
        # Configure encoders
        if self.geo_encoder:
            self.geo_encoder.setup()
        if self.geo_adapter:
            self.geo_adapter.set_input_dim(self.geo_encoder.output_dim)
            self.geo_adapter.setup()

        if self.text_encoder:
            self.text_encoder.setup()
        if self.text_adapter:
            self.text_adapter.set_input_dim(self.text_encoder.output_dim)
            self.text_adapter.setup()

        # Sanity check for dimension matching
        geo_branch_dim = (
            self.geo_adapter.output_dim if self.geo_adapter else self.geo_encoder.output_dim
        )
        text_branch_dim = (
            self.text_adapter.output_dim if self.text_adapter else self.text_encoder.output_dim
        )

        if geo_branch_dim != text_branch_dim:
            raise IllegalArgumentCombination(
                "Provided prediction and alignment model checkpoints are not mergeable"
            )

        # Configure prediction head
        if self.prediction_head and self.prediction_head.net is None:
            if self.num_classes is None:
                raise ValueError(
                    "InferenceModel requires `num_classes` to build the prediction head."
                )
            input_dim = self.geo_encoder.output_dim
            self.prediction_head.set_dim(input_dim=input_dim, output_dim=self.num_classes)
            self.prediction_head.setup()
        print("------------------------")

    @override
    def _step(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        """Step forward computation of the model."""
        pass

    def forward_text(self, text: list[str]) -> torch.Tensor:
        batch = {"text": text}
        text_feats = self.text_encoder(batch, "train")
        if self.text_adapter:
            text_feats = self.text_adapter(text_feats)
        return text_feats

    def forward_geo(self, batched_eo) -> Tuple[torch.Tensor, torch.Tensor | None]:
        geo_feats = self.geo_encoder(batched_eo)

        if self.prediction_head:
            pred = self.prediction_head(geo_feats)

        if self.geo_adapter:
            geo_feats = self.geo_adapter(geo_feats)
        return geo_feats, pred

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        geo_feats, pred = self.forward_geo(batch)
        text_feats = self.text_encoder(batch, "test")

        # Change dtype of geo data if it does not match text dtype
        if geo_feats.dtype != text_feats.dtype:
            geo_feats = geo_feats.to(text_feats.dtype)

        return pred, geo_feats, text_feats

    def concept_similarities(self, geo_embeds, concept=None) -> torch.Tensor:
        # Get concept embeddings
        if concept is not None:
            # If only one concept is provided
            if isinstance(concept, str):
                concept = [concept]
            with torch.no_grad():
                concept_embeds = self.text_encoder({"text": concept}, mode="train")

        elif self.concept_embeds is not None:
            concept_embeds = self.concept_embeds

        # Similarity
        geo_embeds = F.normalize(geo_embeds, dim=1)
        concept_embeds = F.normalize(concept_embeds, dim=1)
        similarity_matrix = concept_embeds @ geo_embeds.T

        return similarity_matrix


def _is_prefix_trained(trainable_modules: list[str], prefix: str) -> bool:
    """True if any trainable module starts with `prefix` (before dot)."""
    return any(m.split(".")[0] == prefix for m in trainable_modules)


def load_inference_model(
    inference_ckpt_path: str, cache_path: str | None, patch_mlp_path: bool = True
) -> InferenceModel:
    """Loads inference model from a merged checkpoint.

    :param inference_ckpt_path: path to inference model weights
    :return: an InferenceModel with pre-trained weights
    """
    inference_ckpt = torch.load(inference_ckpt_path, map_location="cpu", weights_only=False)
    if cache_path:
        inference_ckpt["hyper_parameters"]["text_encoder"]["hf_cache_dir"] = cache_path

    if patch_mlp_path:
        # Older checkpoints may have the MLPProjector path as `src.models.components.geo_encoders.mlp_projector.MLPProjector`, but it has been moved to `src.models.components.projectors_adapters.mlp_projector.MLPProjector`. This patch updates the path in the hyperparameters.
        ckpt_cfg = inference_ckpt["hyper_parameters"]
        yaml_str = omegaconf.OmegaConf.to_yaml(ckpt_cfg)
        yaml_str = yaml_str.replace(
            "src.models.components.geo_encoders.mlp_projector.MLPProjector",
            "src.models.components.projectors_adapters.mlp_projector.MLPProjector",  # ← update this
        )
        ckpt_cfg = omegaconf.OmegaConf.create(yaml_str)
        model = hydra.utils.instantiate(ckpt_cfg)
    else:
        model = hydra.utils.instantiate(inference_ckpt["hyper_parameters"])

    model.setup("inference")
    res = model.load_state_dict(inference_ckpt["state_dict"], strict=False)
    log_model_loading("inference_ckpt", res)
    return model


def merge_inference_model(cfg, save_ckpt=False) -> InferenceModel | None:
    """Configures the inference model from the predictive + alignment checkpoints.

    :param cfg: A DictConfig configuration composed by Hydra.
    :param save_ckpt: Whether to save the model or not.
    :return: an InferenceModel with pre-trained weights
    """

    # Stitch the inference model from the predictive + alignment checkpoints.
    pred_ckpt_path = cfg.get("predictive_ckpt_path") or FileNotSpecified(
        'You must specify predictive model weight path as "predictive_ckpt_path"'
    )
    align_ckpt_path = cfg.get("alignment_ckpt_path") or FileNotSpecified(
        'You must specify alignment model weight path as "alignment_ckpt_path"'
    )
    # TODO: remove dataset saving into the checkpoint
    pred_ckpt = torch.load(pred_ckpt_path, map_location="cpu", weights_only=False)
    align_ckpt = torch.load(align_ckpt_path, map_location="cpu", weights_only=False)

    inference_hparams = {}
    # Sanity check: ensure geo encoder configs match.
    if pred_ckpt["hyper_parameters"].get("geo_encoder") != align_ckpt["hyper_parameters"].get(
        "geo_encoder"
    ):
        raise IllegalArgumentCombination(
            "Geo encoder configs differ between checkpoints; results may be invalid."
        )

    pred_trainable_modules = pred_ckpt["hyper_parameters"].get("trainable_modules", [])
    align_trainable_modules = align_ckpt["hyper_parameters"].get("trainable_modules", [])

    geo_pred_encoder_trained = _is_prefix_trained(pred_trainable_modules, "geo_encoder")
    geo_align_encoder_trained = _is_prefix_trained(align_trainable_modules, "geo_encoder")

    if geo_pred_encoder_trained and geo_align_encoder_trained:
        raise ValueError("Models are not aligned: both checkpoints trained geo_encoder.")

    # Instantiate InferenceModel via hydra, using alignment encoder configs with prediction model head configs
    inference_hparams.update(align_ckpt["hyper_parameters"])
    inference_hparams.update(
        {
            "_target_": "src.models.inference_model.InferenceModel",
            "prediction_head": pred_ckpt["hyper_parameters"].get("prediction_head"),
            "num_classes": pred_ckpt["hyper_parameters"].get("num_classes"),
        }
    )
    inference_hparams["trainable_modules"] = None
    inference_hparams["text_encoder"]["hf_cache_dir"] = os.path.join(
        cfg.paths.cache_dir, "huggingface"
    )

    model: InferenceModel = hydra.utils.instantiate(inference_hparams)
    model.setup("inference")

    collected_states = {}

    # Get text encoder
    collected_states.update(
        {
            k: v
            for k, v in align_ckpt["state_dict"].items()
            if k.startswith("text_encoder.")
            and k != "text_encoder.model.text_model.embeddings.position_ids"
        }
    )

    # Get text adapter
    collected_states.update(
        {k: v for k, v in align_ckpt["state_dict"].items() if k.startswith("text_adapter.")}
    )

    # Get geo_encoder
    if cfg.training_order[0] == "prediction_model" and not geo_align_encoder_trained:
        collected_states.update(
            {k: v for k, v in pred_ckpt["state_dict"].items() if k.startswith("geo_encoder.")}
        )
    else:
        collected_states.update(
            {
                k: v
                for k, v in align_ckpt["state_dict"].items()
                if k.startswith("geo_encoder.") or k.startswith("geo_adapter.")
            }
        )

    # Get geo_adapter
    collected_states.update(
        {k: v for k, v in align_ckpt["state_dict"].items() if k.startswith("geo_adapter.")}
    )

    # Get prediction head weights from predictive ckpt.
    collected_states.update(
        {k: v for k, v in pred_ckpt["state_dict"].items() if k.startswith("prediction_head.")}
    )

    # Load collected states
    res = model.load_state_dict(collected_states, strict=False)
    log_model_loading("Inference Model", res)

    # Save model
    if save_ckpt:
        save_path = cfg.get("save_inference_ckpt_path")
        if not save_path:
            print("Model could not be saved as save_path was not provided")

        # Get `state_dict`
        state_dict = model.state_dict()

        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({"state_dict": state_dict, "hyper_parameters": inference_hparams}, save_path)
        log.info(f"Saved merged inference checkpoint to: {save_path}")

    return model
