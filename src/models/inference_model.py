import os
from typing import Dict, Tuple, override

import hydra
import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.metrics.metrics_wrapper import MetricsWrapper
from src.models.components.pred_heads.linear_pred_head import BasePredictionHead
from src.models.components.text_encoders.base_text_encoder import (
    BaseTextEncoder,
)
from src.utils import RankedLogger
from src.utils.errors import FileNotSpecified
from src.utils.logging_utils import log_model_loading

log = RankedLogger(__name__, rank_zero_only=True)


class InferenceModel(BaseModel):
    def __init__(
        self,
        geo_encoder: BaseGeoEncoder | None,
        text_encoder: BaseTextEncoder | None,
        prediction_head: BasePredictionHead | None,
        num_classes: int | None,
        metrics: MetricsWrapper | None = None,
        ks: list[int] | None = [5, 10, 15],
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

        # Params from alignment model
        self.match_to_geo = match_to_geo
        self.ks = ks

    @override
    def _setup(self, stage: str) -> None:
        """Set up the network."""
        if stage != "inference":
            raise ValueError(f"Trying to {stage} inference model")

        print("-------Model------------")
        # Configure encoders
        if hasattr(self, "geo_encoder"):
            self.geo_encoder.setup()
        if hasattr(self, "text_encoder"):
            self.text_encoder.setup()

        if hasattr(self, "text_encoder") and hasattr(self, "geo_encoder"):
            # Configure optional extra projection so text embeddings match geo embeddings.
            if self.text_encoder.output_dim != self.geo_encoder.output_dim:
                if self.match_to_geo:
                    self.text_encoder.add_projector(projected_dim=self.geo_encoder.output_dim)
                else:
                    self.geo_encoder.add_projector(projected_dim=self.text_encoder.output_dim)
        # Configure prediction head
        if hasattr(self, "prediction_head") and self.prediction_head.net is None:
            if self.num_classes is None:
                raise ValueError(
                    "InferenceModel requires `num_classes` to build the prediction head."
                )
            self.prediction_head.set_dim(
                input_dim=self.geo_encoder.output_dim, output_dim=self.num_classes
            )
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

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Model forward logic."""

        # Embed modalities
        if hasattr(self, "geo_encoder"):
            geo_feats = self.geo_encoder(batch)
        if hasattr(self, "text_encoder"):
            text_feats = self.text_encoder(batch, mode)
        if hasattr(self, "prediction_head"):
            pred_feats = self.prediction_head(geo_feats)

        # Change dtype of geo data if it does not match text dtype
        if (
            hasattr(self, "text_encoder")
            and hasattr(self, "geo_encoder")
            and geo_feats.dtype != text_feats.dtype
        ):
            geo_feats = geo_feats.to(text_feats.dtype)

            return pred_feats, geo_feats, text_feats

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
        else:
            with torch.no_grad():
                concept_embeds = self.text_encoder({"text": self.concepts}, mode="train")

        # Similarity
        geo_embeds = F.normalize(geo_embeds, dim=1)
        concept_embeds = F.normalize(concept_embeds, dim=1)
        similarity_matrix = concept_embeds @ geo_embeds.T

        return similarity_matrix


def _is_prefix_trained(trainable_modules: list[str], prefix: str) -> bool:
    """True if any trainable module starts with `prefix` (before dot)."""
    return any(m.split(".")[0] == prefix for m in trainable_modules)


def load_inference_model(inference_ckpt_path: str) -> InferenceModel:
    """Loads inference model from a merged checkpoint.

    :param inference_ckpt_path: path to inference model weights
    :return: an InferenceModel with pre-trained weights
    """
    inference_ckpt = torch.load(inference_ckpt_path, map_location="cpu", weights_only=False)
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

    # Sanity check: ensure geo encoder configs match.
    align_ckpt["hyper_parameters"]["geo_encoder"] = pred_ckpt["hyper_parameters"].get(
        "geo_encoder"
    )
    if pred_ckpt["hyper_parameters"].get("geo_encoder") != align_ckpt["hyper_parameters"].get(
        "geo_encoder"
    ):
        log.warning("Geo encoder configs differ between checkpoints; results may be invalid.")
        if input("Do you want to proceed? y/n").lower() == "n":
            return None

    pred_trainable_modules = pred_ckpt["hyper_parameters"].get("trainable_modules", [])
    align_trainable_modules = align_ckpt["hyper_parameters"].get("trainable_modules", [])

    geo_pred_encoder_trained = _is_prefix_trained(pred_trainable_modules, "geo_encoder")
    geo_align_encoder_trained = _is_prefix_trained(align_trainable_modules, "geo_encoder")

    if geo_pred_encoder_trained and geo_align_encoder_trained:
        raise ValueError("Models are not aligned: both checkpoints trained geo_encoder.")

    # Instantiate InferenceModel via hydra, using alignment encoder configs with prediction model head configs
    inference_hparams = align_ckpt["hyper_parameters"]
    inference_hparams.update(
        {
            "_target_": "src.models.inference_model.InferenceModel",
            "prediction_head": pred_ckpt["hyper_parameters"].get("prediction_head"),
            "num_classes": pred_ckpt["hyper_parameters"].get("num_classes"),
        }
    )
    inference_hparams["text_encoder"]["hf_cache_dir"] = os.path.join(
        cfg.paths.cache_dir, "huggingface"
    )

    model: InferenceModel = hydra.utils.instantiate(inference_hparams)
    model.setup("inference")

    # Load alignment weights first (text encoder).
    text_state = {
        k: v for k, v in align_ckpt["state_dict"].items() if k.startswith("text_encoder.")
    }
    res = model.load_state_dict(text_state, strict=False)
    log_model_loading("text_encoder", res)

    if cfg.training_order[0] == "prediction_model" and not geo_align_encoder_trained:
        geo_state = {
            k: v for k, v in pred_ckpt["state_dict"].items() if k.startswith("geo_encoder.")
        }
    else:
        geo_state = {
            k: v for k, v in align_ckpt["state_dict"].items() if k.startswith("geo_encoder.")
        }
    res = model.load_state_dict(geo_state, strict=False)
    log_model_loading("geo_encoder", res)

    # Load prediction head weights from predictive ckpt.
    head_state = {
        k: v for k, v in pred_ckpt["state_dict"].items() if k.startswith("prediction_head.")
    }
    res = model.load_state_dict(head_state, strict=False)
    log_model_loading("Predictive_head", res)

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
