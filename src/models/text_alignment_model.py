from io import text_encoding
from typing import Dict, Tuple, override

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.eo_encoders.multimodal_encoder import MultiModalEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.metrics.contrastive_validation import (
    RetrievalContrastiveValidation,
)
from src.models.components.metrics.metrics_wrapper import MetricsWrapper
from src.models.components.pred_heads.linear_pred_head import (
    BasePredictionHead,
)
from src.models.components.text_encoders.base_text_encoder import (
    BaseTextEncoder,
)


class TextAlignmentModel(BaseModel):
    def __init__(
        self,
        eo_encoder: BaseEOEncoder,
        text_encoder: BaseTextEncoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: BaseLossFn,
        trainable_modules: list[str],
        metrics: MetricsWrapper,
        num_classes: int | None = None,
        tabular_dim: int | None = None,
        prediction_head: BasePredictionHead | None = None,
        ks: list[int] | None = [5, 10, 15],
    ) -> None:
        """Implementation of contrastive text-eo modality alignment model.

        :param eo_encoder: eo encoder module (replaceable)
        :param text_encoder: text encoder module (replaceable)
        :param optimizer: optimizer to use for training
        :param scheduler: scheduler to use for training
        :param loss_fn: loss function to use (contrastive)
        :param trainable_modules: list of modules to train (parts/modules or modules, modules)
        :param metrics: metrics to use for model performance evaluation
        :param num_classes: number of target classes
        :param tabular_dim: number of tabular features
        :param prediction_head: prediction head
        """
        super().__init__(
            trainable_modules, optimizer, scheduler, loss_fn, metrics, num_classes, tabular_dim
        )

        self.ks = ks
        self.log_kwargs = dict(on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Encoders configuration
        self.eo_encoder = eo_encoder
        # TODO: move to multi-modal eo encoder
        if (
            isinstance(self.eo_encoder, MultiModalEncoder)
            and self.eo_encoder.use_tabular
            and not self.eo_encoder._tabular_ready
        ):
            self.eo_encoder.build_tabular_branch(tabular_dim)

        self.text_encoder = text_encoder
        # TODO: if eo==geoclip_img pass on shared mlp

        # Extra projector for text encoder if eo and text dim not match
        if self.eo_encoder.output_dim != self.text_encoder.output_dim:
            self.text_encoder.add_projector(projected_dim=self.eo_encoder.output_dim)
            self.trainable_modules.append("text_encoder.extra_projector")

        # Prediction head
        self.prediction_head = prediction_head
        if self.prediction_head is not None:
            self.prediction_head.set_dim(
                input_dim=self.eo_encoder.output_dim, output_dim=num_classes
            )
            self.prediction_head.configure_nn()

        # Unify dtypes
        if self.eo_encoder.dtype != self.text_encoder.dtype:
            self.eo_encoder = self.eo_encoder.to(self.text_encoder.dtype)
            print(f"Eo encoder dtype changed to {self.eo_encoder.dtype}")

        # Freezing requested parts
        self.freezer()

    def setup(self, stage: str) -> None:
        self.concept_configs = self.trainer.datamodule.concept_configs
        self.concepts = [c["concept_caption"] for c in self.concept_configs]

        self.contrastive_val = RetrievalContrastiveValidation(self.ks, self.concept_configs)
        self.outputs_epoch_memory = []

        for trainable_module in self.trainable_modules:
            if "text" in trainable_module:
                self.concept_embeds = None
                return

        # Encode concepts if text branch is frozen
        with torch.no_grad():
            self.concept_embeds = self.text_encoder({"text": self.concepts}, mode="train")

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Model forward logic."""

        # Embed modalities
        eo_feats = self.eo_encoder(batch)
        text_feats = self.text_encoder(batch, mode)
        return eo_feats, text_feats

    @override
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train"):
        """Model step logic."""

        # Embed
        eo_feats, text_feats = self.forward(batch, mode)
        local_batch_size = eo_feats.size(0)

        # batch recomposing in ddp
        if self.trainer.world_size > 1:
            feats = torch.stack([eo_feats, text_feats], dim=0)
            feats = self.all_gather(feats)
            feats = feats.reshape(2, -1, feats.size(-1))
            eo_feats, text_feats = feats[0], feats[1]

        # Get loss
        loss = self.loss_fn(eo_feats, text_feats)

        # Get similarities
        with torch.no_grad():
            metrics = self.metrics(
                mode=mode,
                eo_feats=eo_feats,
                text_feats=text_feats,
                local_batch_size=local_batch_size,
            )

        # Logging
        self.log(f"{mode}_loss", loss, batch_size=local_batch_size, **self.log_kwargs)

        if self.loss_fn.__getattr__("log_temp") and mode == "train":
            self.log(
                "temp",
                self.loss_fn.__getattr__("log_temp").exp(),
                batch_size=local_batch_size,
                **self.log_kwargs,
            )

        self.log_dict(metrics, batch_size=local_batch_size, **self.log_kwargs)

        if mode in ["val", "test"]:
            self.outputs_epoch_memory.append(
                {
                    "eo_feats": eo_feats.detach(),
                    "aux_vals": batch.get("aux", {}).get("aux").detach(),
                }
            )

        return loss

    @override
    def _on_epoch_end(self, mode: str):

        # Combine batches
        eo_feats = torch.cat([x["eo_feats"] for x in self.outputs_epoch_memory], dim=0)

        aux_vals = torch.cat([x["aux_vals"] for x in self.outputs_epoch_memory], dim=0)

        # Rank on similarity
        similarity = self.concept_similarities(eo_feats)

        concept_scores = self.contrastive_val(similarity, aux_values=aux_vals)
        # TODO pearson

        avr_scores = {f"{mode}_avr_top-{k}": [] for k in self.ks}
        for i, result in concept_scores.items():
            print(f'\nConcept "{self.concepts[i]}" average top-k accuracies in {mode} split:')
            for k, v in result.items():
                print(f"Top-{k}: {v:.1f}%")
                avr_scores[f"{mode}_avr_top-{k}"].append(v)

        for k, v in avr_scores.items():
            avr_scores[k] = sum(v) / len(v)

        self.log_dict(avr_scores)

        # Reset memory
        self.outputs_epoch_memory.clear()

    @override
    def on_validation_epoch_end(self):
        return self._on_epoch_end("val")

    @override
    def on_test_epoch_end(self):
        return self._on_epoch_end("test")

    def concept_similarities(self, eo_embeds, concept=None) -> torch.Tensor:
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
        eo_embeds = F.normalize(eo_embeds, dim=1)
        concept_embeds = F.normalize(concept_embeds, dim=1)
        similarity_matrix = concept_embeds @ eo_embeds.T

        return similarity_matrix
