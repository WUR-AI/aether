from io import text_encoding
from typing import Dict, Tuple, override

import numpy as np
import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
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
        geo_encoder: BaseGeoEncoder,
        text_encoder: BaseTextEncoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: BaseLossFn,
        trainable_modules: list[str],
        metrics: MetricsWrapper,
        prediction_head: BasePredictionHead | None = None,
        ks: list[int] | None = [5, 10, 15],
        match_to_geo: bool = True,
        num_classes: int | None = None,
        tabular_dim: int | None = None,
    ) -> None:
        """Implementation of contrastive text-eo modality alignment model.

        :param geo_encoder: geo encoder module (replaceable)
        :param text_encoder: text encoder module (replaceable)
        :param optimizer: optimizer to use for training
        :param scheduler: scheduler to use for training
        :param loss_fn: loss function to use (contrastive)
        :param trainable_modules: list of modules to train (parts/modules or modules, modules)
        :param metrics: metrics to use for model performance evaluation
        :param prediction_head: prediction head
        :param ks: list of ks
        :param match_to_geo: whether to match dimensions of text encoder to geo_encoder or visa-
            versa
        :param num_classes: number of target classes
        :param tabular_dim: number of tabular features
        """
        super().__init__(
            trainable_modules, optimizer, scheduler, loss_fn, metrics, num_classes, tabular_dim
        )

        # Metrics
        self.ks = ks
        self.log_kwargs = dict(on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Encoders configuration
        self.geo_encoder = geo_encoder
        self.text_encoder = text_encoder
        self.match_to_geo = match_to_geo

        # Prediction head
        self.prediction_head = prediction_head

    @override
    def setup(self, stage: str = "fit") -> None:
        """Updates model based data-bound configurations (through datamodule), This method is
        called after trainer is initialized and datamodule is available.

        Otherwise, some configuration variables must be made available
        """

        if self._trainer is not None:
            self.num_classes = self.trainer.datamodule.num_classes
            self.tabular_dim = self.trainer.datamodule.tabular_dim

        # Set up encoders and missing adapters/projectors
        print("-------Model------------")
        self.setup_encoders_adapters()
        print("------------------------")

        # Freeze not requested parts
        if stage in ["inference", "test"]:
            self.full_freezer()
        else:
            self.freezer()

            # Configure contrastive retrieval evaluation
            self.setup_retrieval_evaluation()

    def setup_encoders_adapters(self):
        """Set up encoders and missing adapters/projectors."""
        # Setup encoders that need data-depended configurations
        new_modules = [f"geo_encoder.{i}" for i in self.geo_encoder.setup()]
        self.trainable_modules.extend(new_modules)

        # Extra projector for text encoder if eo and text dim not match
        if self.geo_encoder.output_dim != self.text_encoder.output_dim:
            if self.match_to_geo:
                self.text_encoder.add_projector(projected_dim=self.geo_encoder.output_dim)
                self.trainable_modules.append("text_encoder.extra_projector")
            else:
                self.geo_encoder.add_projector(projected_dim=self.text_encoder.output_dim)
                self.trainable_modules.append("geo_encoder.extra_projector")

        # Configure prediction head based on geo-encoder output_dim
        if self.prediction_head is not None:
            self.prediction_head.set_dim(
                input_dim=self.geo_encoder.output_dim, output_dim=self.num_classes
            )
            self.prediction_head.setup()

    def setup_retrieval_evaluation(self):
        self.concept_configs = self.trainer.datamodule.concept_configs
        self.concepts = [c["concept_caption"] for c in self.concept_configs]
        self.concept_names = [
            f"{c['col'].replace('aux_', '')}_{'max' if c['is_max'] else 'min'}"
            for c in self.concept_configs
        ]

        dataset_names = ["train", "val", "test"]
        self.dynamic_k_baselines = {}
        for dataset_name in dataset_names:
            if not hasattr(self.trainer.datamodule, f"data_{dataset_name}"):
                continue

            tmp_ds = getattr(self.trainer.datamodule, f"data_{dataset_name}")
            n_ds = len(tmp_ds)
            self.dynamic_k_baselines[dataset_name] = {}

            # Placeholder for all concepts
            aux_vals_per_concept = {i: [] for i in range(len(self.concept_configs))}

            for item in tmp_ds:
                aux_data = item["aux"]["aux"]
                for i_c, c in enumerate(self.concept_configs):
                    aux_col_id = c["id"]
                    aux_vals_per_concept[i_c].append(aux_data[aux_col_id])

            # Compute per concept
            for i_c, c in enumerate(self.concept_configs):
                c_name = self.concept_names[i_c]
                aux_vals_current_ds = aux_vals_per_concept[i_c]

                theta_k = self.find_elbow_point(aux_vals_current_ds)
                self.concept_configs[i_c][
                    "theta_k"
                ] = theta_k  # assign new theta_k to concept_configs for later use in validation
                if c["is_max"]:
                    n_baseline = sum(aux_val >= theta_k for aux_val in aux_vals_current_ds)
                else:
                    n_baseline = sum(aux_val <= theta_k for aux_val in aux_vals_current_ds)
                self.dynamic_k_baselines[dataset_name][c_name] = n_baseline / n_ds * 100

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
        geo_feats = self.geo_encoder(batch)
        text_feats = self.text_encoder(batch, mode)

        # Change dtype of geo data if it does not match text dtype
        if geo_feats.dtype != text_feats.dtype:
            geo_feats = geo_feats.to(text_feats.dtype)
        return geo_feats, text_feats

    @override
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train"):
        """Model step logic."""

        # Embed
        geo_feats, text_feats = self.forward(batch, mode)
        local_batch_size = geo_feats.size(0)

        # batch recomposing in ddp
        if self.trainer.world_size > 1:
            feats = torch.stack([geo_feats, text_feats], dim=0)
            feats = self.all_gather(feats)
            feats = feats.reshape(2, -1, feats.size(-1))
            geo_feats, text_feats = feats[0], feats[1]

        # Get loss
        loss = self.loss_fn(geo_feats, text_feats)

        # Get similarities
        with torch.no_grad():
            metrics = self.metrics(
                mode=mode,
                geo_feats=geo_feats,
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
                    "geo_feats": geo_feats.detach(),
                    "aux_vals": batch.get("aux", {}).get("aux").detach(),
                }
            )

        return loss

    def _on_epoch_end(self, mode: str):

        # Combine batches
        geo_feats = torch.cat([x["geo_feats"] for x in self.outputs_epoch_memory], dim=0)

        aux_vals = torch.cat([x["aux_vals"] for x in self.outputs_epoch_memory], dim=0)

        # Rank on similarity
        similarity = self.concept_similarities(geo_feats)

        concept_scores = self.contrastive_val(similarity, aux_values=aux_vals)
        # TODO pearson

        avr_scores = {f"{mode}_avr_top-{k}": [] for k in self.ks}
        for i, result in concept_scores.items():
            print(f'\nConcept "{self.concepts[i]}" average top-k accuracies in {mode} split:')
            for k, v in result.items():
                if k == "dynamic_k":
                    self.log(f"dyn_k_{self.self.concept_names[i]}", v, **self.log_kwargs)
                    indexed_v = (v - self.dynamic_k_baselines[mode][self.concept_names[i]]) / (
                        100 - self.dynamic_k_baselines[mode][self.concept_names[i]]
                    )
                    self.log(f"dyn_k_index_{self.concept_names[i]}", indexed_v, **self.log_kwargs)
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

    @staticmethod
    def find_elbow_point(vals):
        vals = np.sort(vals)
        x = np.arange(len(vals)) / len(vals)
        y = vals
        slope = (y[-1] - y[0]) / (x[-1] - x[0])  # diagonal from first to last point
        intercept = y[0] - slope * x[0]
        orthogonal_slope = -1 / slope

        intercepts_orthogonal = y - orthogonal_slope * x
        intersection_diagonal_orthogonal = (intercepts_orthogonal - intercept) / (
            slope - orthogonal_slope
        )
        distances = np.sqrt(
            (x - intersection_diagonal_orthogonal) ** 2 + (y - (slope * x + intercept)) ** 2
        )  # distance to diagonal
        elbow_index = np.argmax(distances)
        elbow_point = y[elbow_index]
        return elbow_point
