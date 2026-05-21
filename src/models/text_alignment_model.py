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
from src.models.components.text_encoders.base_text_encoder import (
    BaseTextEncoder,
)


class TextAlignmentModel(BaseModel):
    def __init__(
        self,
        trainable_modules: list[str],
        geo_encoder: BaseGeoEncoder,
        text_encoder: BaseTextEncoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: BaseLossFn,
        metrics: MetricsWrapper,
        num_classes: int | None = None,
        tabular_dim: int | None = None,
        ks: list[int] | None = [5, 10, 15],
        match_to_geo: bool = True,
    ) -> None:
        """Implementation of contrastive text-eo modality alignment model.

        :param trainable_modules: which modules to train
        :param geo_encoder: module for encoding geo data
        :param text_encoder: module for encoding text data
        :param optimizer: optimizer for the model weight update
        :param scheduler: scheduler for the model weight update
        :param loss_fn: loss function
        :param metrics: metrics to track for model performance estimation
        :param num_classes: number of target classes
        :param tabular_dim: number of tabular features
        :param ks: list of ks
        :param match_to_geo: whether to match dimensions of text encoder to geo_encoder or visa-
            versa
        """
        super().__init__(
            trainable_modules=trainable_modules,
            geo_encoder=geo_encoder,
            text_encoder=text_encoder,
            prediction_head=None,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            num_classes=num_classes,
            tabular_dim=tabular_dim,
        )

        # Metrics
        self.ks = ks
        self.log_kwargs = dict(on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.match_to_geo = match_to_geo

    @override
    def _setup(self, stage: str = "fit") -> None:
        """Set up encoders and missing adapters/projectors based data-bound configurations (through
        datamodule), This method is called after trainer is initialized and datamodule is
        available.

        Otherwise, some configuration variables must be made available
        """
        # Set up encoders and missing adapters/projectors
        print("-------Model------------")
        new_modules = [f"geo_encoder.{i}" for i in self.geo_encoder.setup() or []]
        new_modules.extend([f"text_encoder.{i}" for i in self.text_encoder.setup() or []])
        self.trainable_modules.extend(new_modules)

        # Extra projector for text encoder if eo and text dim not match
        if self.geo_encoder.output_dim != self.text_encoder.output_dim:
            if self.match_to_geo:
                self.text_encoder.add_projector(projected_dim=self.geo_encoder.output_dim)
                self.trainable_modules.append("text_encoder.extra_projector")
            else:
                self.geo_encoder.add_projector(projected_dim=self.text_encoder.output_dim)
                self.trainable_modules.append("geo_encoder.extra_projector")

        # Configure contrastive retrieval evaluation
        self.setup_retrieval_evaluation(verbose=0)
        print("------------------------")

    def setup_retrieval_evaluation(
        self,
        use_saved_threshold_if_available=True,
        overwrite_existing_thresholds=False,
        save_newly_computed_threshold=True,
        compute_train_threshold=True,
        verbose=1,
    ):
        """Set up the contrastive retrieval evaluation by computing dynamic k baselines and
        initializing the validation object.

        Parameters:
        - use_saved_threshold_if_available: whether to use saved thresholds if available
        - overwrite_existing_thresholds: whether to overwrite existing thresholds with newly computed ones
        - save_newly_computed_threshold: whether to save newly computed thresholds to a new concept configs file
        - compute_train_threshold: whether to compute thresholds only using the train set
        - verbose: whether to print the dynamic ks and their baselines
        - verbose: whether to print the dynamic ks and their baselines
        """
        assert (
            compute_train_threshold
        ), "Currently the only implementation of computing thresholds is by computing them on the train set only, and then using these for the validation and test set too."
        if overwrite_existing_thresholds:
            use_saved_threshold_if_available = False
            save_newly_computed_threshold = True  # for reproducibility.

        self.concept_configs = self.trainer.datamodule.concept_configs
        self.concepts = [c["concept_caption"] for c in self.concept_configs]
        self.concept_names = [
            f"{c['col'].replace('aux_', '')}_{'max' if c['is_max'] else 'min'}"
            for c in self.concept_configs
        ]
        list_concept_ids_drop = []

        self.dynamic_k_baselines = {}
        for dataset_name in [
            "train",
            "val",
            "test",
        ]:  # ensure 'train' is first for use_train_threshold logic!
            if not hasattr(self.trainer.datamodule, f"data_{dataset_name}"):
                if dataset_name == "test":
                    assert (
                        save_newly_computed_threshold is False
                    ), "No test dataloader found, but save_newly_computed_threshold is True. However, the current implementation saves thresholds only after they have been calculated for the test set. Please provide a test dataloader or set save_newly_computed_threshold to False."
                continue

            tmp_ds = getattr(self.trainer.datamodule, f"data_{dataset_name}")
            n_ds = len(tmp_ds)
            self.dynamic_k_baselines[dataset_name] = {}

            if use_saved_threshold_if_available and all(
                (c.get("theta_k") is not None and c.get(f"n_baseline_{dataset_name}") is not None)
                for c in self.concept_configs
            ):  # no need to compute if all concepts have saved thresholds and baselines
                for i_c, c in enumerate(self.concept_configs):
                    c_name = self.concept_names[i_c]
                    theta_k = c["theta_k"]
                    n_baseline = c[f"n_baseline_{dataset_name}"]

                    if verbose:
                        print(
                            f"Concept '{self.concept_names[i_c]}' in {dataset_name} set: is_max={c['is_max']}, saved theta_k={theta_k:.6f}, saved baseline={n_baseline}/{n_ds} ({n_baseline / n_ds * 100:.1f}%)"
                        )
                    self.dynamic_k_baselines[dataset_name][c_name] = n_baseline / n_ds * 100

            else:
                print(
                    f"WARNING: No saved thresholds and baselines found for some or all concepts for {dataset_name}, computing new ones. This may take a while..."
                )
                print(
                    "To speed up this computation, make sure to run this method with a dataloader that has only the coordinates and aux data (no other EO data)."
                )
                if save_newly_computed_threshold:
                    print(
                        "The threshold values will be written to a new concept configs file if they are computed anew."
                    )
                else:
                    print(
                        "Consider setting save_newly_computed_threshold=True to store the computed thresholds and avoid recomputation in the future."
                    )
                # Iterate through dataset once to get aux values for all concepts (to avoid multiple iterations if multiple concepts). Best done with coords only dataset for speed!
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

                    if dataset_name == "train":
                        if overwrite_existing_thresholds:
                            theta_k = self.find_elbow_point(
                                aux_vals_current_ds
                            )  # only compute if not present
                        else:
                            theta_k = c.get("theta_k") or self.find_elbow_point(
                                aux_vals_current_ds
                            )  # only compute if not present
                        self.concept_configs[i_c]["theta_k"] = float(
                            theta_k
                        )  # assign new theta_k to concept_configs for later use in validation
                    else:
                        theta_k = self.concept_configs[i_c]["theta_k"]

                    n_baseline_max = sum(aux_val >= theta_k for aux_val in aux_vals_current_ds)
                    n_baseline_min = sum(aux_val <= theta_k for aux_val in aux_vals_current_ds)

                    if n_baseline_max < n_baseline_min:
                        if not c.get("is_max", True):
                            print(
                                f"Concept {c_name} has n_baseline_max < n_baseline_min but is_max is False. Therefore it will NOT be used/stored. Please check the concept configs or the computed theta_k for this concept."
                            )
                            if i_c not in list_concept_ids_drop:
                                list_concept_ids_drop.append(i_c)
                        n_baseline = n_baseline_max
                    else:
                        if c.get("is_max", False):
                            print(
                                f"Concept {c_name} has n_baseline_max >= n_baseline_min but is_max is True. Therefore it will NOT be used/stored. Please check the concept configs or the computed theta_k for this concept."
                            )
                            if i_c not in list_concept_ids_drop:
                                list_concept_ids_drop.append(i_c)
                        n_baseline = n_baseline_min

                    if n_baseline == n_ds:
                        n_baseline = (
                            n_ds - 1
                        )  # to avoid having a baseline of 100% (will still yield index score of 1)
                        if dataset_name == "train":
                            theta_k = (
                                min(aux_vals_current_ds) + 1e-6
                                if c["is_max"]
                                else max(aux_vals_current_ds) - 1e-6
                            )
                    self.concept_configs[i_c][f"n_baseline_{dataset_name}"] = int(n_baseline)

                    if verbose:
                        print(
                            f"Concept '{self.concept_names[i_c]}' in {dataset_name} set: is_max={c['is_max']}, original theta_k={self.concept_configs[i_c]['theta_k']:.6f}, new theta_k={theta_k:.6f}, baseline={n_baseline}/{n_ds} ({n_baseline / n_ds * 100:.1f}%)"
                        )
                    self.dynamic_k_baselines[dataset_name][c_name] = n_baseline / n_ds * 100

                if len(list_concept_ids_drop) > 0 and dataset_name == "test":
                    print(
                        f"Dropping concepts with ids {list_concept_ids_drop} and names {[self.concept_names[i] for i in list_concept_ids_drop]} from evaluation due to mismatch between is_max and whether n_baseline_max or n_baseline_min is smaller."
                    )
                    self.concept_configs = [
                        c
                        for i, c in enumerate(self.concept_configs)
                        if i not in list_concept_ids_drop
                    ]
                    self.concepts = [c["concept_caption"] for c in self.concept_configs]
                    self.concept_names = [
                        f"{c['col'].replace('aux_', '')}_{'max' if c['is_max'] else 'min'}"
                        for c in self.concept_configs
                    ]
                    self.dynamic_k_baselines[dataset_name] = {
                        c_name: baseline
                        for i, (c_name, baseline) in enumerate(
                            self.dynamic_k_baselines[dataset_name].items()
                        )
                        if i not in list_concept_ids_drop
                    }

                if (
                    save_newly_computed_threshold and dataset_name == "test"
                ):  # only save after computing on train set, and only if we are computing new thresholds (not just using saved ones), to avoid overwriting with the same values or with values computed on val/test set
                    self.trainer.datamodule.caption_builder.store_concept_thresholds(
                        self.concept_configs, update_self=True
                    )
                else:
                    self.trainer.datamodule.caption_builder.update_concept_thresholds(
                        self.concept_configs
                    )

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

    def _on_epoch_end(self, mode: str, verbose=0):

        # Combine batches
        geo_feats = torch.cat([x["geo_feats"] for x in self.outputs_epoch_memory], dim=0)

        aux_vals = torch.cat([x["aux_vals"] for x in self.outputs_epoch_memory], dim=0)

        # Rank on similarity
        similarity = self.concept_similarities(geo_feats)

        concept_scores = self.contrastive_val(similarity, aux_values=aux_vals)
        # TODO pearson

        avr_scores = {f"{mode}_avr_top-{k}": [] for k in self.ks}
        for i, result in concept_scores.items():
            if verbose:
                print(f'\nConcept "{self.concepts[i]}" average top-k accuracies in {mode} split:')
            for k, v in result.items():
                if k == "dynamic_k":
                    self.log(f"dyn_k_{self.concept_names[i]}", v, **self.log_kwargs)
                    indexed_v = (v - self.dynamic_k_baselines[mode][self.concept_names[i]]) / (
                        100 - self.dynamic_k_baselines[mode][self.concept_names[i]]
                    )
                    self.log(f"dyn_k_index_{self.concept_names[i]}", indexed_v, **self.log_kwargs)
                if verbose:
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
        """Vals is a list of tensor values."""
        with torch.no_grad():
            vals = torch.tensor(vals).cpu().numpy()
            vals = vals[~np.isnan(vals)]  # remove NaN values

            vals = np.sort(vals)
            vals = vals[vals > vals[0]]
            x = np.arange(len(vals)) / len(vals)
            y = vals
            if x[0] == x[-1]:  # all values are the same
                print(
                    "All values are the same, returning the value itself as elbow point.", vals[0]
                )
                return vals[0]
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
