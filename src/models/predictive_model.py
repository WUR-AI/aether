from typing import Dict, override

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.geo_encoders.encoder_wrapper import EncoderWrapper
from src.models.components.geo_encoders.tabular_encoder import TabularEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.metrics.metrics_wrapper import MetricsWrapper
from src.models.components.pred_heads.linear_pred_head import BasePredictionHead


class PredictiveModel(BaseModel):
    def __init__(
        self,
        geo_encoder: BaseGeoEncoder,
        prediction_head: BasePredictionHead,
        trainable_modules: list[str],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        loss_fn: BaseLossFn,
        metrics: MetricsWrapper,
        num_classes: int | None = None,
        tabular_dim: int | None = None,
        normalize_features: bool = True,
        standardize_targets: bool = False,
    ) -> None:
        """Implementation of the predictive model with replaceable GEO encoder, and prediction
        head.

        :param trainable_modules: which modules to train
        :param geo_encoder: module for encoding geo data
        :param prediction_head: module for making prediction from geo features
        :param optimizer: optimizer for the model weight update
        :param scheduler: scheduler for the model weight update
        :param loss_fn: loss function
        :param metrics: metrics to track for model performance estimation
        :param num_classes: number of target classes
        :param tabular_dim: number of tabular features
        :param normalize_features: if True, apply L2 normalisation to encoder output before the
            prediction head (default: True)
        :param standardize_targets: if True, z-score the target before computing the loss and
            invert the scaling before computing metrics, so all reported metrics are in the
            original target units (default: False)
        """

        super().__init__(
            trainable_modules=trainable_modules,
            geo_encoder=geo_encoder,
            text_encoder=None,
            prediction_head=prediction_head,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            metrics=metrics,
            num_classes=num_classes,
            tabular_dim=tabular_dim,
        )

        # Normalise features boolean
        self.normalize_features = normalize_features

        # Target standardisation — buffers are populated in setup() from the datamodule stats
        self.standardize_targets = standardize_targets
        self.register_buffer("target_mean", None)
        self.register_buffer("target_std", None)

    @override
    def _setup(self, stage: str) -> None:
        """Set up encoders and missing adapters/projectors based data-bound configurations (through
        datamodule), This method is called after trainer is initialized and datamodule is
        available.

        Otherwise, some configuration variables must be made available
        """
        self.num_classes = self.trainer.datamodule.num_classes
        self.tabular_dim = self.trainer.datamodule.tabular_dim

        if stage != "fit" and isinstance(self.trainable_modules, tuple):
            self.trainable_modules = list(self.trainable_modules)

        print("-------Model------------")
        self._setup_encoders_adapters()
        print("------------------------")

        # freeze requested parts
        self.freezer()

    def _setup_encoders_adapters(self):
        """Set up encoders and missing adapters/projectors."""
        # If tabular encoder used, we need to specify tabular dim and normalisation stats
        if isinstance(self.geo_encoder, TabularEncoder) or isinstance(
                self.geo_encoder, EncoderWrapper
        ):
            self.geo_encoder.set_tabular_input_dim(self.tabular_dim)

            stats = getattr(self.trainer.datamodule, "tabular_normalisation_stats", None)
            if stats is not None:
                mean, std = stats
                if isinstance(self.geo_encoder, TabularEncoder):
                    self.geo_encoder.set_normalisation_stats(mean, std)
                else:
                    self.geo_encoder.set_tabular_normalisation_stats(mean, std)

        # standarize target values if so requested
        if self.standardize_targets:
            stats = getattr(self.trainer.datamodule, "target_normalisation_stats", None)
            if stats is not None:
                self.target_mean, self.target_std = stats

        # Setup encoders that need data-depended configurations
        new_modules = [f"geo_encoder.{i}]" for i in self.geo_encoder.setup()]
        self.trainable_modules.extend(new_modules)

        # Configure prediction head based on geo-encoder output_dim
        self.prediction_head.set_dim(
            input_dim=self.geo_encoder.output_dim, output_dim=self.num_classes
        )
        self.prediction_head.setup()
        if "prediction_head" not in self.trainable_modules:
            self.trainable_modules.append("prediction_head")

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of a batch through the model."""
        feats = self.geo_encoder(batch)
        if self.normalize_features:
            feats = F.normalize(feats, dim=-1)
        return self.prediction_head(feats)

    @override
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> torch.Tensor:
        """Step logic of forward pass, metric calculation."""

        # Forward pass
        preds = self.forward(batch)
        target = batch.get("target")

        # args for logging
        log_kwargs = dict(
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=preds.size(0)
        )

        # calculate loss (standardized or raw targets)
        if self.standardize_targets and self.target_mean is not None:
            # Compute loss in standardised space (preds are in standardised space too)
            target_scaled = (target - self.target_mean) / self.target_std
            loss = self.loss_fn(preds, target_scaled)
            # Invert scaling so all logged metrics are in original target units
            preds_for_metrics = preds * self.target_std + self.target_mean
        else:
            loss = self.loss_fn(preds, target)
            preds_for_metrics = preds

        # logging
        metrics = self.metrics(pred=preds_for_metrics, batch=batch, mode=mode)
        self.log(f"{mode}_loss", loss, **log_kwargs)
        self.log_dict(metrics, **log_kwargs)

        return loss
