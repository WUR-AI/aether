from typing import Dict, override

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.geo_encoders.encoder_wrapper import EncoderWrapper
from src.models.components.geo_encoders.tabular_encoder import TabularEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.metrics.metrics_wrapper import MetricsWrapper
from src.models.components.pred_heads.linear_pred_head import (
    BasePredictionHead,
)


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

    @override
    def _setup(self, stage: str) -> None:
        """Set up encoders and missing adapters/projectors based data-bound configurations (through
        datamodule), This method is called after trainer is initialized and datamodule is
        available.

        Otherwise, some configuration variables must be made available
        """
        if stage != "fit" and isinstance(self.trainable_modules, tuple):
            self.trainable_modules = list(self.trainable_modules)

        print("-------Model------------")
        # If tabular encoder used, we need to specify tabular dim
        if isinstance(self.geo_encoder, TabularEncoder) or isinstance(
            self.geo_encoder, EncoderWrapper
        ):
            self.geo_encoder.set_tabular_input_dim(self.tabular_dim)

        # Setup encoders
        new_modules = [f"geo_encoder.{i}" for i in self.geo_encoder.setup() or []]
        self.trainable_modules.extend(new_modules)

        # Configure prediction head based on geo-encoder output_dim
        self.prediction_head.set_dim(
            input_dim=self.geo_encoder.output_dim, output_dim=self.num_classes
        )
        self.trainable_modules.extend(self.prediction_head.setup() or [])
        print("------------------------")

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

        loss = self.loss_fn(preds, batch.get("target"))
        metrics = self.metrics(pred=preds, batch=batch, mode=mode)

        # Logging
        log_kwargs = dict(
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=preds.size(0)
        )
        self.log(f"{mode}_loss", loss, **log_kwargs)
        self.log_dict(metrics, **log_kwargs)

        return loss
