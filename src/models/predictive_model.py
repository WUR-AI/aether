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
        scheduler: torch.optim.lr_scheduler,
        loss_fn: BaseLossFn,
        metrics: MetricsWrapper,
        normalize_features: bool = True,
    ) -> None:
        """Implementation of the predictive model with replaceable GEO encoder, and prediction
        head.

        :param geo_encoder: geo encoder module (replaceable)
        :param prediction_head: prediction head module (replaceable)
        :param trainable_modules: list of modules to train (parts/modules or modules, modules)
        :param optimizer: optimizer to use for training
        :param scheduler: scheduler to use for training
        :param loss_fn: loss function to use
        :param metrics: metrics to use for model performance evaluation
        :param num_classes: number of target classes
        :param tabular_dim: number of tabular features
        :param normalize_features: if True, apply L2 normalisation to encoder output before the
            prediction head (default: True)
        """

        super().__init__(trainable_modules, optimizer, scheduler, loss_fn, metrics)

        # Geo encoder configuration
        self.geo_encoder = geo_encoder

        # Prediction head
        self.prediction_head = prediction_head

        # Normalise features boolean
        self.normalize_features = normalize_features

    @override
    def setup(self, stage: str) -> None:
        self.num_classes = self.trainer.datamodule.num_classes
        self.tabular_dim = self.trainer.datamodule.tabular_dim

        if stage != "fit":
            if isinstance(self.trainable_modules, tuple):
                self.trainable_modules = list(self.trainable_modules)

        print("-------Model------------")
        self.setup_encoders_adapters()
        print("------------------------")

        # Freezing requested parts
        self.freezer()

    def setup_encoders_adapters(self):
        """Set up encoders and missing adapters/projectors."""
        # TODO: move to multi-modal eo encoder

        # If tabular encoder used, we need to specify tabular dim
        if isinstance(self.geo_encoder, TabularEncoder) or isinstance(
            self.geo_encoder, EncoderWrapper
        ):
            self.geo_encoder.set_tabular_input_dim(self.tabular_dim)

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
        feats = self.geo_encoder(batch)
        if self.normalize_features:
            feats = F.normalize(feats, dim=-1)
        return self.prediction_head(feats)

    @override
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> torch.Tensor:
        preds = self.forward(batch)

        log_kwargs = dict(
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=preds.size(0)
        )
        loss = self.loss_fn(preds, batch.get("target"))
        self.log(f"{mode}_loss", loss, **log_kwargs)

        metrics = self.metrics(pred=preds, batch=batch, mode=mode)
        self.log_dict(metrics, **log_kwargs)

        return loss


if __name__ == "__main__":
    _ = PredictiveModel(None, None, None, None, None, None, None)
