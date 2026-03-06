from typing import Dict, override

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.eo_encoders.multimodal_encoder import MultiModalEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.metrics.metrics_wrapper import MetricsWrapper
from src.models.components.pred_heads.linear_pred_head import (
    BasePredictionHead,
)


class PredictiveModel(BaseModel):
    def __init__(
        self,
        eo_encoder: BaseEOEncoder,
        prediction_head: BasePredictionHead,
        trainable_modules: list[str],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: BaseLossFn,
        metrics: MetricsWrapper,
    ) -> None:
        """Implementation of the predictive model with replaceable EO encoder, and prediction head.

        :param eo_encoder: eo encoder module (replaceable)
        :param prediction_head: prediction head module (replaceable)
        :param trainable_modules: list of modules to train (parts/modules or modules, modules)
        :param optimizer: optimizer to use for training
        :param scheduler: scheduler to use for training
        :param loss_fn: loss function to use
        :param metrics: metrics to use for model performance evaluation
        :param num_classes: number of target classes
        :param tabular_dim: number of tabular features
        """

        super().__init__(trainable_modules, optimizer, scheduler, loss_fn, metrics)

        # EO encoder configuration
        self.eo_encoder = eo_encoder

        # Prediction head
        self.prediction_head = prediction_head

    @override
    def setup(self, stage: str) -> None:
        self.num_classes = self.trainer.datamodule.num_classes
        self.tabular_dim = self.trainer.datamodule.tabular_dim

        self.setup_encoders_adapters()

        # Freezing requested parts
        self.freezer()

    def setup_encoders_adapters(self):
        """Set up encoders and missing adapters/projectors."""
        # TODO: move to multi-modal eo encoder
        if (
            isinstance(self.eo_encoder, MultiModalEncoder)
            and self.eo_encoder.use_tabular
            and not self.eo_encoder._tabular_ready
        ):
            self.eo_encoder.build_tabular_branch(self.tabular_dim)

        self.prediction_head.set_dim(
            input_dim=self.eo_encoder.output_dim, output_dim=self.num_classes
        )
        self.prediction_head.configure_nn()
        if "prediction_head" not in self.trainable_modules:
            self.trainable_modules.append("prediction_head")

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = self.eo_encoder(batch)
        feats = F.normalize(feats, dim=-1)
        return self.prediction_head(feats)

    @override
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> torch.Tensor:
        feats = self.forward(batch)

        log_kwargs = dict(
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=feats.size(0)
        )
        loss = self.loss_fn(feats, batch.get("target"))
        self.log(f"{mode}_loss", loss, **log_kwargs)

        metrics = self.metrics(pred=feats, batch=batch, mode=mode)
        self.log_dict(metrics, **log_kwargs)


if __name__ == "__main__":
    _ = PredictiveModel(None, None, None, None, None, None, None)
