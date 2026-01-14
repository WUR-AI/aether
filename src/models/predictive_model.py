from typing import Dict, override

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.loss_fns.top_k_accuracy import TopKAccuracy
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
        num_classes: int,
    ) -> None:
        """Implementation of the predictive model with replaceable EO encoder, and prediction head.

        :param eo_encoder: eo encoder module (replaceable)
        :param prediction_head: prediction head module (replaceable)
        :param trainable_modules: list of modules to train (parts/modules or modules, modules)
        :param optimizer: optimizer to use for training
        :param scheduler: scheduler to use for training
        :param loss_fn: loss function to use
        :param num_classes: number of target classes
        """
        super().__init__(trainable_modules, optimizer, scheduler, loss_fn, num_classes)
        # EO encoder configuration
        self.eo_encoder = eo_encoder

        # Prediction head
        self.prediction_head = prediction_head
        self.prediction_head.set_dim(input_dim=self.eo_encoder.output_dim, output_dim=num_classes)
        self.prediction_head.configure_nn()
        self.trainable_modules.append("prediction_head")

        # Freezing requested parts
        self.freezer()

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = self.eo_encoder(batch)
        return self.prediction_head(feats)

    @override
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> torch.Tensor:
        feats = self.forward(batch)
        loss = self.loss_fn(feats, batch.get("target"))

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        mse_loss = F.mse_loss(feats, batch.get("target"))
        self.log(f"{mode}_mse_loss", mse_loss, on_step=False, on_epoch=True)
        top_k_accs = TopKAccuracy(k_list=[1, 5, 10])(feats, batch.get("target"))
        for k, acc in top_k_accs.items():
            self.log(f"{mode}_top_{k}_acc", acc, on_step=False, on_epoch=True)
        return loss


if __name__ == "__main__":
    _ = PredictiveModel(None, None, None, None, None, None, None)
