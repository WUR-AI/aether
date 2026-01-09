from typing import override, Dict

import torch

from src.models.base_model import BaseModel
from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.pred_heads.linear_pred_head import BasePredictionHead


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
        super().__init__(trainable_modules, optimizer, scheduler, loss_fn, num_classes)
        # EO encoder configuration
        self.eo_encoder = eo_encoder

        # Prediction head
        self.prediction_head = prediction_head
        self.prediction_head.set_dim(input_dim=self.eo_encoder.output_dim, output_dim=num_classes)
        self.prediction_head.configure_nn()

        # Freezing requested parts
        self.freezer()

    @override
    def forward(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        feats = self.eo_encoder(batch)
        return self.prediction_head(feats)

    @override
    def _step(
            self,
            batch: Dict[str, torch.Tensor],
            mode: str='train'
    ) -> torch.Tensor:
        feats = self.forward(batch)
        loss = self.loss_fn(feats, batch.get('target'))

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

if __name__ == "__main__":
    _ = PredictiveModel(None, None, None, None, None, None, None)