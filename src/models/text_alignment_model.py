from typing import override, Dict, Tuple

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.pred_heads.linear_pred_head import BasePredictionHead
from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder


class TextAlignmentModel(BaseModel):
    def __init__(
            self,
            eo_encoder: BaseEOEncoder,
            text_encoder: BaseTextEncoder,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_fn: BaseLossFn,
            trainable_modules: list[str] | None = None,
            prediction_head: BasePredictionHead | None = None,
            num_classes: int | None = None,
    ) -> None:
        super().__init__(trainable_modules, optimizer, scheduler, loss_fn, num_classes)

        # Encoders configuration
        self.eo_encoder = eo_encoder
        self.text_encoder = text_encoder
        # TODO: if eo==geoclip_img pass on shared mlp

        # Extra projector for text encoder if eo and text dim not match
        if self.eo_encoder.output_dim != self.text_encoder.output_dim:
            self.text_encoder.add_projector(projected_dim = self.eo_encoder.output_dim)
            self.trainable_modules.append('text_encoder.extra_projector')

        # Prediction head
        self.prediction_head = prediction_head
        if self.prediction_head is not None:
            self.prediction_head.set_dim(input_dim=self.eo_encoder.output_dim, output_dim=num_classes)
            self.prediction_head.configure_nn()

        # Freezing requested parts
        self.freezer()

    @override
    def forward(
            self,
            batch: Dict[str, torch.Tensor],
            mode: str = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # EMbed modalities
        eo_feats = self.eo_encoder(batch)
        text_feats = self.text_encoder(batch, mode)
        return eo_feats, text_feats

    @override
    def _step(
            self,
            batch: Dict[str, torch.Tensor],
            mode: str='train'
    ) -> torch.Tensor:
        # Embed
        eo_feats, text_feats = self.forward(batch, mode)
        local_batch_size = eo_feats.size(0)

        # batch recomposing in ddp
        if self.trainer.world_size > 1:
            feats = torch.stack([eo_feats, text_feats], dim=0)
            feats = self.all_gather(feats)
            feats = feats.reshape(2, -1, feats.size(-1))
            eo_feats, text_feats = feats[0], feats[1]

        # Get similarities
        with torch.no_grad():
            _ = self._cos_sim_calc(eo_feats, text_feats, mode)

        # Get loss
        loss = self.loss_fn(eo_feats, text_feats)
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=local_batch_size)
        if self.loss_fn.__getattr__('log_temp') and mode == 'train':
            self.log(f'temp', self.loss_fn.__getattr__('log_temp').exp(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=local_batch_size)

        return loss

    def _cos_sim_calc(self, eo_feats, text_feats, mode, log=True):

        # Similarity matrx
        cos_sim_matrix = F.cosine_similarity(eo_feats[:, None, :], text_feats[None, :, :], dim=-1)
        local_batch_size = eo_feats.size(0)


        # Average for positive and negative pairs
        # TODO change label option if we change what gets treated to be pos/neg
        id_matrix = torch.eye(cos_sim_matrix.shape[0], dtype=torch.bool)
        pos_sim = cos_sim_matrix[id_matrix]
        neg_sim = cos_sim_matrix[~id_matrix]

        # Average
        avr_sim = torch.mean(cos_sim_matrix)
        sub_neg_sim = neg_sim[torch.randperm(len(neg_sim))[:len(pos_sim)]] # pick same amount of negatives as positives
        balanced_sim = torch.cat([pos_sim, sub_neg_sim], dim=0)
        balanced_avr_sim = torch.mean(balanced_sim)

        if log:
            self.log(f'{mode}_avr_sim', avr_sim, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=local_batch_size)
            self.log(f'{mode}_avr_sim_balanced', balanced_avr_sim, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=local_batch_size)
            self.log(f'{mode}_pos_sim', torch.mean(pos_sim), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=local_batch_size)
            self.log(f'{mode}_neg_sim', torch.mean(neg_sim), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=local_batch_size)
        return avr_sim, pos_sim, neg_sim


if __name__ == "__main__":
    _ = TextAlignmentModel(None, None, None, None, None, None, None)