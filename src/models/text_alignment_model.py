from typing import Dict, Tuple, override

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
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> torch.Tensor:
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
        log_kwargs = dict(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=local_batch_size,
        )
        self.log(f"{mode}_loss", loss, **log_kwargs)

        if self.loss_fn.__getattr__("log_temp") and mode == "train":
            self.log("temp", self.loss_fn.__getattr__("log_temp").exp(), **log_kwargs)

        self.log_dict(metrics, **log_kwargs)

        return loss
