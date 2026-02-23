"""
Regression variant of the predictive model (MSE / MAE / RMSE / R²).

Location: src/models/predictive_model_regression.py

Key changes vs original:
  - setup(stage) injects tabular_dim into MultiModalEncoder automatically,
    so tabular_dim never needs to be hardcoded in any config.
  - num_classes defaults to 1 (single LST value).
  - All regression metrics (MSE, RMSE, MAE, R²) are logged per split.
"""

from typing import Dict, override

import torch
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.components.eo_encoders.multimodal_encoder import MultiModalEncoder


class PredictiveRegressionModel(BaseModel):

    def __init__(
        self,
        eo_encoder,
        prediction_head,
        trainable_modules,
        optimizer,
        scheduler,
        loss_fn,
        num_classes: int = 1,
        **kwargs,
    ):
        super().__init__(
            trainable_modules=trainable_modules,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            num_classes=num_classes,
        )

        self.eo_encoder = eo_encoder
        self.prediction_head = prediction_head

        # Prediction head wiring happens AFTER setup() resolves tabular_dim.
        # If the encoder does NOT need tabular data, we can wire immediately.
        if not (
            isinstance(self.eo_encoder, MultiModalEncoder)
            and self.eo_encoder.use_tabular
            and not self.eo_encoder._tabular_ready
        ):
            self._wire_head()
            self.freezer() 

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """
        Called by Lightning after the datamodule is ready.
        Injects tabular_dim into the encoder if it needs it,
        then wires the prediction head dimensions.
        """
        if (
            isinstance(self.eo_encoder, MultiModalEncoder)
            and self.eo_encoder.use_tabular
            and not self.eo_encoder._tabular_ready
        ):
            # Pull tabular_dim from the datamodule — no hardcoding needed!
            tabular_dim = self.trainer.datamodule.tabular_dim
            self.eo_encoder.build_tabular_branch(tabular_dim)
            self._wire_head()

            self.freezer()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _wire_head(self) -> None:
        """Connect encoder output_dim → head input_dim, then build head layers."""
        self.prediction_head.set_dim(
            input_dim=self.eo_encoder.output_dim,
            output_dim=self.num_classes,
        )
        self.prediction_head.configure_nn()
        if "prediction_head" not in self.trainable_modules:  
            self.trainable_modules.append("prediction_head")

    # ------------------------------------------------------------------
    # Forward & step
    # ------------------------------------------------------------------

    @override
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = self.eo_encoder(batch)
        feats = F.normalize(feats, dim=-1)
        return self.prediction_head(feats)

    @override
    def _step(self, batch: Dict[str, torch.Tensor], mode: str = "train") -> torch.Tensor:
        y_hat = self.forward(batch)
        y = batch["target"]

        loss = self.loss_fn(y_hat, y)

        mse  = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(mse)
        mae  = F.l1_loss(y_hat, y)

        ss_res = torch.sum((y - y_hat) ** 2)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot

        log_kwargs = dict(on_step=False, on_epoch=True)
        self.log(f"{mode}_loss", loss, prog_bar=True, **log_kwargs)
        self.log(f"{mode}_mse",  mse,  **log_kwargs)
        self.log(f"{mode}_rmse", rmse, **log_kwargs)
        self.log(f"{mode}_mae",  mae,  **log_kwargs)
        self.log(f"{mode}_r2",   r2,   **log_kwargs)

        return loss
