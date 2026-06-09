from abc import ABC, abstractmethod
from typing import Any, Dict, final

import torch
from lightning import LightningModule

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.metrics.metrics_wrapper import MetricsWrapper
from src.models.components.pred_heads.base_pred_head import BasePredictionHead
from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder
from src.utils.logging_utils import log_model_loading


class BaseModel(LightningModule, ABC):
    def __init__(
        self,
        trainable_modules: list[str],
        geo_encoder: BaseGeoEncoder | None,
        text_encoder: BaseTextEncoder | None,
        prediction_head: BasePredictionHead | None,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        loss_fn: BaseLossFn | None,
        metrics: MetricsWrapper | None,
        num_classes: int | None,
        tabular_dim: int | None,
    ) -> None:
        """Interface for any model.

        :param trainable_modules: which modules to train
        :param geo_encoder: module for encoding geo data
        :param text_encoder: module for encoding text data
        :param prediction_head: module for making prediction from geo features
        :param optimizer: optimizer for the model weight update
        :param scheduler: scheduler for the model weight update
        :param loss_fn: loss function
        :param metrics: metrics to track for model performance estimation
        :param num_classes: number of target classes
        :param tabular_dim: number of tabular features
        """
        super().__init__()

        # Ignore objects
        self.save_hyperparameters(
            ignore=[
                "geo_encoder",
                "text_encoder",
                "prediction_head",
                "optimizer",
                "scheduler",
                "loss_fn",
                "metrics",
            ]
        )

        self.trainable_modules = trainable_modules
        if geo_encoder:
            self.geo_encoder = geo_encoder
        if text_encoder:
            self.text_encoder = text_encoder
        if prediction_head:
            self.prediction_head = prediction_head

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.loss_fn = loss_fn
        self.metrics = metrics

        self.num_classes = num_classes
        self.tabular_dim = tabular_dim

        self.setup_flag = False

    @final
    def setup(self, stage: str) -> None:
        """Updates model based data-bound configurations (through datamodule), This method is
        called after trainer is initialized and datamodule is available."""
        if self.setup_flag:
            print(f"Model {self.__str__()} is already set up!")
            return

        # If trainer is attached get num_classes and tabular_dim from datamodule (data-dependent)
        if self._trainer is not None:
            self.num_classes = self.trainer.datamodule.num_classes
            self.tabular_dim = self.trainer.datamodule.tabular_dim

        # Per model logic of setting up
        self._setup(stage)
        self.setup_flag = True

        # Freezing requested parts
        if stage in ["inference", "test"]:
            self.full_freezer()
            self.trainable_modules = []
        else:
            self.freezer()

    @abstractmethod
    def _setup(self, stage: str) -> None:
        pass

    @final
    def full_freezer(self):
        """Freeze the whole network."""
        for name, param in self.named_parameters():
            param.requires_grad = False

        for name, module in self.named_modules():
            module.eval()

        return

    @final
    def freezer(self) -> None:
        """Freezes modules based on provided trainable modules."""
        # Convert for checking with .startswith()
        trainable_set = tuple(set(self.trainable_modules)) or tuple()
        expanded_trainable = set()

        # Freeze modules
        for name, param in self.named_parameters():
            # Unfreeze trainable parts
            if name.startswith(trainable_set):
                param.requires_grad = True
                expanded_trainable.add(name)
            else:
                # Freeze the rest
                param.requires_grad = False

        # Set module modes correctly.
        # A module should be in train() if:
        #   - it IS a trainable module (name == t), or
        #   - it is a CHILD of a trainable module (name starts with t + "."), or
        #   - it is an ANCESTOR of a trainable module (t starts with name + "."),
        #     so that container modules reflect the correct mode, or
        #   - it is the root module (""), which must be train when any child is.
        def _in_train_scope(name: str) -> bool:
            if not name:  # root module
                return bool(trainable_set)
            for t in trainable_set:
                if name == t or name.startswith(t + ".") or t.startswith(name + "."):
                    return True
            return False

        for name, module in self.named_modules():
            if _in_train_scope(name):
                module.train()
            else:
                module.eval()

        print("------Set to train------")
        for m in sorted(expanded_trainable):
            print(f"  {m}")
        print("------------------------")
        self.trainable_modules = list(expanded_trainable)

    @abstractmethod
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward computation of the model."""
        pass

    @abstractmethod
    def _step(
        self,
        batch: Dict[str, torch.Tensor],
        mode: str = "train",
    ) -> torch.Tensor:
        """Step forward computation of the model."""
        pass

    @final
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.log(
            "lr",
            self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0],
            prog_bar=False,
            on_step=True,
            sync_dist=True,
        )
        return self._step(batch, "train")

    @final
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    @final
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    @final
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""

        optimizer = self.optimizer(params=self.trainer.model.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def update_configs(self, cfg):
        """Update hyper-parameters from the model."""
        if hasattr(self, "geo_encoder"):
            self.geo_encoder.update_configs(cfg["geo_encoder"])

        if hasattr(self, "text_encoder"):
            self.text_encoder.cfg_dict = cfg["text_encoder"]

        if hasattr(self, "prediction_head"):
            self.prediction_head.cfg_dict = cfg["prediction_head"]

    def on_save_checkpoint(self, checkpoint):
        """Save checkpoint.

        - Save only trainable parts of the model.
        - Append configurations of the model
        """
        if not self.setup_flag:
            raise ValueError("Model cannot be saved as it was not set up.")

        # Remove unnecessary keys
        pop_list = [
            "state_dict",
            "loops",
            "hparams_name",
            "datamodule_hyper_parameters",
            "datamodule_hparams_name",
        ]
        for i in pop_list:
            checkpoint.pop(i)

        # Save only trainable parts
        checkpoint["state_dict"] = {
            k: v
            for k, v in self.state_dict().items()
            if any(k.startswith(part) for part in self.trainable_modules)
        }

        # Update model configurations
        checkpoint["hyper_parameters"].update(
            {
                "num_classes": self.num_classes,
                "tabular_dim": self.tabular_dim,
                "trainable_modules": self.trainable_modules,
            }
        )

        if hasattr(self, "geo_encoder"):
            checkpoint["hyper_parameters"]["geo_encoder"] = self.geo_encoder.cfg_dict
        if hasattr(self, "prediction_head"):
            checkpoint["hyper_parameters"]["prediction_head"] = self.prediction_head.cfg_dict
        if hasattr(self, "text_encoder"):
            checkpoint["hyper_parameters"]["text_encoder"] = self.text_encoder.cfg_dict

        return

    def on_load_checkpoint(self, checkpoint):
        """Load pre-trained parts of the model."""
        res = self.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Model loaded from a checkpoint.")
        log_model_loading("Model from checkpoint", res)

    # TODO feels illegal
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)
