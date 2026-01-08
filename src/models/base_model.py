from abc import ABC, abstractmethod
from typing import Any, final, Dict

import torch
from lightning import LightningModule

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class BaseModel(LightningModule, ABC):
    def __init__(
            self,
            trainable_modules: list[str] | None,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_fn: BaseLossFn,
            num_classes: int | None = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'eo_encoder', 'prediction_head', 'text_encoder'])

        self.trainable_modules = tuple(trainable_modules) or tuple()
        self.num_classes: int = num_classes

        # Loss
        self.loss_fn = loss_fn

    @final
    def freezer(self) -> None:
        """Freezes and unfreezes modules based on freezing strategy and freezing exceptions"""

        trainable = set()
        # Freeze modules
        for name, param in self.named_parameters():
            # Enable exceptions
            if name.startswith(self.trainable_modules):
                param.requires_grad = True
                top_name = name.split(".", 2)[:2]
                trainable.add('.'.join(top_name))
            else:
                # Freeze the rest
                param.requires_grad = False

        # Set module modes correctly
        for name, module in self.named_modules():
            if any(t.startswith(name) for t in self.trainable_modules):
                module.train()
            else:
                module.eval()

        print('----------------------------')
        print(f'Set to train')
        for m in sorted(trainable):
            print(f"  {m}")
        print('----------------------------')

    @abstractmethod
    def forward(
            self,
            batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def _step(
            self,
            batch: Dict[str, torch.Tensor],
            mode: str='train',
    ) -> torch.Tensor:
        pass

    @final
    def training_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, 'train')

    @final
    def validation_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, 'val')

    @final
    def test_step(
            self,
            batch: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, 'test')

    @final
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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

    def on_save_checkpoint(self, checkpoint):
        """Save only trainable parts of the model"""
        checkpoint['state_dict'] = {
            k: v for k, v in self.state_dict().items()
            if any(k.startswith(part) for part in self.trainable_modules)
        }

    def on_load_checkpoint(self, checkpoint):
        """Load only trainable parts of the model"""
        missing_keys, unexpected_keys =   self.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f'Model loaded from a checkpoint.')

        if missing_keys:
            missing_keys = set(['.'.join(i.split('.')[:3]) for i in missing_keys])
            print(f"The following keys are missing from the pretrained model: {missing_keys}")
        if unexpected_keys:
            unexpected_keys = set(['.'.join(i.split('.')[:3]) for i in unexpected_keys])
            print(f"The following keys are unexpected from the pretrained model:{unexpected_keys}")

    # TODO feels illegal
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)