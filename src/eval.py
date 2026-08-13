import os
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
from dotenv import load_dotenv
from lightning import Trainer
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.data.base_datamodule import BaseDataModule
from src.models.base_model import BaseModel
from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.experiment_tracking import compose_experiment_name

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

# Optimize Tensor Core usage (L40S / A100 / H100 all benefit from this)
torch.set_float32_matmul_precision("high")

# Disable tokenizers parallelism to avoid warnings when using multiprocessing
if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = RankedLogger(__name__, rank_zero_only=True)

OmegaConf.register_new_resolver("str", str, replace=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    if not cfg.baseline:
        assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: BaseModel = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    wandb_logger = next((log for log in logger if isinstance(log, WandbLogger)), None)
    if wandb_logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

        group = cfg.get("experiment_name", "null")
        if group == "null":
            compose_experiment_name(cfg)
        wandb_logger.log_metrics({"experiment": group})

    log.info("Starting testing!")
    trainer.test(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), weights_only=False
    )
    metric_dict = trainer.callback_metrics

    if cfg.get("validate") and wandb_logger is not None:
        # Run validation
        log.info("Validating!")

        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )

        val_metrics = trainer.callback_metrics
        wandb_logger.log_metrics({f"best_{k}": v for k, v in val_metrics.items()})

        metric_dict = {**metric_dict, **val_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
