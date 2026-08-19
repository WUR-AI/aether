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
from utils.errors import FileNotSpecified

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
    """Evaluates given checkpoint on a datamodule testset. Used to evaluate prediction or alignment
    model individually. Can also evaluate the alignment baselines from other studies.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: BaseDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    wandb_logger = next((log for log in logger if isinstance(log, WandbLogger)), None)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = (
        hydra.utils.instantiate(cfg.trainer, logger=logger)
        if cfg.trainer
        else Trainer(logger=logger)
    )

    if not cfg.get("baseline"):
        assert cfg.ckpt_path
        ckpt_path = cfg.get("ckpt_path") or FileNotSpecified(
            'You must specify model weight path as "ckpt_path"'
        )
        model_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model_hparams = model_ckpt["hyper_parameters"]
        pred_model = "prediction" in ckpt_path
        model_hparams["_target_"] = (
            "src.models.predictive_model.PredictiveModel"
            if pred_model
            else "src.models.text_alignment_model.TextAlignmentModel"
        )

        model_hparams["trainable_modules"] = None
        if not pred_model:
            model_hparams["text_encoder"]["hf_cache_dir"] = os.path.join(
                cfg.paths.cache_dir, "huggingface"
            )
        if "AverageEncoder" in model_hparams["geo_encoder"]["_target_"]:
            if "aef_avr" in cfg.data.dataset.modalities.keys():
                model_hparams["geo_encoder"].update(
                    {
                        "_target_": "src.models.components.geo_encoders.identity_encoder.IdentityEncoder",
                        "geo_data_name": "aef_avr",
                    }
                )
            elif "tessera_avr" in cfg.data.dataset.modalities.keys():
                model_hparams["geo_encoder"].update(
                    {
                        "_target_": "src.models.components.geo_encoders.identity_encoder.IdentityEncoder",
                        "geo_data_name": "tessera_avr",
                    }
                )

        if "loss_fn" not in model_hparams.keys():
            model_hparams["loss_fn"] = cfg.get("model", {}).get("loss_fn")
        if "scheduler" not in model_hparams.keys():
            model_hparams["scheduler"] = cfg.get("model", {}).get("scheduler")
        if "optimizer" not in model_hparams.keys():
            model_hparams["optimizer"] = cfg.get("model", {}).get("optimizer")
        if "metrics" not in model_hparams.keys():
            model_hparams["metrics"] = cfg.get("metrics")
        model: BaseModel = hydra.utils.instantiate(model_hparams)
        trainer.datamodule = datamodule
        model.trainer = trainer

        model.setup("test")
        model.load_state_dict(model_ckpt["state_dict"])
        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "logger": logger,
            "trainer": trainer,
        }
    else:
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: BaseModel = hydra.utils.instantiate(cfg.model)

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "logger": logger,
            "trainer": trainer,
        }

        if wandb_logger:
            log.info("Logging hyperparameters!")
            log_hyperparameters(object_dict)

            group = cfg.get("experiment_name", "null")
            if group == "null":
                compose_experiment_name(cfg)
            wandb_logger.log_metrics({"experiment": group})
    log.info("Starting testing!")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )
    metric_dict = trainer.callback_metrics

    if cfg.get("validate") and wandb_logger is not None:
        # Run validation
        log.info("Validating!")

        trainer.validate(
            model=model,
            datamodule=datamodule,
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
