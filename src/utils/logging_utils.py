from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]
    # explicitly surface optimizer and scheduler configs so loggers (e.g. wandb)
    # display them alongside the model definition
    hparams["optimizer"] = cfg["model"].get("optimizer")
    hparams["scheduler"] = cfg["model"].get("scheduler")

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def _group_keys(keys: list[str]) -> dict[str, list[str]]:
    """Groups module names (keys)"""
    grouped: dict[str, list[str]] = {}
    for k in keys:
        top = k.split(".", 1)[0] if "." in k else k
        grouped.setdefault(top, []).append(k)
    return grouped


def log_model_loading(tag: str, result) -> None:
    """Log missing/unexpected keys from `load_state_dict`."""
    missing_keys, unexpected_keys = result
    if missing_keys:
        grouped = _group_keys(list(missing_keys))
        summary = {k: len(v) for k, v in grouped.items()}
        log.warning(f"[{tag}] Missing keys: {summary}")
    if unexpected_keys:
        grouped = _group_keys(list(unexpected_keys))
        summary = {k: len(v) for k, v in grouped.items()}
        log.warning(f"[{tag}] Unexpected keys: {summary}")
    if not missing_keys and not unexpected_keys:
        log.info(f"[{tag}] Module weights loaded successfully")
