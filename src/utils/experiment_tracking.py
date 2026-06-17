import os
import re
from pathlib import Path

import pandas as pd
import wandb
from omegaconf import DictConfig

# Which wandb projects to log runs from
PROJECTS = ["s2bms_prediction", "s2bms_alignment"]  # Important to keep updating!
EXPERIMENT_TRACKER_NAME = "experiment_tracker.csv"
ENTITY = "aether_xai"


def parse_data_name(run=None, cfg=None):
    if cfg is None:
        assert run is not None
        data_dict = run.config["data"]["dataset"]["modalities"]
    else:
        data_dict = cfg["data"]["dataset"]["modalities"]

    if len(data_dict) == 1:
        k = list(data_dict.keys())[0]
        if k == "coords":
            if "GeoClip" in run.config["model"]["geo_encoder"]["_target_"]:
                data_name = "geoclip"
            else:
                data_name = "satclip"
        else:
            ks = list(data_dict.keys())
            ks_new = [
                f"{k}{f'_{k.get('size')}' if isinstance(k, dict) and k.get('size') else ''}"
                for k in ks
            ]
            data_name = "-".join(map(str, ks_new))

        return data_name


def compose_experiment_name(cfg: DictConfig) -> str:
    """For alignment
    (unlab_)(geo_encoder)_(eo_mod)_(geo_adapter)_(text_encoder)_(text_adapter)_(batch_size) # noqa:

    RST306.
    """

    name = ""

    # Unlabeled
    if cfg.data.dataset.use_unlabelled_data:
        name += "unlab_"  # noqa: RST306

    # geo encoder
    name += cfg.model.geo_encoder._target_.split(".")[-1]

    # eo mod
    name += "_" + parse_data_name(cfg=cfg)

    # batch size
    if cfg.model._target_ == "src.models.text_alignment_model.TextAlignmentModel":
        # geo adapter
        if cfg.model.get("geo_adapter") is not None:
            name += "_" + cfg.model.geo_adapter._target_.split(".")[-1]

        # text encoder
        if cfg.model.get("text_adapter") is not None:
            name += "_" + cfg.model.text_encoder._target_.split(".")[-1]

        # text adapter
        if cfg.model.get("text_adapter") is not None:
            name += "_" + cfg.model.text_adapter._target_.split(".")[-1]

        # batch size
        name += f"_b-{cfg.data.batch_size}"
    else:
        # prediction head
        if cfg.model.get("prediction_head") is not None:
            name += "_" + cfg.model.prediction_head._target_.split(".")[-1]

    # extras for experiment identification
    if cfg.get("experiment_name_extra"):
        name += "_" + cfg.experiment_name_extra

    return name


def update_wandb_run_id(run_id, project_name, updates_dict):
    api = wandb.Api()
    entity = ENTITY

    run = api.run(f"{entity}/{project_name}/{run_id}")

    # Update the config
    for key, value in updates_dict.items():
        run.config[key] = value

    run.config.save()
    print(f"Successfully updated run {run_id}")


def experiment_check(cfg: DictConfig):
    """Check if requested through configs experiments has already been run."""
    # Get experiments from wandb (cross-device sync)
    df = get_experiments_from_wandb(cfg)

    # If there are none, return to training
    if df is None:
        return False

    # Filter for experiment and see all trained seeds
    if cfg.get("experiment_name"):
        df = df[df["experiment"] == cfg["experiment_name"]]

    # If seed is not trained, return to training
    if len(df) == 0 or (cfg.seed not in list(df.seed)):
        return False

    print(f"Experiment {cfg.tags[-1]} has been already trained with seed: {cfg.seed}.")
    return True


def get_experiments_from_wandb(cfg: DictConfig) -> pd.DataFrame | None:
    """Get all experiments from wandb projects."""

    # Get existing experiment table
    df, df_path = open_experiment_df(cfg)
    ids = list(df.run_id) if len(df) > 0 else []
    df_len = len(df)

    # Connect to wandb api
    api = wandb.Api()
    entity = (
        cfg.logger.wandb.entity if cfg.get("logger", {}).get("wandb", {}).get("entity") else ENTITY
    )
    projects = cfg.get("projects") or PROJECTS
    if isinstance(projects, str):
        projects = [projects]

    # Find all experiments on wandb and add missing/faulty ones in the table
    runs_list = []
    for project in projects:
        print(f"Fetching runs from {entity}/{project}...")
        runs_iterator = api.runs(f"{entity}/{project}")
        for run in runs_iterator:
            # Look for missing experiments
            if run.state != "finished":
                continue

            if run.id not in ids:
                runs_list.append(run)

                # Get all values for logging
                task = project
                run_id = run.id
                seed = run.config["seed"]
                best_path = run.summary.get("best_model_path", "null")
                best_epoch = float(re.search(r".*_([0-9]{3})\.ckpt$", best_path).group(1))
                source_dir = run.summary.get("source_dir", "null")
                best_val_loss = run.config.get("best_val_loss", None)
                if project == "s2bms_prediction":
                    best_val_mse_loss = run.config.get("best_val_mse_loss", None)

                # Get the best loss based on tracked history
                results = run.history(pandas=True)
                if best_val_loss is None:
                    if len(results) > 0:
                        loss_per_epoch = results.groupby("epoch")["val_loss"].mean()
                        if best_epoch:
                            best_val_loss = loss_per_epoch[best_epoch]
                        else:
                            best_val_loss = loss_per_epoch.min().item()
                        run.summary.update({"best_val_loss": best_val_loss})
                if project == "s2bms_prediction" and best_val_mse_loss is None:
                    mse_loss_per_epoch = results.groupby("epoch")["val_mse_loss"].mean()
                    best_val_mse_loss = mse_loss_per_epoch[best_epoch].item()
                    run.summary.update({"best_val_mse_loss": best_val_mse_loss})

                experiment = run.summary.get("experiment")
                if experiment is None:
                    experiment = run.config["experiment_name"]
                    run.summary.update({"experiment": experiment})
                    data_name = parse_data_name(run=run)
                    run.summary.update({"data_used": data_name})

                # Add missing experiments to the table
                df = update_experiment_df(
                    run_id=run_id,
                    best_metric=best_val_loss,
                    best_path=best_path,
                    source_dir=source_dir,
                    experiment=experiment,
                    task=task,
                    seed=seed,
                    df=df,
                    df_path=df_path,
                )

    # Return experiment df
    if df_len - len(df) > 0:
        print(f"Saved {len(runs_list)} runs to {df_path}.")
        return df
    if len(df) > 0:
        print("No runs updated.")
        return df
    else:
        print("No runs found.")
        return None


def open_experiment_df(cfg: DictConfig) -> tuple[pd.DataFrame, str]:
    """Get existing experiment table."""
    df_path = os.path.join(cfg.paths.log_dir, EXPERIMENT_TRACKER_NAME)
    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame()

    return df, df_path


def update_experiment_df(
    run_id,
    best_metric,
    best_path,
    source_dir,
    experiment,
    task,
    seed,
    cfg=None,
    df=None,
    df_path=None,
):
    """Update experiment tracker log table."""
    if cfg is None:
        assert df is not None and df_path is not None, "Must provide configurations or dataframe."
    elif df is None and df_path is None:
        # Get table of executed experiments
        df, df_path = open_experiment_df(cfg)

    # Compile information about the experiment
    row = {
        "run_id": run_id,
        "task": task,
        "seed": seed,
        "experiment": experiment,
        "best_metric": best_metric,
        "path": best_path,
        "source_dir": source_dir,
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Log it to the table
    df.to_csv(df_path, index=False)
    print(f"Saved logs to {df_path}.")

    return df


def clean_local_ckpts(cfg: DictConfig, df: pd.DataFrame) -> None:
    """Deletes all local checkpoint files that are not logged in csv file (associated with wandb)
    or are not specified as exceptions."""

    # Create a keep set
    keep = set()
    exceptions = cfg.get("exceptions")
    exceptions = set(exceptions) if exceptions is not None else set()
    keep.update(exceptions)

    # Add paths from wandb csv log
    local_ckpt_root = cfg.paths.checkpoint_root
    if df is not None:
        for r in df.itertuples():
            p = os.path.join(local_ckpt_root, r.task + "_ckpt", r.path)
            if os.path.exists(p):
                keep.add(p)
            elif os.path.dirname(r.source_dir) != local_ckpt_root:
                print(
                    f"{p} is not on this device. Copy it from {r.source_dir} to {cfg.paths.checkpoint_dir}."
                )
            else:
                print(f"{p} is missing from {local_ckpt_root}.")

    # Get locally available checkpoints
    all_ckpts = list(Path(local_ckpt_root).rglob("*.ckpt"))

    # Delete non-keep ckpts (with user confirmation)
    for local_ckpt in all_ckpts:
        if str(local_ckpt) not in keep:
            print(f"Do you want to remove {local_ckpt}? (y/n)")
            answer = input()
            if answer == "y":
                os.remove(str(local_ckpt))
                print(f"Removed {local_ckpt}.")
