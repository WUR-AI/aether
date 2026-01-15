"""This file prepares config fixtures for other tests."""

from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict
import pandas as pd
import torch
from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.base_datamodule import BaseDataModule
from src.data.butterfly_dataset import ButterflyDataset


def pytest_addoption(parser):
    parser.addoption("--use-mock", action="store_true", help="Use mock data instead of real data")

@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """A pytest fixture for setting up a default Hydra DictConfig for training.

    :return: A DictConfig object containing a default Hydra configuration for training.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, 
                      overrides=["data=butterfly_coords", "hydra.job.chdir=false"])

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.1
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.trainer.strategy = "single_device"
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


# @pytest.fixture(scope="package")
# def cfg_eval_global() -> DictConfig:
#     """A pytest fixture for setting up a default Hydra DictConfig for evaluation.

#     :return: A DictConfig containing a default Hydra configuration for evaluation.
#     """
#     with initialize(version_base="1.3", config_path="../configs"):
#         cfg = compose(
#             config_name="eval.yaml",
#             return_hydra_config=True,
#             overrides=["ckpt_path=."],
#         )

#         # set defaults for all tests
#         with open_dict(cfg):
#             cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
#             cfg.trainer.max_epochs = 1
#             cfg.trainer.limit_test_batches = 0.1
#             cfg.trainer.accelerator = "cpu"
#             cfg.trainer.devices = 1
#             cfg.data.num_workers = 0
#             cfg.data.pin_memory = False
#             cfg.extras.print_config = False
#             cfg.extras.enforce_tags = False
#             cfg.logger = None

#     return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    """A pytest fixture built on top of the `cfg_train_global()` fixture, which accepts a temporary
    logging path `tmp_path` for generating a temporary logging path.

    This is called by each test which uses the `cfg_train` arg. Each test generates its own temporary logging path.

    :param cfg_train_global: The input DictConfig object to be modified.
    :param tmp_path: The temporary logging path.

    :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
    """
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


# @pytest.fixture(scope="function")
# def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
#     """A pytest fixture built on top of the `cfg_eval_global()` fixture, which accepts a temporary
#     logging path `tmp_path` for generating a temporary logging path.

#     This is called by each test which uses the `cfg_eval` arg. Each test generates its own temporary logging path.

#     :param cfg_train_global: The input DictConfig object to be modified.
#     :param tmp_path: The temporary logging path.

#     :return: A DictConfig with updated output and log directories corresponding to `tmp_path`.
#     """
#     cfg = cfg_eval_global.copy()

#     with open_dict(cfg):
#         cfg.paths.output_dir = str(tmp_path)
#         cfg.paths.log_dir = str(tmp_path)

#     yield cfg

#     GlobalHydra.instance().clear()

@pytest.fixture
def sample_csv(tmp_path) -> str:
    df = pd.DataFrame(
        {
            "name_loc": [f"loc_{i}" for i in range(6)],
            "lat": [50.0, 50.5, 51.0, 51.5, 52.0, 52.5],
            "lon": [4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            "target_a": [1, 0, 1, 0, 1, 0],
            "target_b": [0, 1, 0, 1, 0, 1],
            "aux_temp": [10, 11, 12, 13, 14, 15],
        }
    )
    path = tmp_path / "butterflies.csv"
    df.to_csv(path, index=False)
    return str(path)

@pytest.fixture()
def create_butterfly_dataset(request, sample_csv):
    """A pytest fixture for creating a ButterflyDataset instance."""
    use_mock = request.config.getoption("--use-mock")
    if use_mock:
        path_csv = sample_csv
    else:
        assert False, "Real data not available in test environment."
    dataset = ButterflyDataset(
        path_csv=path_csv,
        modalities=["coords"],
        use_target_data=True,
        use_aux_data=False,
        seed=0,
    )

    dm = BaseDataModule(
        dataset,
        batch_size=2,
        train_val_test_split=(4, 1, 1),
        num_workers=0,
        pin_memory=False,
        split_mode="random",
        save_split=False,
    )

    return (dataset, dm)