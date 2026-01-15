import json

import pandas as pd
import pytest
import torch

from src.data.base_caption_builder import BaseCaptionBuilder, DummyCaptionBuilder
from src.data.base_datamodule import BaseDataModule
from src.data.butterfly_dataset import ButterflyDataset

def test_base_datamodule_random_split_and_loaders(create_butterfly_dataset):
    dataset, dm = create_butterfly_dataset

    assert len(dm.data_train) == 4
    assert len(dm.data_val) == 1
    assert len(dm.data_test) == 1

    batch = next(iter(dm.train_dataloader()))
    assert batch["eo"]["coords"].shape == (2, 2)
    assert batch["target"].shape == (2, 2)


def test_random_split_is_deterministic(create_butterfly_dataset):
    dataset1, dm1 = create_butterfly_dataset
    dataset2, dm2 = create_butterfly_dataset

    assert dm1.data_train.indices == dm2.data_train.indices
    assert dm1.data_val.indices == dm2.data_val.indices
    assert dm1.data_test.indices == dm2.data_test.indices

def test_datamodule_uses_collate_when_aux_data(sample_csv, tmp_path):
    templates_path = tmp_path / "templates.json"
    templates_path.write_text(json.dumps(["<name_loc> text"]))
    caption_builder = DummyCaptionBuilder(str(templates_path), data_dir=str(tmp_path), seed=0)

    dataset = ButterflyDataset(
        path_csv=sample_csv,
        modalities=["coords"],
        use_target_data=True,
        use_aux_data=True,
        seed=0,
    )

    dm = BaseDataModule(
        dataset,
        batch_size=2,
        train_val_test_split=(4, 2, 0),
        split_mode="random",
        caption_builder=caption_builder,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "text" in batch
    assert len(batch["text"]) == dm.batch_size_per_device
