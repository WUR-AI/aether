import json

import pandas as pd
import pytest
import torch

from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.base_datamodule import BaseDataModule
from src.data.butterfly_dataset import ButterflyDataset


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


class DummyCaptionBuilder(BaseCaptionBuilder):
    def __init__(self, templates_path: str, data_dir: str):
        super().__init__(templates_path, data_dir)

    def sync_with_dataset(self, dataset) -> None:
        self.dataset = dataset

    def _build_from_template(self, template_idx: int, row: torch.Tensor) -> str:
        first_val = row[0].item() if torch.is_tensor(row) else row[0]
        return f"aux-{first_val}"


def test_base_datamodule_random_split_and_loaders(sample_csv):
    dataset = ButterflyDataset(
        path_csv=sample_csv,
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

    assert len(dm.data_train) == 4
    assert len(dm.data_val) == 1
    assert len(dm.data_test) == 1

    batch = next(iter(dm.train_dataloader()))
    assert batch["eo"]["coords"].shape == (2, 2)
    assert batch["target"].shape == (2, 2)


def test_random_split_is_deterministic(sample_csv):
    kwargs = dict(
        modalities=["coords"],
        use_target_data=True,
        use_aux_data=False,
        random_state=0,
    )
    ds1 = ButterflyDataset(path_csv=sample_csv, **kwargs)
    ds2 = ButterflyDataset(path_csv=sample_csv, **kwargs)

    dm1 = BaseDataModule(ds1, batch_size=2, train_val_test_split=(4, 1, 1), split_mode="random")
    dm2 = BaseDataModule(ds2, batch_size=2, train_val_test_split=(4, 1, 1), split_mode="random")

    assert dm1.data_train.indices == dm2.data_train.indices
    assert dm1.data_val.indices == dm2.data_val.indices
    assert dm1.data_test.indices == dm2.data_test.indices


def test_datamodule_uses_collate_when_aux_data(sample_csv, tmp_path):
    templates_path = tmp_path / "templates.json"
    templates_path.write_text(json.dumps(["<name_loc> text"]))
    caption_builder = DummyCaptionBuilder(str(templates_path), data_dir=str(tmp_path))

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
