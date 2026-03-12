"""Tests for the yield_africa use case.

The mock CSV mirrors the schema produced by make_model_ready_yield_africa.py:
  - name_loc, lat, lon
  - target_yld_ton_ha
  - feat_* (continuous soil/climate/terrain features + tabular categorical soil texture)
  - aux_*  (derived classification columns, used for caption generation)
  - metadata: country, year, location_accuracy
"""

import hydra
import pandas as pd
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.data.base_datamodule import BaseDataModule
from src.data.yield_africa_dataset import YieldAfricaDataset

# ---------------------------------------------------------------------------
# Representative column sets that match the real model_ready_yield-africa.csv
# ---------------------------------------------------------------------------

MOCK_FEAT_COLS = {
    # continuous soil features
    "feat_c_0_20": [1.2, 0.9, 1.5, 1.1, 0.8, 1.4, 1.6, 1.0, 1.3, 1.1],
    "feat_n_0_20": [0.12, 0.09, 0.15, 0.11, 0.08, 0.14, 0.16, 0.10, 0.13, 0.11],
    "feat_ph_0_20": [6.1, 5.8, 6.5, 6.0, 5.5, 6.3, 6.8, 5.9, 6.2, 6.4],
    # continuous climate features
    "feat_map": [820, 750, 910, 860, 700, 880, 930, 770, 815, 875],
    "feat_mat": [22.1, 21.5, 23.0, 22.5, 21.0, 22.8, 23.3, 21.9, 22.2, 22.7],
    # continuous terrain feature
    "feat_dem": [450, 380, 510, 470, 360, 490, 530, 400, 460, 500],
    # tabular categorical: soil texture class (real columns, not derived)
    "feat_tx_0_20_cl": [2, 3, 1, 2, 4, 1, 3, 2, 1, 3],
    "feat_tx_20_50_cl": [2, 2, 1, 3, 4, 1, 2, 3, 1, 2],
}

MOCK_AUX_COLS = {
    # derived classification columns (paired with the continuous feat_* above)
    "aux_yld_ton_ha_cl": [1, 0, 2, 1, 0, 2, 2, 0, 1, 1],
    "aux_c_0_20_cl": [1, 0, 2, 1, 0, 2, 2, 0, 1, 1],
    "aux_ph_0_20_cl": [2, 1, 2, 2, 0, 2, 3, 1, 2, 2],
    "aux_map_cl": [1, 0, 2, 1, 0, 2, 2, 0, 1, 2],
}

MOCK_N_ROWS = 10
# feat_year (1) + feat_country_{code} (8) are injected by YieldAfricaDataset
# when country and year columns are present, so the effective tabular dim grows.
from src.data.yield_africa_dataset import _ALL_COUNTRIES

MOCK_INJECTED_FEAT_NAMES = {"feat_year"} | {f"feat_country_{c}" for c in _ALL_COUNTRIES}
MOCK_TABULAR_DIM = len(MOCK_FEAT_COLS) + len(MOCK_INJECTED_FEAT_NAMES)  # 8 + 9 = 17
MOCK_N_AUX = len(MOCK_AUX_COLS)  # 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def yield_africa_csv(tmp_path) -> str:
    """Mock CSV with column names matching the real model_ready_yield-africa.csv."""
    data = {
        "name_loc": [f"ETH_{i:04d}" for i in range(MOCK_N_ROWS)],
        "lat": [5.0 + i * 0.5 for i in range(MOCK_N_ROWS)],
        "lon": [30.0 + i * 0.5 for i in range(MOCK_N_ROWS)],
        "target_yld_ton_ha": [2.1, 1.8, 3.0, 2.5, 1.2, 2.8, 3.3, 1.9, 2.0, 2.7],
        "country": ["ETH"] * MOCK_N_ROWS,
        "year": [2019] * MOCK_N_ROWS,
        "location_accuracy": [1] * MOCK_N_ROWS,
    }
    data.update(MOCK_FEAT_COLS)
    data.update(MOCK_AUX_COLS)

    mock_dir = tmp_path / "mock"
    mock_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(mock_dir / "model_ready_mock.csv", index=False)
    return str(tmp_path)


@pytest.fixture
def yield_africa_dataset(yield_africa_csv, tmp_path):
    """YieldAfricaDataset backed by mock data, features enabled, no aux."""
    return YieldAfricaDataset(
        data_dir=yield_africa_csv,
        cache_dir=str(tmp_path / "cache"),
        modalities={"coords": {}},
        use_target_data=True,
        use_aux_data="none",
        seed=42,
        mock=True,
        use_features=True,
    )


@pytest.fixture
def yield_africa_dataset_with_aux(yield_africa_csv, tmp_path):
    """YieldAfricaDataset backed by mock data, features enabled, aux enabled."""
    return YieldAfricaDataset(
        data_dir=yield_africa_csv,
        cache_dir=str(tmp_path / "cache"),
        modalities={"coords": {}},
        use_target_data=True,
        use_aux_data="all",
        seed=42,
        mock=True,
        use_features=True,
    )


@pytest.fixture
def yield_africa_datamodule(yield_africa_csv, tmp_path):
    """BaseDataModule wrapping YieldAfricaDataset with a random split."""
    dataset = YieldAfricaDataset(
        data_dir=yield_africa_csv,
        cache_dir=str(tmp_path / "cache"),
        modalities={"coords": {}},
        use_target_data=True,
        use_aux_data="none",
        seed=42,
        mock=True,
        use_features=True,
    )
    return BaseDataModule(
        dataset=dataset,
        batch_size=4,
        train_val_test_split=(7, 2, 1),
        num_workers=0,
        pin_memory=False,
        split_mode="random",
        save_split=False,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


def test_yield_africa_dataset_length(yield_africa_dataset):
    assert len(yield_africa_dataset) == MOCK_N_ROWS


def test_yield_africa_dataset_sample_keys(yield_africa_dataset):
    sample = yield_africa_dataset[0]
    assert "eo" in sample
    assert "coords" in sample["eo"]
    assert "tabular" in sample["eo"]
    assert "target" in sample


def test_yield_africa_dataset_sample_shapes(yield_africa_dataset):
    sample = yield_africa_dataset[0]
    assert sample["eo"]["coords"].shape == (2,)
    assert sample["eo"]["tabular"].shape == (MOCK_TABULAR_DIM,)
    assert sample["target"].shape == (1,)


def test_yield_africa_dataset_sample_dtypes(yield_africa_dataset):
    sample = yield_africa_dataset[0]
    assert sample["eo"]["coords"].dtype == torch.float32
    assert sample["eo"]["tabular"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32


def test_yield_africa_dataset_target_name(yield_africa_dataset):
    assert yield_africa_dataset.target_names == ["target_yld_ton_ha"]


def test_yield_africa_dataset_attributes(yield_africa_dataset):
    assert yield_africa_dataset.num_classes == 1
    assert yield_africa_dataset.tabular_dim == MOCK_TABULAR_DIM
    expected_feat_names = set(MOCK_FEAT_COLS.keys()) | MOCK_INJECTED_FEAT_NAMES
    assert set(yield_africa_dataset.feat_names) == expected_feat_names


def test_yield_africa_dataset_feat_prefix(yield_africa_dataset):
    """All tabular features must carry the feat prefix."""
    for name in yield_africa_dataset.feat_names:
        assert name.startswith("feat_"), f"Unexpected feature name: {name}"


def test_yield_africa_dataset_coords_values(yield_africa_dataset):
    """Coordinates returned must match the CSV values."""
    sample = yield_africa_dataset[0]
    coords = sample["eo"]["coords"]
    assert coords[0].item() == pytest.approx(5.0)  # lat of row 0
    assert coords[1].item() == pytest.approx(30.0)  # lon of row 0


def test_yield_africa_dataset_target_values(yield_africa_dataset):
    """Target values returned must match the CSV values."""
    expected = [2.1, 1.8, 3.0, 2.5, 1.2, 2.8, 3.3, 1.9, 2.0, 2.7]
    for idx, exp in enumerate(expected):
        sample = yield_africa_dataset[idx]
        assert sample["target"][0].item() == pytest.approx(exp, rel=1e-5)


def test_yield_africa_dataset_no_features(yield_africa_csv, tmp_path):
    """With use_features=False, tabular is absent and tabular_dim is None."""
    ds = YieldAfricaDataset(
        data_dir=yield_africa_csv,
        cache_dir=str(tmp_path / "cache"),
        modalities={"coords": {}},
        use_target_data=True,
        use_aux_data="none",
        seed=0,
        mock=True,
        use_features=False,
    )
    sample = ds[0]
    assert "tabular" not in sample["eo"]
    assert ds.tabular_dim is None


def test_yield_africa_dataset_aux_keys(yield_africa_dataset_with_aux):
    """When aux is enabled, sample must contain an 'aux' dict."""
    sample = yield_africa_dataset_with_aux[0]
    assert "aux" in sample
    assert "aux" in sample["aux"]


def test_yield_africa_dataset_aux_columns(yield_africa_dataset_with_aux):
    """Aux columns picked up must match the aux_* columns in the mock CSV."""
    resolved_aux = yield_africa_dataset_with_aux.use_aux_data["aux"]
    assert set(resolved_aux) == set(MOCK_AUX_COLS.keys())


def test_yield_africa_dataset_aux_shape(yield_africa_dataset_with_aux):
    """Aux tensor shape must equal the number of aux_* columns."""
    sample = yield_africa_dataset_with_aux[0]
    assert sample["aux"]["aux"].shape == (MOCK_N_AUX,)


# ---------------------------------------------------------------------------
# Datamodule tests
# ---------------------------------------------------------------------------


def test_yield_africa_datamodule_split_sizes(yield_africa_datamodule):
    dm = yield_africa_datamodule
    assert len(dm.data_train) == 7
    assert len(dm.data_val) == 2
    assert len(dm.data_test) == 1


def test_yield_africa_datamodule_train_loader(yield_africa_datamodule):
    dm = yield_africa_datamodule
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert "eo" in batch
    assert "coords" in batch["eo"]
    assert "tabular" in batch["eo"]
    assert batch["eo"]["coords"].shape == (4, 2)
    assert batch["eo"]["tabular"].shape == (4, MOCK_TABULAR_DIM)
    assert batch["target"].shape == (4, 1)


def test_yield_africa_datamodule_split_deterministic(yield_africa_csv, tmp_path):
    def make_dm():
        dataset = YieldAfricaDataset(
            data_dir=yield_africa_csv,
            cache_dir=str(tmp_path / "cache"),
            modalities={"coords": {}},
            use_target_data=True,
            use_aux_data="none",
            seed=42,
            mock=True,
        )
        return BaseDataModule(
            dataset=dataset,
            batch_size=4,
            train_val_test_split=(7, 2, 1),
            num_workers=0,
            split_mode="random",
            save_split=False,
            seed=42,
        )

    dm1, dm2 = make_dm(), make_dm()
    assert dm1.data_train.indices == dm2.data_train.indices
    assert dm1.data_val.indices == dm2.data_val.indices


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_yield_africa_config_loads():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=yield_africa_tabular_reg", "hydra.job.chdir=false"],
        )
    assert cfg.data._target_ == "src.data.base_datamodule.BaseDataModule"
    assert cfg.data.dataset._target_ == "src.data.yield_africa_dataset.YieldAfricaDataset"
    assert cfg.model._target_ == "src.models.predictive_model.PredictiveModel"
    GlobalHydra.instance().clear()


def test_yield_africa_model_instantiates():
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["experiment=yield_africa_tabular_reg", "hydra.job.chdir=false"],
        )
    model = hydra.utils.instantiate(cfg.model)
    assert model is not None
    GlobalHydra.instance().clear()
