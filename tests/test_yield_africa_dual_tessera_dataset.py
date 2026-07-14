"""Tests for dual-year TESSERA loading in YieldAfricaDataset.

All data (CSV + .npy tiles) is created synthetically via tmp_path — no
GeoTessera service access required.  Run with: pytest --use-mock

Mock layout
-----------
Three records are created:

  LOC_A  (lat=1.0, lon=1.0, year=2019)  — serves as the prev-year source for LOC_B
  LOC_B  (lat=1.0, lon=1.0, year=2020)  — same physical location; prev-year = LOC_A
  LOC_C  (lat=2.0, lon=2.0, year=2020)  — unique location; no prev-year tile on disk

Tiles written to {mock_dir}/eo/tessera/:
  tessera_LOC_A_2019.npy  shape [TILE_SIZE, TILE_SIZE, 128]
  tessera_LOC_B_2020.npy
  tessera_LOC_C_2020.npy
  (no tile for lat=2.0, lon=2.0, year=2019)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.yield_africa_dataset import YieldAfricaDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILE_SIZE = 3
TILE_CHANNELS = 128  # fixed by the geotessera model

DUAL_MODALITIES = {
    "tessera": {"size": TILE_SIZE, "format": "npy"},
    "tessera_prev": {"size": TILE_SIZE, "format": "npy"},
}
SINGLE_MODALITIES = {
    "tessera": {"size": TILE_SIZE, "format": "npy"},
}

# Minimal feature columns (must start with feat_)
_FEAT_COLS = {
    "feat_map": [820, 750, 700],
    "feat_mat": [22.1, 21.5, 21.0],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tile(rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a synthetic tessera tile with shape [H, W, C] = [TILE_SIZE, TILE_SIZE, 128]."""
    gen = rng or np.random.default_rng(0)
    return gen.random((TILE_SIZE, TILE_SIZE, TILE_CHANNELS)).astype(np.float32)


def _write_mock_dir(tmp_path: Path) -> tuple[str, Path]:
    """Create the mock CSV and tessera tiles; return (data_dir, tessera_dir)."""
    mock_dir = tmp_path / "mock"
    tessera_dir = mock_dir / "eo" / "tessera"
    tessera_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "name_loc": ["LOC_A", "LOC_B", "LOC_C"],
            "lat": [1.0, 1.0, 2.0],
            "lon": [1.0, 1.0, 2.0],
            "year": [2019, 2020, 2020],
            "country": ["ETH", "ETH", "ETH"],
            "target_yld_ton_ha": [2.0, 2.5, 1.8],
            **_FEAT_COLS,
        }
    )
    df.to_csv(mock_dir / "model_ready_mock.csv", index=False)

    rng = np.random.default_rng(42)
    np.save(tessera_dir / "tessera_LOC_A_2019.npy", _make_tile(rng))
    np.save(tessera_dir / "tessera_LOC_B_2020.npy", _make_tile(rng))
    np.save(tessera_dir / "tessera_LOC_C_2020.npy", _make_tile(rng))
    # No tile for (lat=2.0, lon=2.0, year=2019) — LOC_C has no prev-year tile.

    return str(tmp_path), tessera_dir


def _make_dataset(
    data_dir: str,
    tmp_path: Path,
    modalities: dict,
    **kwargs,
) -> YieldAfricaDataset:
    return YieldAfricaDataset(
        data_dir=data_dir,
        cache_dir=str(tmp_path / "cache"),
        modalities=modalities,
        use_target_data=True,
        use_aux_data="none",
        seed=42,
        mock=True,
        use_features=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def yield_africa_dual_csv(request, tmp_path) -> tuple[str, Path]:
    """Mock CSV + tessera tiles; returns (data_dir, tessera_dir)."""
    use_mock = request.config.getoption("--use-mock")
    if not use_mock:
        assert False, "Real data not available in test environment."
    return _write_mock_dir(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tessera_filename_includes_year(yield_africa_dual_csv, tmp_path):
    """tessera_path in records must end with _{year}.npy after path rewriting."""
    data_dir, _ = yield_africa_dual_csv
    ds = _make_dataset(data_dir, tmp_path, SINGLE_MODALITIES)
    for rec in ds.records:
        assert rec["tessera_path"].endswith(".npy")
        # Filename stem is tessera_{name_loc}_{year}
        fname = os.path.basename(rec["tessera_path"])
        assert fname.startswith("tessera_")
        # Year appears as a four-digit suffix immediately before .npy
        stem = fname.removesuffix(".npy")
        year_part = stem.split("_")[-1]
        assert year_part.isdigit() and len(year_part) == 4, f"Expected year suffix in '{fname}'"


def test_single_year_unaffected(yield_africa_dual_csv, tmp_path):
    """Dataset with only `tessera` modality loads without tessera_prev_path in records."""
    data_dir, _ = yield_africa_dual_csv
    ds = _make_dataset(data_dir, tmp_path, SINGLE_MODALITIES)

    assert len(ds) == 3  # all records have year-Y tiles on disk
    for rec in ds.records:
        assert "tessera_path" in rec
        assert "tessera_prev_path" not in rec

    sample = ds[0]
    assert "tessera" in sample["eo"]
    assert "tessera_prev" not in sample["eo"]


def test_prev_path_points_to_existing_tile(yield_africa_dual_csv, tmp_path):
    """tessera_prev_path for LOC_B must point to LOC_A's tessera file — no copy."""
    data_dir, tessera_dir = yield_africa_dual_csv
    # require=True → only LOC_B survives (A has no 2018 tile; C has no 2019 tile)
    ds = _make_dataset(data_dir, tmp_path, DUAL_MODALITIES, require_prev_year_tessera=True)

    assert len(ds) == 1
    rec = ds.records[0]
    assert rec["tessera_path"].endswith("tessera_LOC_B_2020.npy")

    prev_path = rec["tessera_prev_path"]
    assert prev_path.endswith("tessera_LOC_A_2019.npy")
    # Must be the original file — no copy written
    assert os.path.exists(prev_path)
    assert os.path.dirname(prev_path) == str(tessera_dir)


def test_dual_year_both_keys_in_sample(yield_africa_dual_csv, tmp_path):
    """Sample['eo'] must contain both 'tessera' and 'tessera_prev' tensors."""
    data_dir, _ = yield_africa_dual_csv
    ds = _make_dataset(data_dir, tmp_path, DUAL_MODALITIES, require_prev_year_tessera=True)

    assert len(ds) == 1
    sample = ds[0]
    assert "tessera" in sample["eo"]
    assert "tessera_prev" in sample["eo"]
    assert isinstance(sample["eo"]["tessera"], torch.Tensor)
    assert isinstance(sample["eo"]["tessera_prev"], torch.Tensor)


def test_tensor_shape_matches_size_config(yield_africa_dual_csv, tmp_path):
    """Both tessera tensors must have shape [channels, size, size]."""
    data_dir, _ = yield_africa_dual_csv
    ds = _make_dataset(data_dir, tmp_path, DUAL_MODALITIES, require_prev_year_tessera=True)

    sample = ds[0]
    expected = (TILE_CHANNELS, TILE_SIZE, TILE_SIZE)
    assert sample["eo"]["tessera"].shape == expected
    assert sample["eo"]["tessera_prev"].shape == expected


def test_missing_prev_tile_dropped_by_default(yield_africa_dual_csv, tmp_path):
    """Records with no resolvable year-1 tile are dropped when require_prev_year_tessera=True."""
    data_dir, _ = yield_africa_dual_csv
    ds = _make_dataset(data_dir, tmp_path, DUAL_MODALITIES, require_prev_year_tessera=True)

    # LOC_A has no 2018 tile; LOC_C has no 2019 tile at its location.
    # Only LOC_B (whose prev-year tile is LOC_A's 2019 tile) survives.
    assert len(ds) == 1
    assert ds.records[0]["tessera_path"].endswith("tessera_LOC_B_2020.npy")


def test_missing_prev_tile_kept_when_not_required(yield_africa_dual_csv, tmp_path):
    """Records with no prev tile are retained with tessera_prev_path=None when require=False."""
    data_dir, _ = yield_africa_dual_csv
    ds = _make_dataset(data_dir, tmp_path, DUAL_MODALITIES, require_prev_year_tessera=False)

    # All 3 records are retained (prev tile missing → None, not dropped).
    assert len(ds) == 3

    prev_paths = [rec["tessera_prev_path"] for rec in ds.records]
    none_count = sum(1 for p in prev_paths if p is None)
    resolved_count = sum(1 for p in prev_paths if p is not None)
    assert none_count == 2  # LOC_A (no 2018 tile) + LOC_C (no 2019 tile at its location)
    assert resolved_count == 1  # LOC_B resolves to LOC_A's 2019 tile


def test_year1_row_excluded_by_filter_still_resolves(yield_africa_dual_csv, tmp_path):
    """LOC_B resolves its year-1 path even when years=[2020] excludes LOC_A from training."""
    data_dir, _ = yield_africa_dual_csv
    ds = _make_dataset(
        data_dir,
        tmp_path,
        DUAL_MODALITIES,
        require_prev_year_tessera=True,
        years=[2020],
    )

    # years=[2020] removes LOC_A (2019) from self.df after filtering.
    # But the cross-year index was built before the filter → LOC_B still resolves.
    # LOC_C (2020, unique location) still has no prev tile → dropped.
    assert len(ds) == 1
    rec = ds.records[0]
    assert rec["tessera_path"].endswith("tessera_LOC_B_2020.npy")
    assert rec["tessera_prev_path"].endswith("tessera_LOC_A_2019.npy")
    assert os.path.exists(rec["tessera_prev_path"])


def test_no_duplicate_file_written(yield_africa_dual_csv, tmp_path):
    """Dataset initialisation must not write any new files to the tessera directory."""
    data_dir, tessera_dir = yield_africa_dual_csv
    files_before = set(tessera_dir.iterdir())

    _make_dataset(data_dir, tmp_path, DUAL_MODALITIES, require_prev_year_tessera=False)

    files_after = set(tessera_dir.iterdir())
    assert files_after == files_before, f"Unexpected new files: {files_after - files_before}"
