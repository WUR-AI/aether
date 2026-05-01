import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.data.yield_africa_caption_builder import YieldAfricaCaptionBuilder

DATA_DIR = str(Path(__file__).parent.parent / "data" / "yield_africa")

_CSV_FILES = [
    "soil_classes.csv",
    "climate_classes.csv",
    "terrain_classes.csv",
    "landcover_classes.csv",
    "ndvi_classes.csv",
    "agera5_classes.csv",
    "derived_classes.csv",
    "target_classes.csv",
]


def _all_aux_columns() -> list[str]:
    cols = []
    for fname in _CSV_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            cols.extend(pd.read_csv(fpath)["col"].tolist())
    return cols


def _col_index(cols: list[str], col: str) -> int:
    return cols.index(col)


@pytest.fixture(scope="module")
def aux_columns() -> list[str]:
    return _all_aux_columns()


@pytest.fixture(scope="module")
def mock_dataset(aux_columns):
    return SimpleNamespace(use_aux_data={"aux": aux_columns})


@pytest.fixture(scope="module")
def builder(mock_dataset) -> YieldAfricaCaptionBuilder:
    cb = YieldAfricaCaptionBuilder(
        templates_fname="v1.json",
        concepts_fname="v1.json",
        data_dir=DATA_DIR,
        seed=42,
    )
    cb.sync_with_dataset(mock_dataset)
    return cb


@pytest.fixture(scope="module")
def zero_aux(aux_columns) -> torch.Tensor:
    """All-zero encoded tensor: encoded 0 → ordinal rank 3 for standard columns."""
    return torch.zeros(len(aux_columns), dtype=torch.long)


# --- sync_with_dataset ---


def test_sync_populates_all_groups(builder):
    assert set(builder.group_columns.keys()) == set(YieldAfricaCaptionBuilder.GROUPS)
    for group in YieldAfricaCaptionBuilder.GROUPS:
        assert len(builder.group_columns[group]) > 0, f"Group '{group}' is empty after sync"


def test_sync_column_to_metadata_has_aux_key(builder):
    assert "aux" in builder.column_to_metadata_map


def test_sync_all_columns_have_id(builder, aux_columns):
    for col in aux_columns:
        assert col in builder.column_to_metadata_map["aux"]
        assert builder.column_to_metadata_map["aux"][col]["id"] >= 0


# --- ordinal_rank ---


def test_ordinal_rank_standard_column(builder):
    # encoded 3 → "Very high ..." → ordinal rank 4
    assert builder.ordinal_rank("aux_c_0_20_cl", 3) == 4


def test_ordinal_rank_standard_column_low(builder):
    # encoded 1 → "Low ..." → ordinal rank 1
    assert builder.ordinal_rank("aux_map_cl", 1) == 1


def test_ordinal_rank_slope(builder):
    # SLOPE4 map {0:1, 1:2, 2:3, 3:0} — encoded 3 → ordinal rank 0
    assert builder.ordinal_rank("aux_slope_cl", 3) == 0


def test_ordinal_rank_nodata_tree_c(builder):
    # tree_c encoded 3 → NoData → ordinal rank -1
    assert builder.ordinal_rank("aux_tree_c_cl", 3) == -1


def test_ordinal_rank_nodata_pop(builder):
    # pop_10km encoded 3 → NoData → ordinal rank -1
    assert builder.ordinal_rank("aux_pop_10km_cl", 3) == -1


def test_ordinal_rank_ndvi(builder):
    # NDVI uses standard map: encoded 3 → ordinal rank 4
    assert builder.ordinal_rank("aux_ndvi_mean_grow_cl", 3) == 4


def test_ordinal_rank_agera5(builder):
    assert builder.ordinal_rank("aux_agera5_prec_grow_cl", 3) == 4


def test_ordinal_rank_nominal_raises(builder):
    with pytest.raises(ValueError, match="nominal"):
        builder.ordinal_rank("aux_glad_cl", 1)


def test_ordinal_rank_unknown_column_raises(builder):
    with pytest.raises(KeyError):
        builder.ordinal_rank("aux_nonexistent_cl", 0)


# --- _class_label ---


def test_class_label_standard(builder):
    label = builder._class_label("aux_map_cl", 3)
    assert label != ""
    assert "very high" in label.lower()


def test_class_label_ndvi(builder):
    label = builder._class_label("aux_ndvi_mean_mam_cl", 3)
    assert label != ""
    assert "ndvi" in label.lower() or "march" in label.lower() or "mam" in label.lower()


def test_class_label_nominal_glad(builder):
    label = builder._class_label("aux_glad_cl", 1)
    assert label == "Cropland"


def test_class_label_nodata(builder):
    label = builder._class_label("aux_tree_c_cl", 3)
    assert "nodata" in label.lower()


# --- _get_top_n_for_group ---


def test_top_n_excludes_nominal_terrain(builder, zero_aux, aux_columns):
    # Terrain group has nominal columns aspect and landform — they must never appear as top-N
    nominal_labels = {
        "east",
        "north",
        "northeast",
        "northwest",
        "south",
        "southeast",
        "southwest",
        "west",
        "cliff",
        "lower slope",
        "mountain/divide",
        "peak/ridge",
        "upper slope",
        "valley",
    }
    for n in (1, 2):
        result = builder._get_top_n_for_group("terrain", zero_aux, n)
        assert (
            result.lower() not in nominal_labels
        ), f"Nominal label '{result}' returned as terrain top-{n}"


def test_top_n_ranking_uses_deviation(builder, aux_columns):
    # Set all terrain ordinal columns to encoded 2 (ordinal rank 2, deviation 0),
    # then set aux_dem_cl to encoded 3 (ordinal rank 4, deviation 2) — it must win.
    tensor = torch.zeros(len(aux_columns), dtype=torch.long)
    for col in builder.group_columns["terrain"]:
        meta = builder.column_to_metadata_map["aux"][col]
        tensor[meta["id"]] = 2
    dem_id = builder.column_to_metadata_map["aux"]["aux_dem_cl"]["id"]
    tensor[dem_id] = 3  # ordinal rank 4, deviation 2

    result = builder._get_top_n_for_group("terrain", tensor, 1)
    assert "very high elevation" in result.lower()


def test_top_n_ndvi_group(builder, zero_aux):
    result = builder._get_top_n_for_group("ndvi", zero_aux, 1)
    assert result != ""
    # Should mention NDVI or a season
    assert any(
        kw in result.lower() for kw in ("ndvi", "growing", "march", "mam", "SON", "contrast")
    )


def test_top_n_agera5_group(builder, zero_aux):
    result = builder._get_top_n_for_group("agera5", zero_aux, 1)
    assert result != ""


def test_top_n_nodata_excluded(builder, aux_columns):
    # Set tree_c to encoded 3 (NoData, ordinal rank -1); other landcover columns
    # to encoded 2. Top-1 must not mention NoData.
    tensor = torch.zeros(len(aux_columns), dtype=torch.long)
    for col in builder.group_columns["landcover"]:
        meta = builder.column_to_metadata_map["aux"][col]
        if builder.column_to_metadata_map["aux"][col].get("ordinal_map") is not None:
            tensor[meta["id"]] = 2
    tree_id = builder.column_to_metadata_map["aux"]["aux_tree_c_cl"]["id"]
    tensor[tree_id] = 3  # NoData

    result = builder._get_top_n_for_group("landcover", tensor, 1)
    assert "nodata" not in result.lower()


def test_top_n_returns_empty_when_n_exceeds_valid(builder, aux_columns):
    # Set all terrain columns to encoded 2 (deviation 0) except the ones we know
    tensor = torch.zeros(len(aux_columns), dtype=torch.long)
    # Requesting top-100 from a group of ~4 ordinal columns must return ""
    result = builder._get_top_n_for_group("terrain", tensor, 100)
    assert result == ""


# --- _build_from_template ---


def test_all_templates_render(builder, zero_aux):
    for i in range(len(builder.templates)):
        caption = builder._build_from_template(i, zero_aux)
        assert isinstance(caption, str)
        assert "<" not in caption, f"Template {i} has unfilled token: '{caption}'"
        assert caption != ""


def test_build_uses_class_label_text(builder, aux_columns):
    # Encoded 3 → "Very high ..." for aux_map_cl
    tensor = torch.zeros(len(aux_columns), dtype=torch.long)
    map_id = builder.column_to_metadata_map["aux"]["aux_map_cl"]["id"]
    tensor[map_id] = 3

    # Find a template that uses aux_map_cl directly
    idx = next(
        i
        for i, tokens in enumerate(builder.tokens_in_template)
        if "aux_map_cl" in tokens
        and not any("top" in t for t in tokens)  # avoid top-N templates for simplicity
    )
    caption = builder._build_from_template(idx, tensor)
    assert "very high" in caption.lower()


def test_build_top_n_token_resolves(builder, zero_aux):
    # Find a template that uses a top-N token
    idx = next(
        i
        for i, tokens in enumerate(builder.tokens_in_template)
        if any("_top_" in t for t in tokens)
    )
    caption = builder._build_from_template(idx, zero_aux)
    assert "<" not in caption


# --- sync_concepts ---


def test_sync_concepts_all_have_id(builder):
    for concept in builder.concepts:
        assert "id" in concept, f"Concept '{concept['concept_caption']}' has no 'id' after sync"
        assert concept["id"] >= 0


def test_sync_concepts_count(builder):
    assert len(builder.concepts) == 48
