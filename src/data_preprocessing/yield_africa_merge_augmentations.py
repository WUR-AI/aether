"""Merge augmented yield Africa CSVs into a single combined CSV.

Combines the base model-ready CSV with any combination of NDVI and AgERA5
augmented CSVs.  Shared columns (all base feat_* columns, plus name_loc, lat,
lon, year, country, target_*, aux_*) appear exactly once in the output.
Augmentation-specific columns are added alongside without duplication.

After merging, augmentation features (NDVI seasonal means, AgERA5 variables)
are classified into 5-class ordinal aux_*_cl columns using training-split
quintile boundaries.  The fitted LabelEncoders are added to the existing
encoders pkl saved by make_model_ready_yield_africa.py.

Usage:
    # Base + NDVI + AgERA5 (full merge, with classification)
    python src/data_preprocessing/yield_africa_merge_augmentations.py \\
        --base_csv   data/yield_africa/model_ready_yield_africa.csv \\
        --ndvi_csv   data/yield_africa/model_ready_yield_africa_ndvi.csv \\
        --agera5_csv data/yield_africa/model_ready_yield_africa_agera5.csv \\
        --out_csv    data/yield_africa/model_ready_yield_africa_merged.csv

    # Base + AgERA5 only
    python src/data_preprocessing/yield_africa_merge_augmentations.py \\
        --base_csv   data/yield_africa/model_ready_yield_africa.csv \\
        --agera5_csv data/yield_africa/model_ready_yield_africa_agera5.csv \\
        --out_csv    data/yield_africa/model_ready_yield_africa_merged.csv

    # Merge without aux classification (classification step skipped)
    python src/data_preprocessing/yield_africa_merge_augmentations.py \\
        --base_csv   data/yield_africa/model_ready_yield_africa.csv \\
        --ndvi_csv   data/yield_africa/model_ready_yield_africa_ndvi.csv \\
        --no_classify \\
        --out_csv    data/yield_africa/model_ready_yield_africa_merged.csv

Optional arguments:
    --base_csv      Path to model_ready_yield_africa.csv (required)
    --ndvi_csv      Path to model_ready_yield_africa_ndvi.csv (optional)
    --agera5_csv    Path to model_ready_yield_africa_agera5.csv (optional)
    --out_csv       Output path for merged CSV
                    (default: model_ready_yield_africa_merged.csv beside --base_csv)
    --how           Join type: 'left' keeps all base rows (default); 'inner' drops
                    rows not present in every provided CSV
    --encoders_pkl  Path to label_encoders_yield_africa.pkl to update with new
                    aux encoders (default: auto-detected beside --base_csv)
    --no_classify   Skip aux classification of augmentation features
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

JOIN_KEYS = ["name_loc", "year"]

# Countries used for training (determines the train mask for quintile fitting).
# Must match TRAIN_COUNTRIES in make_model_ready_yield_africa.py.
_TRAIN_COUNTRIES = ["BUR", "ETH", "KEN", "MAL", "RWA", "TAN", "ZAM"]

# ---------------------------------------------------------------------------
# Augmentation classification specs
#
# Each entry maps a source feat_* column to:
#   (output_aux_col, description, unit, n_decimals, scale)
#
# description  — inserted into label strings, e.g. "mean NDVI in the growing season"
# unit         — appended after the value range, e.g. "mm" or "°C"; empty for dimensionless
# n_decimals   — decimal places used in the value range within label strings
# scale        — multiply raw values by this before formatting labels (e.g. 1e-6 for J→MJ)
#
# Snow columns (feat_agera5_snow_*) are intentionally excluded: all-zero for sub-Saharan Africa.
# ---------------------------------------------------------------------------

_NDVI_CLASSIFY_SPECS: Dict[str, Tuple[str, str, str, int, float]] = {
    "feat_ndvi_mean_djf": (
        "aux_ndvi_mean_djf_cl",
        "mean NDVI in December-January-February",
        "",
        2,
        1.0,
    ),
    "feat_ndvi_mean_mam": (
        "aux_ndvi_mean_mam_cl",
        "mean NDVI in March-April-May (long rains)",
        "",
        2,
        1.0,
    ),
    "feat_ndvi_mean_jja": ("aux_ndvi_mean_jja_cl", "mean NDVI in June-July-August", "", 2, 1.0),
    "feat_ndvi_mean_son": (
        "aux_ndvi_mean_son_cl",
        "mean NDVI in September-October-November (short rains)",
        "",
        2,
        1.0,
    ),
    "feat_ndvi_mean_grow": (
        "aux_ndvi_mean_grow_cl",
        "mean NDVI in the growing season",
        "",
        2,
        1.0,
    ),
    # Phenological contrast columns — derived from seasonal means in compute_ndvi_contrasts()
    "feat_ndvi_contrast_mam_jja": (
        "aux_ndvi_contrast_mam_jja_cl",
        "long rains green-up contrast (MAM minus JJA NDVI)",
        "",
        2,
        1.0,
    ),
    "feat_ndvi_contrast_son_djf": (
        "aux_ndvi_contrast_son_djf_cl",
        "short rains green-up contrast (SON minus DJF NDVI)",
        "",
        2,
        1.0,
    ),
}

_AGERA5_CLASSIFY_SPECS: Dict[str, Tuple[str, str, str, int, float]] = {
    # Maximum temperature
    "feat_agera5_tmax_djf": (
        "aux_agera5_tmax_djf_cl",
        "AGERA5 maximum temperature in December-January-February",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmax_mam": (
        "aux_agera5_tmax_mam_cl",
        "AGERA5 maximum temperature in March-April-May",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmax_jja": (
        "aux_agera5_tmax_jja_cl",
        "AGERA5 maximum temperature in June-July-August",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmax_son": (
        "aux_agera5_tmax_son_cl",
        "AGERA5 maximum temperature in September-October-November",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmax_grow": (
        "aux_agera5_tmax_grow_cl",
        "AGERA5 maximum temperature in the growing season",
        "°C",
        1,
        1.0,
    ),
    # Minimum temperature
    "feat_agera5_tmin_djf": (
        "aux_agera5_tmin_djf_cl",
        "AGERA5 minimum temperature in December-January-February",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmin_mam": (
        "aux_agera5_tmin_mam_cl",
        "AGERA5 minimum temperature in March-April-May",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmin_jja": (
        "aux_agera5_tmin_jja_cl",
        "AGERA5 minimum temperature in June-July-August",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmin_son": (
        "aux_agera5_tmin_son_cl",
        "AGERA5 minimum temperature in September-October-November",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tmin_grow": (
        "aux_agera5_tmin_grow_cl",
        "AGERA5 minimum temperature in the growing season",
        "°C",
        1,
        1.0,
    ),
    # Mean temperature
    "feat_agera5_tavg_djf": (
        "aux_agera5_tavg_djf_cl",
        "AGERA5 mean temperature in December-January-February",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tavg_mam": (
        "aux_agera5_tavg_mam_cl",
        "AGERA5 mean temperature in March-April-May",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tavg_jja": (
        "aux_agera5_tavg_jja_cl",
        "AGERA5 mean temperature in June-July-August",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tavg_son": (
        "aux_agera5_tavg_son_cl",
        "AGERA5 mean temperature in September-October-November",
        "°C",
        1,
        1.0,
    ),
    "feat_agera5_tavg_grow": (
        "aux_agera5_tavg_grow_cl",
        "AGERA5 mean temperature in the growing season",
        "°C",
        1,
        1.0,
    ),
    # Vapor pressure
    "feat_agera5_vp_djf": (
        "aux_agera5_vp_djf_cl",
        "AGERA5 vapor pressure in December-January-February",
        "hPa",
        1,
        1.0,
    ),
    "feat_agera5_vp_mam": (
        "aux_agera5_vp_mam_cl",
        "AGERA5 vapor pressure in March-April-May",
        "hPa",
        1,
        1.0,
    ),
    "feat_agera5_vp_jja": (
        "aux_agera5_vp_jja_cl",
        "AGERA5 vapor pressure in June-July-August",
        "hPa",
        1,
        1.0,
    ),
    "feat_agera5_vp_son": (
        "aux_agera5_vp_son_cl",
        "AGERA5 vapor pressure in September-October-November",
        "hPa",
        1,
        1.0,
    ),
    "feat_agera5_vp_grow": (
        "aux_agera5_vp_grow_cl",
        "AGERA5 vapor pressure in the growing season",
        "hPa",
        1,
        1.0,
    ),
    # Wind speed
    "feat_agera5_ws_djf": (
        "aux_agera5_ws_djf_cl",
        "AGERA5 wind speed in December-January-February",
        "m/s",
        1,
        1.0,
    ),
    "feat_agera5_ws_mam": (
        "aux_agera5_ws_mam_cl",
        "AGERA5 wind speed in March-April-May",
        "m/s",
        1,
        1.0,
    ),
    "feat_agera5_ws_jja": (
        "aux_agera5_ws_jja_cl",
        "AGERA5 wind speed in June-July-August",
        "m/s",
        1,
        1.0,
    ),
    "feat_agera5_ws_son": (
        "aux_agera5_ws_son_cl",
        "AGERA5 wind speed in September-October-November",
        "m/s",
        1,
        1.0,
    ),
    "feat_agera5_ws_grow": (
        "aux_agera5_ws_grow_cl",
        "AGERA5 wind speed in the growing season",
        "m/s",
        1,
        1.0,
    ),
    # Precipitation
    "feat_agera5_prec_djf": (
        "aux_agera5_prec_djf_cl",
        "AGERA5 precipitation in December-January-February",
        "mm",
        0,
        1.0,
    ),
    "feat_agera5_prec_mam": (
        "aux_agera5_prec_mam_cl",
        "AGERA5 precipitation in March-April-May (long rains)",
        "mm",
        0,
        1.0,
    ),
    "feat_agera5_prec_jja": (
        "aux_agera5_prec_jja_cl",
        "AGERA5 precipitation in June-July-August",
        "mm",
        0,
        1.0,
    ),
    "feat_agera5_prec_son": (
        "aux_agera5_prec_son_cl",
        "AGERA5 precipitation in September-October-November (short rains)",
        "mm",
        0,
        1.0,
    ),
    "feat_agera5_prec_grow": (
        "aux_agera5_prec_grow_cl",
        "AGERA5 total precipitation in the growing season",
        "mm",
        0,
        1.0,
    ),
    # Solar radiation (raw J/m², labels in MJ/m² via scale=1e-6)
    "feat_agera5_rad_djf": (
        "aux_agera5_rad_djf_cl",
        "AGERA5 solar radiation in December-January-February",
        "MJ/m²",
        0,
        1e-6,
    ),
    "feat_agera5_rad_mam": (
        "aux_agera5_rad_mam_cl",
        "AGERA5 solar radiation in March-April-May",
        "MJ/m²",
        0,
        1e-6,
    ),
    "feat_agera5_rad_jja": (
        "aux_agera5_rad_jja_cl",
        "AGERA5 solar radiation in June-July-August",
        "MJ/m²",
        0,
        1e-6,
    ),
    "feat_agera5_rad_son": (
        "aux_agera5_rad_son_cl",
        "AGERA5 solar radiation in September-October-November",
        "MJ/m²",
        0,
        1e-6,
    ),
    "feat_agera5_rad_grow": (
        "aux_agera5_rad_grow_cl",
        "AGERA5 total solar radiation in the growing season",
        "MJ/m²",
        0,
        1e-6,
    ),
    # Growing season aggregates
    "feat_agera5_gdd10_grow": (
        "aux_agera5_gdd10_grow_cl",
        "AGERA5 growing degree days (base 10°C) in the growing season",
        "days",
        0,
        1.0,
    ),
    "feat_agera5_wetdays_grow": (
        "aux_agera5_wetdays_grow_cl",
        "AGERA5 wet days in the growing season",
        "days",
        0,
        1.0,
    ),
}

_ALL_CLASSIFY_SPECS = {**_NDVI_CLASSIFY_SPECS, **_AGERA5_CLASSIFY_SPECS}

_LEVEL_PREFIXES = ["Very low", "Low", "Medium", "High", "Very high"]


# ---------------------------------------------------------------------------
# Aux classification helpers
# ---------------------------------------------------------------------------


def compute_ndvi_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Derive phenological contrast columns from NDVI seasonal means.

    Both source columns must be present; if either is missing the contrast column is silently
    skipped (classify_augmentation_features will also skip it via the missing-column guard).
    """
    df = df.copy()
    if "feat_ndvi_mean_mam" in df.columns and "feat_ndvi_mean_jja" in df.columns:
        df["feat_ndvi_contrast_mam_jja"] = df["feat_ndvi_mean_mam"] - df["feat_ndvi_mean_jja"]
    if "feat_ndvi_mean_son" in df.columns and "feat_ndvi_mean_djf" in df.columns:
        df["feat_ndvi_contrast_son_djf"] = df["feat_ndvi_mean_son"] - df["feat_ndvi_mean_djf"]
    return df


def _classify_continuous_to_5class(
    series: pd.Series,
    train_mask: pd.Series,
    description: str,
    unit: str,
    n_decimals: int,
    scale: float = 1.0,
) -> pd.Series:
    """Classify a continuous series into 5 ordinal label strings.

    Quintile boundaries (p20, p40, p60, p80) are fitted on training rows only. NaN inputs produce
    NaN outputs.  Labels follow the "Very low / Low / Medium / High / Very high" naming convention
    so that sklearn LabelEncoder produces the same alphabetical encoding map as all other aux_*_cl
    columns: {0:3, 1:1, 2:2, 3:4, 4:0} (H=0, L=1, M=2, VH=3, VL=4).
    """
    train_vals = (series[train_mask] * scale).dropna()
    if len(train_vals) == 0:
        return pd.Series([np.nan] * len(series), index=series.index)

    bounds = train_vals.quantile([0.2, 0.4, 0.6, 0.8]).values
    fmt = f".{n_decimals}f"

    unit_str = f" {unit}" if unit else ""

    def _label(v: float) -> Optional[str]:
        if pd.isna(v):
            return np.nan
        sv = v * scale
        if sv < bounds[0]:
            return f"Very low {description} (<{bounds[0]:{fmt}}{unit_str})"
        elif sv < bounds[1]:
            return f"Low {description} ({bounds[0]:{fmt}}-{bounds[1]:{fmt}}{unit_str})"
        elif sv < bounds[2]:
            return f"Medium {description} ({bounds[1]:{fmt}}-{bounds[2]:{fmt}}{unit_str})"
        elif sv < bounds[3]:
            return f"High {description} ({bounds[2]:{fmt}}-{bounds[3]:{fmt}}{unit_str})"
        else:
            return f"Very high {description} (>{bounds[3]:{fmt}}{unit_str})"

    return series.map(_label)


def classify_augmentation_features(
    df: pd.DataFrame,
    specs: Dict[str, Tuple[str, str, str, int, float]],
    train_countries: List[str],
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Classify augmentation ``feat_*`` columns into 5-class ``aux_*_cl`` columns.

    Only specs whose source ``feat_*`` column is present in df are processed. Returns the augmented
    dataframe and a dict of fitted LabelEncoders keyed by the output ``aux_*_cl`` column name
    (already renamed to the ``aux_`` convention).
    """
    df = df.copy()
    train_mask = df["country"].str.upper().isin([c.upper() for c in train_countries])
    encoders: Dict[str, LabelEncoder] = {}
    processed = 0

    for feat_col, (aux_col, description, unit, n_decimals, scale) in specs.items():
        if feat_col not in df.columns:
            continue

        label_series = _classify_continuous_to_5class(
            df[feat_col], train_mask, description, unit, n_decimals, scale
        )
        df[aux_col] = label_series

        # Fit encoder on training labels only (dropna excludes NaN from fits)
        train_labels = label_series[train_mask].dropna()
        if len(train_labels) == 0:
            warnings.warn(f"No training labels generated for {feat_col}; skipping encoder")
            continue

        enc = LabelEncoder()
        enc.fit(train_labels)
        # Apply encoder: unknown values (NaN after label generation) become -1
        df[aux_col] = df[aux_col].apply(
            lambda x: (
                enc.transform([x])[0]
                if (
                    x is not None
                    and not (isinstance(x, float) and np.isnan(x))
                    and x in enc.classes_
                )
                else -1
            )
        )
        encoders[aux_col] = enc
        processed += 1

    print(f"  Classified {processed} augmentation features into aux_*_cl columns")
    return df, encoders


def update_encoders_pkl(
    encoders_pkl: Path,
    new_encoders: Dict[str, LabelEncoder],
) -> None:
    """Load existing encoders pkl, merge new encoders, and save."""
    if encoders_pkl.exists():
        existing = joblib.load(encoders_pkl)
        existing.update(new_encoders)
        joblib.dump(existing, encoders_pkl)
        print(f"  Updated encoders pkl ({len(existing)} total): {encoders_pkl}")
    else:
        joblib.dump(new_encoders, encoders_pkl)
        print(f"  Created encoders pkl ({len(new_encoders)} encoders): {encoders_pkl}")


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------


def sep(title: str = "", width: int = 72) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


def _load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure year is int so join keys match across files
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    print(
        f"  {label:12s}: {df.shape[0]:,} rows × {df.shape[1]} cols  "
        f"({len(_feat_cols(df))} feat_*)  [{path.name}]"
    )
    return df


def _augmentation_only_cols(aug_df: pd.DataFrame, base_cols: set[str]) -> list[str]:
    """Return columns in aug_df that are not already in base_cols.

    Always includes sentinel columns like 'agera5_fetched' even though they don't start with
    ``feat_``, as long as they aren't in the base.
    """
    return [c for c in aug_df.columns if c not in base_cols]


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------


def merge_augmentations(
    base_df: pd.DataFrame,
    augmentations: list[tuple[str, pd.DataFrame]],
    how: str,
) -> pd.DataFrame:
    """Left- or inner-join each augmentation onto base_df by name_loc + year.

    Only the columns that are genuinely new in each augmentation CSV are attached; any column
    already present in the running result is skipped.
    """
    result = base_df.copy()
    result_cols = set(result.columns)

    for label, aug_df in augmentations:
        sep(f"Merging {label}")

        new_cols = _augmentation_only_cols(aug_df, result_cols)
        if not new_cols:
            print(f"  No new columns found in {label} — skipping.")
            continue

        print(
            f"  New columns to add ({len(new_cols)}): "
            + ", ".join(new_cols[:8])
            + (" …" if len(new_cols) > 8 else "")
        )

        # Coverage check before merge
        base_keys = set(zip(result["name_loc"], result["year"]))
        aug_keys = set(zip(aug_df["name_loc"], aug_df["year"]))
        only_base = base_keys - aug_keys
        only_aug = aug_keys - base_keys
        if only_base:
            print(
                f"  {len(only_base):,} base rows have no match in {label} "
                f"{'(will be NaN)' if how == 'left' else '(dropped by inner join)'}"
            )
        if only_aug:
            print(f"  {len(only_aug):,} {label} rows have no match in base (ignored)")

        # Select only join keys + new columns from the augmentation dataframe
        merge_cols = JOIN_KEYS + new_cols
        # Guard: join keys might not be in new_cols list, ensure no duplicates
        merge_subset = aug_df[merge_cols].copy()

        result = result.merge(
            merge_subset, on=JOIN_KEYS, how=how, suffixes=("", f"_{label.lower()}")
        )

        # Update known columns
        result_cols = set(result.columns)

        n_nan = result[new_cols].isna().any(axis=1).sum()
        print(f"  Rows with ≥1 NaN in new {label} columns: {n_nan:,} / {len(result):,}")

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def merge_report(
    base_df: pd.DataFrame,
    result: pd.DataFrame,
    augmentations: list[tuple[str, pd.DataFrame]],
) -> None:
    sep("Merge summary")

    base_feat = [c for c in base_df.columns if c.startswith("feat_")]
    result_feat = _feat_cols(result)
    added_feat = [c for c in result_feat if c not in base_feat]

    print(f"  Base rows          : {len(base_df):,}")
    print(f"  Output rows        : {len(result):,}")
    print(f"  Base feat_* cols   : {len(base_feat)}")
    print(f"  Output feat_* cols : {len(result_feat)}  (+{len(added_feat)} added)")
    print(f"  Total output cols  : {result.shape[1]}")

    # Non-feat augmentation columns (e.g. sentinel agera5_fetched)
    non_feat_extra = [
        c for c in result.columns if c not in base_df.columns and not c.startswith("feat_")
    ]
    if non_feat_extra:
        print(f"  Extra non-feat cols: {non_feat_extra}")

    # Completeness per augmentation block
    print()
    for label, aug_df in augmentations:
        aug_only = _augmentation_only_cols(aug_df, set(base_df.columns))
        aug_feat_in_result = [c for c in aug_only if c in result.columns and c.startswith("feat_")]
        if not aug_feat_in_result:
            continue
        complete = result[aug_feat_in_result].notna().all(axis=1).sum()
        print(
            f"  {label}: {complete:,} / {len(result):,} rows fully populated "
            f"({100 * complete / len(result):.1f}%)"
        )

    # Duplicate column guard
    dupes = [c for c in result.columns if result.columns.tolist().count(c) > 1]
    if dupes:
        print(f"\n  WARNING: duplicate columns detected: {dupes}")
    else:
        print("\n  No duplicate columns.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base_csv", required=True, help="Path to model_ready_yield_africa.csv (base)"
    )
    parser.add_argument(
        "--ndvi_csv", default=None, help="Path to model_ready_yield_africa_ndvi.csv (optional)"
    )
    parser.add_argument(
        "--agera5_csv", default=None, help="Path to model_ready_yield_africa_agera5.csv (optional)"
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help=(
            "Output path for merged CSV "
            "(default: model_ready_yield_africa_merged.csv beside --base_csv)"
        ),
    )
    parser.add_argument(
        "--how",
        choices=["left", "inner"],
        default="left",
        help=(
            "Join type: 'left' keeps all base rows and fills missing augmentation "
            "columns with NaN; 'inner' keeps only rows present in every provided CSV "
            "(default: left)"
        ),
    )
    parser.add_argument(
        "--encoders_pkl",
        default=None,
        help=(
            "Path to label_encoders_yield_africa.pkl to update with new aux encoders "
            "(default: auto-detected as label_encoders_yield_africa.pkl beside --base_csv)"
        ),
    )
    parser.add_argument(
        "--no_classify",
        action="store_true",
        default=False,
        help="Skip classification of augmentation features into aux_*_cl columns.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_csv)
    if not base_path.exists():
        print(f"ERROR: base CSV not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    if args.ndvi_csv is None and args.agera5_csv is None:
        print(
            "ERROR: at least one of --ndvi_csv or --agera5_csv must be provided.", file=sys.stderr
        )
        sys.exit(1)

    out_path = (
        Path(args.out_csv)
        if args.out_csv
        else base_path.parent / "model_ready_yield_africa_merged.csv"
    )

    sep("Loading CSVs")
    base_df = _load(base_path, "Base")

    # Build ordered list of (label, dataframe) for augmentations
    augmentations: list[tuple[str, pd.DataFrame]] = []
    for label, csv_arg in [("NDVI", args.ndvi_csv), ("AgERA5", args.agera5_csv)]:
        if csv_arg is None:
            continue
        p = Path(csv_arg)
        if not p.exists():
            print(f"ERROR: {label} CSV not found: {p}", file=sys.stderr)
            sys.exit(1)
        augmentations.append((label, _load(p, label)))

    if not augmentations:
        print("No augmentation CSVs provided — nothing to merge.", file=sys.stderr)
        sys.exit(1)

    result = merge_augmentations(base_df, augmentations, how=args.how)
    merge_report(base_df, result, augmentations)

    if not args.no_classify:
        sep("Classifying augmentation features")
        result = compute_ndvi_contrasts(result)
        result, new_encoders = classify_augmentation_features(
            result, _ALL_CLASSIFY_SPECS, _TRAIN_COUNTRIES
        )
        if new_encoders:
            encoders_pkl = (
                Path(args.encoders_pkl)
                if args.encoders_pkl
                else (base_path.parent / "label_encoders_yield_africa.pkl")
            )
            update_encoders_pkl(encoders_pkl, new_encoders)

            new_aux_cols = list(new_encoders.keys())
            print(
                f"  New aux_*_cl columns added ({len(new_aux_cols)}): "
                + ", ".join(new_aux_cols[:6])
                + (" …" if len(new_aux_cols) > 6 else "")
            )
        else:
            print("  No augmentation features found to classify.")
    else:
        print("  Classification skipped (--no_classify)")

    sep("Writing output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, float_format="%.7f")
    print(f"  Saved: {out_path.resolve()}")
    print(f"  Shape: {result.shape[0]:,} rows × {result.shape[1]} columns")

    sep()
    print("Merge complete.")


if __name__ == "__main__":
    main()
