"""Build model-ready CSV/Parquet for the crop yield Africa use case
(data/yield_africa/model_ready_yield-africa.csv).

Features:
- Load raw dataset (CSV or Parquet)
- Compute derived features (CN_ratio, layer deltas, WHC proxy, aridity index)
- Apply log transforms to skewed features
- Fit StandardScaler on train split only
- Encode categorical features as integer indices
- Remove yield outliers beyond 3 IQR
- Preserve metadata columns
- Save fitted transformers for inference-time reuse
- Calculate and save spatial cross-validation splits
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_READY_DATA_NAME = "yield_africa"

TRAIN_COUNTRIES = ["BUR", "ETH", "KEN", "MAL", "RWA", "TAN", "ZAM"]

SPATIAL_SPLIT_BLOCK_SIZE_KM = 50.0
SPATIAL_SPLIT_N_SPLITS = 7

CONTINUOUS_FEATURES = [
    # Soil features
    "C_0_20",
    "C_20_50",
    "N_0_20",
    "N_20_50",
    "P_0_20",
    "P_20_50",
    "MA_0_20",
    "MA_20_50",
    "PO_0_20",
    "PO_20_50",
    "pH_0_20",
    "pH_20_50",
    "BD_0_20",
    "BD_20_50",
    "ECX_0_20",
    "ECX_20_50",
    "CA_0_20",
    "CA_20_50",
    # Climate features
    "PrecJJA",
    "PrecMAM",
    "PrecSON",
    "PrecDJF",
    "TaveJJA",
    "TaveMAM",
    "TaveSON",
    "TaveDJF",
    "TmaxJJA",
    "TmaxMAM",
    "TmaxSON",
    "TmaxDJF",
    "TminJJA",
    "TminMAM",
    "TminSON",
    "TminDJF",
    "CMD",
    "Eref",
    "MAP",
    "MAT",
    "TD",
    "MWMT",
    "MCMT",
    "DD_above_5",
    "DD_above_18",
    "DD_below_18",
    # Terrain features
    "DEM",
    "Slope",
    "Aspect",
    "CHILI",
    "Top_div",
    # Land cover / context
    "Tree_c",
    "Dist_water",
    "Paved",
    "Unpaved",
    "Pop_10km",
    # Derived features (computed automatically)
    "CN_ratio",
    "C_layer_delta",
    "BD_layer_delta",
    "WHC_proxy",
    "aridity_index",
]

# Categorical columns that are actual tabular inputs to the regression model (feat_ prefix).
# TX_*_cl are soil texture classes — they are not derived from a paired numerical column.
TABULAR_CATEGORICAL_FEATURES = [
    "TX_0_20_cl",
    "TX_20_50_cl",
]

# Categorical columns derived from their paired numerical columns (same name without _cl).
# Used for caption generation (aux_ prefix).
AUX_FEATURES = [
    # target (classified)
    "Yld_ton_ha_cl",
    # soil features
    "C_0_20_cl",
    "C_20_50_cl",
    "N_0_20_cl",
    "N_20_50_cl",
    "P_0_20_cl",
    "P_20_50_cl",
    "MA_0_20_cl",
    "MA_20_50_cl",
    "PO_0_20_cl",
    "PO_20_50_cl",
    "pH_0_20_cl",
    "pH_20_50_cl",
    "BD_0_20_cl",
    "BD_20_50_cl",
    "ECX_0_20_cl",
    "ECX_20_50_cl",
    "CA_0_20_cl",
    "CA_20_50_cl",
    # climate features
    "PrecJJA_cl",
    "PrecMAM_cl",
    "PrecSON_cl",
    "PrecDJF_cl",
    "TaveJJA_cl",
    "TaveMAM_cl",
    "TaveSON_cl",
    "TaveDJF_cl",
    "TmaxJJA_cl",
    "TmaxMAM_cl",
    "TmaxSON_cl",
    "TmaxDJF_cl",
    "TminJJA_cl",
    "TminMAM_cl",
    "TminSON_cl",
    "TminDJF_cl",
    "CMD_cl",
    "Eref_cl",
    "MAP_cl",
    "MAT_cl",
    "TD_cl",
    "MCMT_cl",
    "MWMT_cl",
    "DD_above_5_cl",
    "DD_above_18_cl",
    "DD_below_18_cl",
    # terrain features
    "DEM_cl",
    "Slope_cl",
    "Aspect_cl",
    "Landform_cl",
    "CHILI_cl",
    "Top_div_cl",
    # land cover / context
    "GLAD_cl",
    "Tree_c_cl",
    "Dist_water_cl",
    "Paved_cl",
    "Unpaved_cl",
    "Pop_10km_cl",
]

LOG_TRANSFORM_FEATURES = ["Dist_water", "Paved", "Unpaved", "Pop_10km"]

TARGET_COLUMNS = ["Yld_ton_ha"]

NAME_LOC_COLUMN = "ID"

METADATA_COLUMNS = ["Lat", "Lon", "Country", "Year", "Location_accuracy"]

# Saxton & Rawls (2006) Table 3-derived AWC (FC - WP)
# Conditions: ~2.5% OM, no salinity/gravel/density adjustment
# "Plant avail." (%v) converted to mm/m via mm/m = (%v) * 10
# (because 1% v/v = 0.01 m³/m³ = 10 mm per m soil)
# Values are in mm/m (approximate field capacity - wilting point)
WHC_LOOKUP_SAXTON_RAWLS_2006_OM2P5 = {
    "Sand": 50,
    "Loamy sand": 70,
    "Sandy loam": 100,
    "Loam": 140,
    "Silt loam": 200,
    "Silt": 250,
    "Sandy clay loam": 100,
    "Clay loam": 140,
    "Silty clay loam": 170,
    "Silty clay": 140,
    "Sandy clay": 110,
    "Clay": 120,
}

# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------

def build_column_rename_map(
    continuous_features: List[str],
    tabular_categorical_features: List[str],
    aux_features: List[str],
    target_columns: List[str],
    name_loc_column: str,
    metadata_columns: List[str],
) -> Dict[str, str]:
    """Build a column rename mapping that standardises predictor and target names.

    Convention:
    - Numerical predictors and tabular categorical features: ``feat_{original.lower()}``
    - Aux/caption features (derived categorical classes): ``aux_{original.lower()}``
    - Target columns: ``target_{original.lower()}``
    - Name-location column: ``name_loc``
    - Metadata columns: ``{original.lower()}``
    """
    rename: Dict[str, str] = {}
    for col in continuous_features:
        rename[col] = f"feat_{col.lower()}"
    for col in tabular_categorical_features:
        rename[col] = f"feat_{col.lower()}"
    for col in aux_features:
        rename[col] = f"aux_{col.lower()}"
    for col in target_columns:
        rename[col] = f"target_{col.lower()}"
    for col in metadata_columns:
        rename[col] = col.lower()
    if name_loc_column is not None:
        rename[name_loc_column] = "name_loc"
    return rename


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw measurements."""
    df = df.copy()

    # C:N ratio (guard against division by zero)
    df["CN_ratio"] = np.where(
        df["N_0_20"] > 0,
        df["C_0_20"] / df["N_0_20"],
        np.nan,
    )

    # Layer deltas (stratification indicators)
    df["C_layer_delta"] = df["C_0_20"] - df["C_20_50"]
    df["BD_layer_delta"] = df["BD_0_20"] - df["BD_20_50"]

    # Water Holding Capacity proxy from texture lookup, adjusted by bulk density
    _whc_lookup_lower = {k.lower(): v for k, v in WHC_LOOKUP_SAXTON_RAWLS_2006_OM2P5.items()}
    df["WHC_proxy"] = (
        df["TX_0_20_cl"]
        .astype(str)
        .str.lower()
        .map(_whc_lookup_lower)
        .fillna(WHC_LOOKUP_SAXTON_RAWLS_2006_OM2P5["Sandy loam"])
    )

    # Adjust WHC by bulk density (inverse relationship, reference BD = 1.3 g/cm³)
    bd_factor = np.where(df["BD_0_20"] > 0, 1.3 / df["BD_0_20"], 1.0)
    df["WHC_proxy"] = df["WHC_proxy"] * bd_factor

    # Aridity index (guard against MAP=0)
    df["aridity_index"] = np.where(df["MAP"] > 0, df["CMD"] / df["MAP"], np.nan)

    return df


def apply_log_transforms(df: pd.DataFrame, log_transform_features: List[str]) -> pd.DataFrame:
    """Apply log(x + 1) transform to skewed features."""
    df = df.copy()
    for col in log_transform_features:
        if col in df.columns:
            df[col] = np.log1p(np.maximum(df[col], 0))
    return df


def remove_yield_outliers(
    df: pd.DataFrame,
    target_col: str = "Yld_ton_ha",
    iqr_multiplier: float = 3.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove yield outliers beyond IQR threshold."""
    if target_col not in df.columns:
        warnings.warn(f"Target column '{target_col}' not found; skipping outlier removal")
        return df, pd.Series([False] * len(df), index=df.index)

    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    outlier_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)

    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        log.info(
            f"Removing {n_outliers} yield outliers (< {lower_bound:.2f} or > {upper_bound:.2f} t/ha)"
        )

    return df[~outlier_mask].copy(), outlier_mask


def fit_scaler(df: pd.DataFrame, continuous_features: List[str]) -> StandardScaler:
    """Fit StandardScaler on continuous features."""
    available_features = [f for f in continuous_features if f in df.columns]
    if len(available_features) < len(continuous_features):
        missing = set(continuous_features) - set(available_features)
        warnings.warn(f"Missing continuous features: {missing}")
    scaler = StandardScaler()
    scaler.fit(df[available_features])
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    continuous_features: List[str],
) -> pd.DataFrame:
    """Apply fitted StandardScaler to continuous features."""
    df = df.copy()
    available_features = [f for f in continuous_features if f in df.columns]
    df[available_features] = scaler.transform(df[available_features])
    return df


def fit_label_encoders(
    df: pd.DataFrame,
    categorical_features: List[str],
) -> Dict[str, LabelEncoder]:
    """Fit LabelEncoders for categorical features."""
    encoders = {}
    for col in categorical_features:
        if col not in df.columns:
            warnings.warn(f"Categorical feature '{col}' not found; skipping")
            continue
        encoder = LabelEncoder()
        encoder.fit(df[col].dropna())
        encoders[col] = encoder
        log.info(f"  {col}: {len(encoder.classes_)} classes")
    return encoders


def apply_label_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Apply fitted LabelEncoders to categorical features."""
    df = df.copy()
    for col in categorical_features:
        if col not in df.columns:
            continue
        if col not in encoders:
            warnings.warn(f"No encoder found for '{col}'; skipping")
            continue
        encoder = encoders[col]
        df[col] = df[col].apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )
    return df


def calculate_spatial_splits(
    df: pd.DataFrame,
    block_size_km: float = 50.0,
    n_splits: int = 7,
    lat_col: str = "lat",
    lon_col: str = "lon",
    name_loc_col: str = "name_loc",
    save_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Calculate spatial cross-validation splits using a grid-based blocking approach."""
    log.info(f"Calculating spatial splits with block size {block_size_km}km and {n_splits} folds")

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        log.info(f"Filtered to training split: {len(train_df)} samples")
    else:
        train_df = df.copy()
        log.info(f"No 'split' column found; using all {len(train_df)} samples for spatial splits")

    # Approx conversion: 1 deg ~ 111 km
    block_size_deg = block_size_km / 111.0
    train_df["lat_grid"] = np.floor(train_df[lat_col] / block_size_deg)
    train_df["lon_grid"] = np.floor(train_df[lon_col] / block_size_deg)
    train_df["block_id"] = (
        train_df["lat_grid"].astype(str) + "_" + train_df["lon_grid"].astype(str)
    )

    unique_blocks = train_df["block_id"].unique()
    log.info(f"Created {len(unique_blocks)} spatial blocks")

    # Greedy bin packing: assign blocks largest-first to the smallest fold
    block_counts = train_df["block_id"].value_counts().sort_values(ascending=False)
    fold_samples = [0] * n_splits
    fold_block_ids: List[List[str]] = [[] for _ in range(n_splits)]

    for block_id, count in block_counts.items():
        smallest_fold = int(np.argmin(fold_samples))
        fold_samples[smallest_fold] += count
        fold_block_ids[smallest_fold].append(block_id)

    block_to_names = train_df.groupby("block_id")[name_loc_col].unique().to_dict()

    splits: Dict[str, Any] = {}
    for fold in range(n_splits):
        val_names = [
            name
            for bid in fold_block_ids[fold]
            for name in block_to_names[bid].tolist()
        ]
        train_names = [
            name
            for f in range(n_splits)
            if f != fold
            for bid in fold_block_ids[f]
            for name in block_to_names[bid].tolist()
        ]
        splits[f"fold_{fold}"] = {"train": train_names, "val": val_names}
        log.info(
            f"  Fold {fold}: {len(train_names)} train locations, {len(val_names)} val locations "
            f"({fold_samples[fold]} samples)"
        )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(splits, save_path)
        log.info(f"Saved spatial splits to {save_path}")

    return splits


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(
    source_csv: str,
    out_csv: str,
    out_parquet: str | None = None,
    spatial_splits: bool = False,
    countries: List[str] | None = None,
    years: List[int] | None = None,
    exclude_countries: List[str] | None = None,
    exclude_years: List[int] | None = None,
) -> pd.DataFrame:
    """Preprocessing workflow for the crop yield Africa dataset."""
    data_path = Path(source_csv)
    out_csv_path = Path(out_csv)
    data_dir = out_csv_path.parent

    scaler_path = data_dir / f"fitted_scaler_{MODEL_READY_DATA_NAME}.pkl"
    encoders_path = data_dir / f"label_encoders_{MODEL_READY_DATA_NAME}.pkl"
    spatial_split_path = (
        data_dir
        / "splits"
        / (
            f"split_spatial_{SPATIAL_SPLIT_N_SPLITS}_folds_with_"
            f"{SPATIAL_SPLIT_BLOCK_SIZE_KM}km_blocks_{MODEL_READY_DATA_NAME}.pth"
        )
    )

    log.info("Starting preprocessing pipeline...")
    log.info(f"Input: {data_path}")

    # Load raw data
    raw_df = pd.read_csv(data_path)
    n_raw = len(raw_df)
    log.info(f"Loaded {n_raw} rows from {data_path}")

    # Filter by country and year before any other processing
    df = raw_df.copy()
    if countries is not None:
        before = len(df)
        df = df[df["Country"].isin(countries)].copy()
        log.info(
            f"Country filter ({', '.join(sorted(countries))}): "
            f"kept {len(df)}, excluded {before - len(df)}"
        )
    if years is not None:
        before = len(df)
        df = df[df["Year"].isin(years)].copy()
        log.info(
            f"Year filter ({', '.join(str(y) for y in sorted(years))}): "
            f"kept {len(df)}, excluded {before - len(df)}"
        )
    if exclude_countries is not None:
        before = len(df)
        df = df[~df["Country"].isin(exclude_countries)].copy()
        log.info(
            f"Exclude countries ({', '.join(sorted(exclude_countries))}): "
            f"kept {len(df)}, excluded {before - len(df)}"
        )
    if exclude_years is not None:
        before = len(df)
        df = df[~df["Year"].isin(exclude_years)].copy()
        log.info(
            f"Exclude years ({', '.join(str(y) for y in sorted(exclude_years))}): "
            f"kept {len(df)}, excluded {before - len(df)}"
        )
    n_after_filter = len(df)

    # Determine train mask on the filtered data
    train_mask = df["Country"].isin(TRAIN_COUNTRIES)
    log.info(
        f"Train mask: {train_mask.sum()} rows ({', '.join(TRAIN_COUNTRIES)}) out of {n_after_filter} total"
    )

    # Compute derived features
    log.info("Computing derived features...")
    df = compute_derived_features(df)

    # Remove yield outliers
    log.info("Removing yield outliers...")
    df, outlier_mask = remove_yield_outliers(df, TARGET_COLUMNS[0], iqr_multiplier=3.0)
    # Re-align train_mask after outlier removal
    train_mask = train_mask[df.index]

    # Apply log transforms
    log.info("Applying log transforms...")
    df = apply_log_transforms(df, LOG_TRANSFORM_FEATURES)

    train_df = df[train_mask]
    log.info(f"Training set size: {len(train_df)} samples")

    # Fit transformers on train split only
    log.info("Fitting StandardScaler on train split...")
    scaler = fit_scaler(train_df, CONTINUOUS_FEATURES)

    log.info("Fitting LabelEncoders on train split...")
    all_categorical = TABULAR_CATEGORICAL_FEATURES + AUX_FEATURES
    encoders = fit_label_encoders(train_df, all_categorical)

    # Apply transformations to full dataset
    log.info("Applying transformations to full dataset...")
    df = apply_scaler(df, scaler, CONTINUOUS_FEATURES)
    df = apply_label_encoders(df, encoders, all_categorical)

    # Save transformers
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    log.info(f"Saved scaler to {scaler_path}")

    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, encoders_path)
    log.info(f"Saved encoders to {encoders_path}")

    # Validation checks
    log.info("Validation checks:")
    derived_cols = ["CN_ratio", "C_layer_delta", "BD_layer_delta", "WHC_proxy", "aridity_index"]
    for col in derived_cols:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            n_inf = np.isinf(df[col]).sum()
            if n_nan > 0 or n_inf > 0:
                warnings.warn(f"  {col}: {n_nan} NaN, {n_inf} Inf values")
            else:
                log.info(f"  {col}: no NaN or Inf")

    for col in all_categorical:
        if col in df.columns and col in encoders:
            n_classes = len(encoders[col].classes_)
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < 0 or max_val >= n_classes:
                warnings.warn(
                    f"  {col}: indices [{min_val}, {max_val}] out of range [0, {n_classes-1}]"
                )

    # Apply canonical column rename (feat_/aux_/target_ prefixes, lowercase meta)
    rename_map = build_column_rename_map(
        continuous_features=CONTINUOUS_FEATURES,
        tabular_categorical_features=TABULAR_CATEGORICAL_FEATURES,
        aux_features=AUX_FEATURES,
        target_columns=TARGET_COLUMNS,
        name_loc_column=NAME_LOC_COLUMN,
        metadata_columns=METADATA_COLUMNS,
    )
    df = df.rename(columns=rename_map)

    # Prefix name_loc IDs with country name
    if "name_loc" in df.columns and "country" in df.columns:
        df["name_loc"] = df["country"].astype(str).str.upper() + "_" + df["name_loc"].astype(str)
        log.info("Prefixed name_loc IDs with country names")
        n_duplicates = df["name_loc"].duplicated().sum()
        if n_duplicates > 0:
            warnings.warn(f"Found {n_duplicates} duplicate name_loc IDs after prefixing")
        else:
            log.info(f"  No duplicates in name_loc ({df['name_loc'].nunique()} unique IDs)")

    # Convert location_accuracy from text to numerical values
    if "location_accuracy" in df.columns:
        accuracy_map = {
            "high location accuracy": 1,
            "medium location accuracy": 2,
            "low location accuracy": 3,
        }
        df["location_accuracy"] = df["location_accuracy"].str.lower().map(accuracy_map)

    # Keep scaler metadata in sync with new column names
    if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
        scaler.feature_names_in_ = np.array(
            [rename_map.get(n, n) for n in scaler.feature_names_in_]
        )
    encoders = {rename_map.get(k, k): v for k, v in encoders.items()}

    # Calculate and save spatial splits (optional)
    if spatial_splits:
        calculate_spatial_splits(
            df=df,
            block_size_km=SPATIAL_SPLIT_BLOCK_SIZE_KM,
            n_splits=SPATIAL_SPLIT_N_SPLITS,
            save_path=spatial_split_path,
        )
    else:
        log.info("Skipping spatial split calculation (use --spatial_splits to enable)")

    # Save outputs
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, float_format="%.7f")
    log.info(f"Saved CSV to {out_csv_path}")

    if out_parquet is not None:
        out_parquet_path = Path(out_parquet)
        out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet_path, index=False)
        log.info(f"Saved Parquet to {out_parquet_path}")

    log.info("=== Summary ===")
    log.info(f"  Raw rows loaded: {n_raw}")
    log.info(f"  Rows excluded by country/year filter: {n_raw - n_after_filter}")
    log.info(f"  Rows in output: {len(df)}")
    log.info(f"  Continuous features: {len(CONTINUOUS_FEATURES)}")
    log.info(f"  Tabular categorical features (feat_): {len(TABULAR_CATEGORICAL_FEATURES)}")
    log.info(f"  Aux/caption features (aux_): {len(AUX_FEATURES)}")
    log.info(
        f"  Yield range: {df['target_yld_ton_ha'].min():.2f} - {df['target_yld_ton_ha'].max():.2f} t/ha"
    )
    log.info(f"  Mean yield: {df['target_yld_ton_ha'].mean():.2f} t/ha")
    log.info("  Records per country and year:")
    counts = df.groupby(["country", "year"]).size().unstack(fill_value=0)
    for country, row in counts.iterrows():
        year_counts = [f"{year}: {count}" for year, count in row.items() if count > 0]
        log.info(f"    {country}: {', '.join(year_counts)}")

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ap = argparse.ArgumentParser(
        description="Build model-ready CSV/Parquet for the crop yield Africa use case."
    )
    ap.add_argument(
        "--source_csv",
        required=True,
        help="Path to the raw input CSV (e.g. data/yield_africa/yield_africa_v20260218.csv)",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path for the output model-ready CSV (e.g. data/yield_africa/model_ready_yield-africa.csv)",
    )
    ap.add_argument(
        "--out_parquet",
        default=None,
        help="Optional path for an additional Parquet output.",
    )
    ap.add_argument(
        "--spatial_splits",
        action="store_true",
        default=False,
        help="Calculate and save spatial cross-validation splits (default: off).",
    )
    ap.add_argument(
        "--countries",
        nargs="+",
        default=None,
        metavar="CODE",
        help="Country codes to include (e.g. --countries ETH KEN TAN). Default: all countries.",
    )
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Years to include (e.g. --years 2018 2019 2020). Default: all years.",
    )
    ap.add_argument(
        "--exclude_countries",
        nargs="+",
        default=None,
        metavar="CODE",
        help="Country codes to exclude (e.g. --exclude_countries MOZ ZIM).",
    )
    ap.add_argument(
        "--exclude_years",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Years to exclude (e.g. --exclude_years 2015 2016).",
    )
    args = ap.parse_args()
    main(
        args.source_csv,
        args.out_csv,
        args.out_parquet,
        args.spatial_splits,
        args.countries,
        args.years,
        args.exclude_countries,
        args.exclude_years,
    )