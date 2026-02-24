"""
Build model-ready CSV for the Heat Guatemala use case (data/heat_guatemala/model_ready_heat_guatemala.csv).
"""

import argparse
import re

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Continuous / numeric columns  →  kept as float feat_* columns
# -----------------------------------------------------------------------
NUMERIC_COLS = [
    "AREA_M2",
    "AREA_Ha",
    "UTM_areaM2",
    "UTM_perimeterMeter",
    # Vegetation & water indices (continuous floats)
    "NDVI_mean2022",
    "NDVI_min2022",
    "MODIS_NDVIchange_20002020",
    "NDWI_mean2022",
    "NDWI_minimum2022",
    # Urban structure
    "CopenicusMSZ_BuildingHeightM",
    "BUA_GAIA_Age_Mean",
    # Socio-demographic
    "PopulationDensityPerKm2",
    "SocioEconomicQuality",
    # Terrain
    "DEM5m_Slope%",
    "DEM5m_TerrainRuggednessIndex_MeanTRI",
    "DEM5m_MeanAspect",
    "DEM5m_TopographicPositionIndex_MeanTPI",
    # Forest cover
    "Hansen_ForestCover_sumHa",
    "Hansen_ForestCover_meanPerc",
    "HansenLoss_Ha",
]

# -----------------------------------------------------------------------
# Genuinely categorical / nominal columns  →  one-hot encoded feat_* columns
# -----------------------------------------------------------------------
CATEGORICAL_COLS = [
    "BlockType",
    "BlockTypeIndustry",
    "BlockMAGADominantLanduse",
    "IntrZon",
    "DISTRITOS",
    "ZONAM",
]

# Columns where >this fraction of values are NaN → drop column entirely
NAN_DROP_THRESHOLD = 0.30


def clean_token(x: str) -> str:
    """Make a string safe for use as a column name."""
    x = str(x).strip()
    x = re.sub(r"[^\w]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x if x else "NA"


def main(source_csv: str, out_csv: str, drop_zero_lst: bool = True) -> None:
    df = pd.read_csv(source_csv, encoding="latin-1")
    print(f"Loaded:     {source_csv}  →  {df.shape[0]} rows, {df.shape[1]} cols")

    # ------------------------------------------------------------------ #
    # 1. Clean target: drop zero and NaN LST rows                         #
    # ------------------------------------------------------------------ #
    target_col = "LST_°C_mean_predictor"

    if drop_zero_lst:
        n_before = len(df)
        df = df[df[target_col] != 0].copy().reset_index(drop=True)
        print(f"Dropped {n_before - len(df)} rows with LST == 0")

    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with NaN LST target")
    else:
        print("No NaN LST targets found — good.")

    # ------------------------------------------------------------------ #
    # 2. Build output skeleton                                            #
    # ------------------------------------------------------------------ #
    out = pd.DataFrame({
        "name_loc":   [f"heat_{i:06d}" for i in range(len(df))],
        "lat":        df["LAT"].astype(float),
        "lon":        df["LONG"].astype(float),
        "target_lst": df[target_col].astype(float),
    })

    # Verify target is clean
    assert out["target_lst"].isna().sum() == 0, "BUG: NaN in target after cleaning"
    print(f"Target stats:  mean={out['target_lst'].mean():.2f}  "
          f"std={out['target_lst'].std():.2f}  "
          f"min={out['target_lst'].min():.2f}  "
          f"max={out['target_lst'].max():.2f}")

    # ------------------------------------------------------------------ #
    # 3. Numeric features — impute or drop                                #
    # ------------------------------------------------------------------ #
    numeric_feat_cols = []
    dropped_cols = []

    for c in NUMERIC_COLS:
        if c not in df.columns:
            print(f"  [WARN] numeric column not found, skipping: {c}")
            continue
        col_name = f"feat_{clean_token(c).lower()}"
        series = pd.to_numeric(df[c], errors="coerce").astype(float)
        nan_frac = series.isna().sum() / len(series)

        if nan_frac > NAN_DROP_THRESHOLD:
            print(f"  [DROP] {col_name}: {nan_frac:.0%} NaN — exceeds threshold, dropped")
            dropped_cols.append(col_name)
            continue

        if nan_frac > 0:
            col_mean = series.mean()
            print(f"  [IMPUTE] {col_name}: {nan_frac:.1%} NaN → filled with mean ({col_mean:.4f})")
            series = series.fillna(col_mean)

        out[col_name] = series
        numeric_feat_cols.append(col_name)

    # ------------------------------------------------------------------ #
    # 4. Categorical features (one-hot)                                   #
    # ------------------------------------------------------------------ #
    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            print(f"  [WARN] categorical column not found, skipping: {c}")
            continue
        cats = df[c].astype(str).fillna("NA").map(clean_token)
        prefix = f"feat_{clean_token(c).lower()}"
        dummies = pd.get_dummies(cats, prefix=prefix, prefix_sep="__")
        out = pd.concat([out, dummies.astype(np.float32)], axis=1)

    # ------------------------------------------------------------------ #
    # 5. Final NaN check — should be zero                                 #
    # ------------------------------------------------------------------ #
    total_nan = out.isna().sum().sum()
    if total_nan > 0:
        print("\n[ERROR] NaN values remain in output:")
        print(out.isna().sum()[out.isna().sum() > 0])
        raise ValueError(f"{total_nan} NaN values remain — fix before training.")
    else:
        print("\nNaN check passed — output is clean.")

    # ------------------------------------------------------------------ #
    # 6. Save and report                                                  #
    # ------------------------------------------------------------------ #
    out.to_csv(out_csv, index=False)

    feat_cols = [c for c in out.columns if c.startswith("feat_")]
    print(f"\nWrote:      {out_csv}")
    print(f"Shape:      {out.shape}")
    print(f"tabular_dim (feat_* columns): {len(feat_cols)}")
    print(f"  numeric features kept:   {len(numeric_feat_cols)}")
    print(f"  numeric features dropped:{len(dropped_cols)}")
    print(f"  one-hot features:        {len(feat_cols) - len(numeric_feat_cols)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_csv", required=True)
    ap.add_argument("--out_csv",    required=True)
    ap.add_argument("--drop_zero_lst", type=lambda x: x.lower() != "false", default=True)
    args = ap.parse_args()
    main(args.source_csv, args.out_csv, args.drop_zero_lst)