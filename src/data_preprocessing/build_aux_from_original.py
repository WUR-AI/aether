#!/usr/bin/env python3
"""
Build aux_* columns for the Guatemala LST captioning, by joining:
  - RAW numbers  (model_ready_heat_guatemala.csv, NON-standardized)  -> concept theta_k
  - EXPERT words (Heat_Guatemala.csv, the original legend)           -> caption text

Join key: BLOCK_ID = int(name_loc[5:])  (verified 1:1, lat diff 0, LST diff <0.005).

Output: a copy of the raw model-ready CSV with aux_* columns appended, ready for
the alignment datamodule (which selects columns by the regex ^aux_).
"""
import argparse
import re

import pandas as pd


def cls(s):
    """Take the human class from a legend string ('<0.5 NDVI greenness : high' -> 'high')."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    return s.split(":")[-1].strip() if ":" in s else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="model_ready_heat_guatemala.csv (RAW, not _in_)")
    ap.add_argument("--legend", required=True, help="Heat_Guatemala.csv (original legend)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw, low_memory=False)
    leg = pd.read_csv(args.legend, encoding="cp1252", low_memory=False)
    raw["BLOCK_ID"] = raw["name_loc"].str.replace("heat_", "", regex=False).astype(int)
    df = raw.merge(leg, on="BLOCK_ID", how="left")

    new = {}
    # numeric aux (raw scale) -> concept ground truth / theta_k
    num = {
        "aux_ndvi_mean": "feat_ndvi_mean2022", "aux_ndwi_mean": "feat_ndwi_mean2022",
        "aux_forest_cover_perc": "feat_forcov_meanperc", "aux_tree_cover_perc": "feat_troptreecovperc",
        "aux_builtup_age_years": "feat_bua_gaia_age_mean", "aux_slope_perc": "feat_dem5mslopeperc_mean",
        "aux_socioeconomic": "feat_estrato_s", "aux_lst": "target_lst",
    }
    for a, f in num.items():
        new[a] = df[f]
    if "feat_measurement_month" in df.columns:          # present only if you kept raw month
        new["aux_month"] = df["feat_measurement_month"]

    # word aux (authoritative expert legend) -> caption text
    new["aux_ndvi_label"]    = df["NDVI_mean2022"].map(cls).map(lambda x: f"{x} vegetation greenness" if x else x)
    new["aux_ndwi_label"]    = df["NDWI_mean2022"].map(cls).map(lambda x: x if x and "stress" in x else (f"{x} drought stress" if x else x))
    new["aux_socio_label"]   = df["SocioEconomicQuality"].map(cls).map(lambda x: f"{x} socioeconomic quality" if x else x)
    new["aux_slope_label"]   = df["DEM5m_Slope%"].map(cls).map(lambda x: x.lower() if x else x)
    new["aux_forest_label"]  = df["Hansen_ForestCover_meanPerc"].map(cls)
    new["aux_age_label"]     = df["BUA_GAIA_Age_Mean"].map(cls)
    new["aux_height_label"]  = df["CopenicusMSZ_BuildingHeightM"].map(cls)   # words OK; raw number is unit-broken
    new["aux_density_label"] = df["PopulationDensityPerKm2"].map(cls)        # urban-form label
    new["aux_lst_label"]     = df["LST_mean_predictor_Classified"].map(cls)
    new["aux_landuse"]       = df["BlockMAGADominantLanduse"].astype(str).str.strip().str.lower()
    new["aux_blocktype"]     = df["BlockType"].astype(str).str.strip().str.lower()
    new["aux_interzone"]     = df["IntrZon"].astype(str).str.strip()

    out = pd.concat([raw.drop(columns=["BLOCK_ID"]), pd.DataFrame(new, index=raw.index)], axis=1)
    out.to_csv(args.out, index=False)
    print(f"wrote {out.shape[0]} rows, {sum(k.startswith('aux_') for k in out.columns)} aux columns -> {args.out}")
    print("nulls in aux columns:", int(out[[c for c in out.columns if c.startswith('aux_')]].isna().sum().sum()))


if __name__ == "__main__":
    main()
