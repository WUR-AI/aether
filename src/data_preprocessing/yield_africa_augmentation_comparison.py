"""Compare yield Africa CSV augmentations: base vs. base+NDVI vs. base+AgERA5.

Evaluates each augmentation's contribution to crop yield prediction accuracy,
diagnoses data quality differences, and produces Hydra config recommendations.
Run with any combination of augmented CSVs — only --base_csv is required.

Usage:
    # All three datasets
    python src/data_preprocessing/yield_africa_augmentation_comparison.py \\
        --base_csv   /Volumes/.../yield_africa/model_ready_yield_africa_base.csv \\
        --ndvi_csv   /Volumes/.../yield_africa/model_ready_yield_africa_ndvi.csv \\
        --agera5_csv /Volumes/.../yield_africa/model_ready_yield_africa_agera5.csv

    # Base vs AgERA5 only (NDVI not yet ready)
    python src/data_preprocessing/yield_africa_augmentation_comparison.py \\
        --base_csv   /Volumes/.../yield_africa/model_ready_yield_africa_base.csv \\
        --agera5_csv /Volumes/.../yield_africa/model_ready_yield_africa_agera5.csv

Optional arguments:
    --out_dir           Output directory for figures (default: data/yield_africa/analysis_augmentation)
    --model             Which model(s) to run: rf, xgb, or both (default: both)
    --n_trees           Number of RF trees (default: 300)
    --xgb_n_estimators  Number of XGBoost estimators (default: 300)
    --seed              Random seed (default: 42)
    --merged_csv        Path to model_ready_yield_africa_merged.csv (optional)
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column sets
# ---------------------------------------------------------------------------

TARGET_COL = "target_yld_ton_ha"
COUNTRY_COL = "country"
YEAR_COL = "year"

NDVI_MONTHLY_COLS = [f"feat_ndvi_month_{m}" for m in range(1, 13)]
NDVI_SEASONAL_COLS = [
    "feat_ndvi_mean_djf", "feat_ndvi_mean_mam",
    "feat_ndvi_mean_jja", "feat_ndvi_mean_son", "feat_ndvi_mean_grow",
]
NDVI_COLS = NDVI_MONTHLY_COLS + NDVI_SEASONAL_COLS

_AGERA5_VARS = ["tmax", "tmin", "tavg", "vp", "ws", "prec", "rad", "snow"]
_AGERA5_SEASONS = ["mam", "jja", "son", "djf", "grow"]
AGERA5_COLS = (
    [f"feat_agera5_{v}_{s}" for v in _AGERA5_VARS for s in _AGERA5_SEASONS]
    + ["feat_agera5_gdd10_grow", "feat_agera5_wetdays_grow"]
)

# Known augmentation column sets, in display order
# "key_cols" is the subset used for the --complete_only completeness check:
#   NDVI   → seasonal aggregates only (monthly cols may be partially NaN even for covered years,
#             while the seasonal means are computed from whatever months are available)
#   AgERA5 → uses the "agera5_fetched" sentinel column (1 = real API data, 0 = median-imputed)
#             written by yield_africa_augment_agera5.py before it fills NaNs with medians.
#             Falling back to NaN-checking the feature cols would never catch imputed rows
#             because the augment script writes medians rather than NaN to the CSV.
MERGED_COLS = NDVI_COLS + AGERA5_COLS

AUGMENTATION_DEFS = {
    "NDVI":   {"cols": NDVI_COLS,   "key_cols": NDVI_SEASONAL_COLS, "color": "#6ACC65", "label": "Base+NDVI"},
    "AgERA5": {"cols": AGERA5_COLS, "key_cols": AGERA5_COLS, "sentinel_col": "agera5_fetched",
               "color": "#E07B54", "label": "Base+AgERA5"},
    "Merged": {"cols": MERGED_COLS, "key_cols": NDVI_SEASONAL_COLS + AGERA5_COLS,
               "sentinel_col": "agera5_fetched", "color": "#9B59B6", "label": "Merged"},
}

PALETTE = {
    "Base":        "#4878CF",
    "Base+NDVI":   "#6ACC65",
    "Base+AgERA5": "#E07B54",
    "Merged": "#9B59B6",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title: str = "", width: int = 72) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


def _pct(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.1f}%)"


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]


def _build_X_y(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feat_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    return X, df[TARGET_COL].copy()


def _aug_name(label: str) -> str:
    """Return the augmentation key ('NDVI', 'AgERA5') for a dataset label, or ''."""
    for key, defn in AUGMENTATION_DEFS.items():
        if defn["label"] == label:
            return key
    return ""


def _apply_complete_filter(
    aligned: dict[str, pd.DataFrame],
    complete_only: bool,
) -> dict[str, pd.DataFrame]:
    """When complete_only=True, restrict all datasets to rows where every augmented
    dataset has non-null values in its key augmentation columns.

    The completeness criterion is evaluated per augmentation using its "key_cols"
    (seasonal aggregates for NDVI; all columns for AgERA5).  Rows that pass for all
    augmentations present in *aligned* are kept; the Base dataset is filtered to the
    same (name_loc, year) pairs so the comparison remains fair.
    """
    if not complete_only:
        return aligned

    sep("Completeness filter  (--complete_only)")

    # Build a set of (name_loc, year) keys that are complete in *every* augmentation.
    base_df = aligned["Base"]
    complete_keys: set[tuple] = set(zip(base_df["name_loc"], base_df[YEAR_COL]))

    for label, df in aligned.items():
        if label == "Base":
            continue
        aug_key = _aug_name(label)
        aug_def = AUGMENTATION_DEFS.get(aug_key, {})
        sentinel_col = aug_def.get("sentinel_col", "")

        if sentinel_col and sentinel_col in df.columns:
            # Prefer the explicit sentinel column over NaN-checking: the augment script
            # may have already filled NaNs with medians before writing the CSV.
            complete_mask = df[sentinel_col] == 1
            criterion = f"sentinel column '{sentinel_col}' == 1"
        else:
            if sentinel_col and sentinel_col not in df.columns:
                print(
                    f"  [{label}]  WARNING: sentinel column '{sentinel_col}' not found in CSV.\n"
                    f"  The augment script fills NaNs with medians before saving, so NaN-checking\n"
                    f"  cannot distinguish real data from imputed rows for {aug_key}.\n"
                    f"  Re-run yield_africa_augment_{aug_key.lower()}.py to regenerate the CSV\n"
                    f"  with the sentinel column, then retry --complete_only."
                )
            key_cols = [c for c in aug_def.get("key_cols", aug_def.get("cols", [])) if c in df.columns]
            if not key_cols:
                print(f"  [{label}] No key augmentation columns or sentinel found — skipping completeness filter.")
                continue
            complete_mask = df[key_cols].notna().all(axis=1)
            criterion = f"all {len(key_cols)} key {aug_key} columns non-null (NaN-check fallback)"

        aug_keys = set(zip(df.loc[complete_mask, "name_loc"], df.loc[complete_mask, YEAR_COL]))
        n_complete = len(aug_keys)
        n_total = len(df)
        print(
            f"  [{label}]  {n_complete:,} / {n_total:,} rows ({100 * n_complete / n_total:.1f}%) "
            f"pass completeness check ({criterion})."
        )
        complete_keys &= aug_keys

    n_retained = len(complete_keys)
    if n_retained == 0:
        print(
            "\n  WARNING: No rows survive the completeness filter across all augmentations.\n"
            "  Returning unfiltered data — consider running with only one augmentation CSV,\n"
            "  or wait until more download data is available."
        )
        return aligned

    print(f"\n  Rows retained after intersection across all augmentations: {n_retained:,}")

    filtered: dict[str, pd.DataFrame] = {}
    for label, df in aligned.items():
        pair = list(zip(df["name_loc"], df[YEAR_COL]))
        mask = pd.Series(pair).isin(complete_keys).values
        filtered[label] = df[mask].reset_index(drop=True)
        print(f"  [{label}]  {len(filtered[label]):,} rows after filter")

    return filtered


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_datasets(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load each CSV and return {label: df}. 'Base' must be present."""
    sep("Loading CSVs")
    datasets: dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        df = pd.read_csv(path)
        datasets[label] = df
        feats = len(_feat_cols(df))
        print(f"  {label:12s}: {df.shape[0]:,} rows × {df.shape[1]} cols  "
              f"({feats} feat_*)  [{path.name}]")
    return datasets


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage_report(datasets: dict[str, pd.DataFrame]) -> None:
    sep("Row coverage vs. Base")
    base = datasets["Base"]
    base_locs = set(base["name_loc"])

    for label, df in datasets.items():
        if label == "Base":
            continue
        locs = set(df["name_loc"])
        common = base_locs & locs
        only_base = base_locs - locs
        only_aug = locs - base_locs
        print(f"\n  [{label}]")
        print(f"    In both     : {len(common):,}")
        print(f"    Base-only   : {len(only_base):,}")
        print(f"    {label}-only : {len(only_aug):,}")
        if only_base:
            cc = base.loc[base["name_loc"].isin(only_base), COUNTRY_COL].value_counts()
            print(f"    Base-only by country: " + ", ".join(f"{c}={n}" for c, n in cc.items()))

    # Country × dataset count table
    sep("Sample counts by country")
    all_countries = sorted(base[COUNTRY_COL].unique())
    header = f"  {'Country':8s}" + "".join(f"  {lb:>12s}" for lb in datasets)
    print(header)
    print("  " + "─" * (10 + 14 * len(datasets)))
    for c in all_countries:
        row = f"  {c:8s}"
        for df in datasets.values():
            n = (df[COUNTRY_COL] == c).sum()
            row += f"  {n:>12,}"
        print(row)

    # Year × dataset count table
    sep("Sample counts by year")
    all_years = sorted(base[YEAR_COL].unique())
    print(header.replace("Country", "Year   "))
    print("  " + "─" * (10 + 14 * len(datasets)))
    for y in all_years:
        row = f"  {y:8d}"
        for df in datasets.values():
            n = (df[YEAR_COL] == y).sum()
            row += f"  {n:>12,}"
        print(row)


# ---------------------------------------------------------------------------
# Feature set comparison
# ---------------------------------------------------------------------------

def feature_set_report(datasets: dict[str, pd.DataFrame]) -> None:
    sep("Feature set comparison")
    base_feats = set(_feat_cols(datasets["Base"]))
    print(f"  Base: {len(base_feats)} feat_* columns")

    for label, df in datasets.items():
        if label == "Base":
            continue
        aug_feats = set(_feat_cols(df))
        added = sorted(aug_feats - base_feats)
        removed = sorted(base_feats - aug_feats)
        print(f"\n  [{label}]: {len(aug_feats)} feat_* total  "
              f"(+{len(added)} added, -{len(removed)} removed vs Base)")

        if added:
            total = len(df)
            print(f"  {'Column':40s}  {'Non-null':>10s}  {'NaN':>10s}  "
                  f"{'Min':>7s}  {'Max':>7s}  {'Mean':>7s}")
            print("  " + "─" * 88)
            for col in added:
                nn = df[col].notna().sum()
                mn = df[col].min() if nn > 0 else float("nan")
                mx = df[col].max() if nn > 0 else float("nan")
                mu = df[col].mean() if nn > 0 else float("nan")
                print(f"  {col:40s}  {_pct(nn, total):>10s}  "
                      f"{_pct(total - nn, total):>10s}  {mn:7.3f}  {mx:7.3f}  {mu:7.3f}")

        if removed:
            print(f"  Removed vs Base: {removed}")

    # Consistency check: shared features should be identical
    sep("Shared feature consistency check (1000-row sample)")
    base = datasets["Base"]
    sample_locs = list(set(base["name_loc"]))[:1000]
    base_s = base[base["name_loc"].isin(sample_locs)].set_index("name_loc")

    for label, df in datasets.items():
        if label == "Base":
            continue
        aug_s = df[df["name_loc"].isin(sample_locs)].set_index("name_loc")
        shared = [c for c in _feat_cols(base) if c in aug_s.columns]
        idx = base_s.index.intersection(aug_s.index)
        diffs = []
        for col in shared:
            d = (base_s.loc[idx, col] - aug_s.loc[idx, col]).abs().max()
            if d > 1e-6:
                diffs.append((col, d))
        if diffs:
            print(f"  [{label}] ⚠  {len(diffs)} shared features differ:")
            for col, d in sorted(diffs, key=lambda x: -x[1])[:5]:
                print(f"    {col:40s}  max|Δ|={d:.6f}")
        else:
            print(f"  [{label}] ✓  All {len(shared)} shared features are bit-identical.")


# ---------------------------------------------------------------------------
# RF: train + evaluate one dataset
# ---------------------------------------------------------------------------

def _run_rf(
    df: pd.DataFrame,
    feat_cols: list[str],
    n_trees: int,
    seed: int,
    label: str,
) -> dict:
    X, y = _build_X_y(df, feat_cols)
    groups = df[COUNTRY_COL].astype(str) + "_" + df[YEAR_COL].astype(str)

    print(f"\n  [{label}]  {X.shape[0]:,} samples × {X.shape[1]} features")

    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    rf_cv = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    cv_r2   = cross_val_score(rf_cv, X, y, cv=gss, groups=groups, scoring="r2")
    cv_mae  = cross_val_score(rf_cv, X, y, cv=gss, groups=groups,
                              scoring="neg_mean_absolute_error")
    cv_rmse = cross_val_score(rf_cv, X, y, cv=gss, groups=groups,
                              scoring="neg_root_mean_squared_error")
    print(f"    CV R²   : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}  (folds: {np.round(cv_r2,3)})")
    print(f"    CV MAE  : {-cv_mae.mean():.3f} ± {cv_mae.std():.3f} t/ha")
    print(f"    CV RMSE : {-cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} t/ha")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss1.split(X, y, groups=groups))
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = rf.predict(X.iloc[test_idx])
    y_test = y.iloc[test_idx]

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    base_mae = mean_absolute_error(y_test, np.full(len(y_test), y.iloc[train_idx].mean()))

    print(f"    Holdout R²  : {r2:.3f}")
    print(f"    Holdout MAE : {mae:.3f} t/ha  (baseline: {base_mae:.3f} t/ha, "
          f"skill: {1 - mae/base_mae:.3f})")
    print(f"    Holdout RMSE: {rmse:.3f} t/ha")

    return {
        "rf": rf, "model": rf, "X": X, "y": y,
        "train_idx": train_idx, "test_idx": test_idx,
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "cv_mae_mean": -cv_mae.mean(), "cv_rmse_mean": -cv_rmse.mean(),
        "holdout_r2": r2, "holdout_mae": mae, "holdout_rmse": rmse,
        "baseline_mae": base_mae,
    }


# ---------------------------------------------------------------------------
# RF comparison: all datasets on their common location set
# ---------------------------------------------------------------------------

def rf_comparison(
    datasets: dict[str, pd.DataFrame],
    n_trees: int,
    seed: int,
    out_dir: Path,
    complete_only: bool = False,
) -> dict[str, dict]:
    sep("Random Forest comparison  (spatial group CV, country×year blocks)")

    # Restrict to locations present in all datasets
    common_locs = set(datasets["Base"]["name_loc"])
    for df in datasets.values():
        common_locs &= set(df["name_loc"])
    print(f"\n  Evaluating on {len(common_locs):,} locations present in all CSVs.")

    aligned: dict[str, pd.DataFrame] = {
        label: df[df["name_loc"].isin(common_locs)].reset_index(drop=True)
        for label, df in datasets.items()
    }

    # Optionally restrict to rows that have real (non-imputed) augmentation data.
    aligned = _apply_complete_filter(aligned, complete_only)

    results: dict[str, dict] = {}
    for label, df in aligned.items():
        results[label] = _run_rf(df, _feat_cols(df), n_trees, seed, label)
        results[label]["df"] = df

    # Delta table vs Base
    sep("Performance delta vs. Base")
    m_base = results["Base"]
    print(f"  {'Dataset':14s}  {'CV R²':>8s}  {'ΔCV R²':>8s}  "
          f"{'HO R²':>8s}  {'ΔHO R²':>8s}  {'HO MAE':>8s}  {'ΔMAE':>8s}")
    print("  " + "─" * 76)
    for label, res in results.items():
        dr2_cv = res["cv_r2_mean"] - m_base["cv_r2_mean"]
        dr2_ho = res["holdout_r2"] - m_base["holdout_r2"]
        dmae   = res["holdout_mae"] - m_base["holdout_mae"]
        print(f"  {label:14s}  {res['cv_r2_mean']:8.3f}  {dr2_cv:+8.3f}  "
              f"{res['holdout_r2']:8.3f}  {dr2_ho:+8.3f}  "
              f"{res['holdout_mae']:8.3f}  {dmae:+8.3f}")

    # Bar chart
    labels = list(results.keys())
    colors = [PALETTE.get(lb, "#888888") for lb in labels]
    fig, axes = plt.subplots(1, 3, figsize=(5 * len(labels), 4))
    for ax, (metric_key, metric_label) in zip(
        axes,
        [("cv_r2_mean", "CV R²"), ("holdout_r2", "Holdout R²"), ("holdout_mae", "Holdout MAE (t/ha)")],
    ):
        vals = [results[lb][metric_key] for lb in labels]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        ax.set_title(metric_label, fontsize=10)
        ax.set_ylabel(metric_label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003 * (1 if metric_key != "holdout_mae" else -1),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Random Forest performance by dataset", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "rf_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_dir / 'rf_comparison.png'}")

    return results


# ---------------------------------------------------------------------------
# XGBoost: train + evaluate one dataset
# ---------------------------------------------------------------------------

def _run_xgb(
    df: pd.DataFrame,
    feat_cols: list[str],
    n_estimators: int,
    seed: int,
    label: str,
) -> dict:
    X, y = _build_X_y(df, feat_cols)
    groups = df[COUNTRY_COL].astype(str) + "_" + df[YEAR_COL].astype(str)

    print(f"\n  [{label}]  {X.shape[0]:,} samples × {X.shape[1]} features")

    model_cv = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    cv_r2   = cross_val_score(model_cv, X, y, cv=gss, groups=groups, scoring="r2")
    cv_mae  = cross_val_score(model_cv, X, y, cv=gss, groups=groups,
                              scoring="neg_mean_absolute_error")
    cv_rmse = cross_val_score(model_cv, X, y, cv=gss, groups=groups,
                              scoring="neg_root_mean_squared_error")
    print(f"    CV R²   : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}  (folds: {np.round(cv_r2,3)})")
    print(f"    CV MAE  : {-cv_mae.mean():.3f} ± {cv_mae.std():.3f} t/ha")
    print(f"    CV RMSE : {-cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} t/ha")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss1.split(X, y, groups=groups))
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = model.predict(X.iloc[test_idx])
    y_test = y.iloc[test_idx]

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    base_mae = mean_absolute_error(y_test, np.full(len(y_test), y.iloc[train_idx].mean()))

    print(f"    Holdout R²  : {r2:.3f}")
    print(f"    Holdout MAE : {mae:.3f} t/ha  (baseline: {base_mae:.3f} t/ha, "
          f"skill: {1 - mae/base_mae:.3f})")
    print(f"    Holdout RMSE: {rmse:.3f} t/ha")

    return {
        "model": model, "X": X, "y": y,
        "train_idx": train_idx, "test_idx": test_idx,
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "cv_mae_mean": -cv_mae.mean(), "cv_rmse_mean": -cv_rmse.mean(),
        "holdout_r2": r2, "holdout_mae": mae, "holdout_rmse": rmse,
        "baseline_mae": base_mae,
    }


# ---------------------------------------------------------------------------
# XGBoost comparison: all datasets on their common location set
# ---------------------------------------------------------------------------

def xgb_comparison(
    datasets: dict[str, pd.DataFrame],
    n_estimators: int,
    seed: int,
    out_dir: Path,
    complete_only: bool = False,
) -> dict[str, dict]:
    sep("XGBoost comparison  (spatial group CV, country×year blocks)")

    common_locs = set(datasets["Base"]["name_loc"])
    for df in datasets.values():
        common_locs &= set(df["name_loc"])
    print(f"\n  Evaluating on {len(common_locs):,} locations present in all CSVs.")

    aligned: dict[str, pd.DataFrame] = {
        label: df[df["name_loc"].isin(common_locs)].reset_index(drop=True)
        for label, df in datasets.items()
    }

    aligned = _apply_complete_filter(aligned, complete_only)

    results: dict[str, dict] = {}
    for label, df in aligned.items():
        results[label] = _run_xgb(df, _feat_cols(df), n_estimators, seed, label)
        results[label]["df"] = df

    # Delta table vs Base
    sep("XGBoost performance delta vs. Base")
    m_base = results["Base"]
    print(f"  {'Dataset':14s}  {'CV R²':>8s}  {'ΔCV R²':>8s}  "
          f"{'HO R²':>8s}  {'ΔHO R²':>8s}  {'HO MAE':>8s}  {'ΔMAE':>8s}")
    print("  " + "─" * 76)
    for label, res in results.items():
        dr2_cv = res["cv_r2_mean"] - m_base["cv_r2_mean"]
        dr2_ho = res["holdout_r2"] - m_base["holdout_r2"]
        dmae   = res["holdout_mae"] - m_base["holdout_mae"]
        print(f"  {label:14s}  {res['cv_r2_mean']:8.3f}  {dr2_cv:+8.3f}  "
              f"{res['holdout_r2']:8.3f}  {dr2_ho:+8.3f}  "
              f"{res['holdout_mae']:8.3f}  {dmae:+8.3f}")

    # Bar chart
    labels = list(results.keys())
    colors = [PALETTE.get(lb, "#888888") for lb in labels]
    fig, axes = plt.subplots(1, 3, figsize=(5 * len(labels), 4))
    for ax, (metric_key, metric_label) in zip(
        axes,
        [("cv_r2_mean", "CV R²"), ("holdout_r2", "Holdout R²"), ("holdout_mae", "Holdout MAE (t/ha)")],
    ):
        vals = [results[lb][metric_key] for lb in labels]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        ax.set_title(metric_label, fontsize=10)
        ax.set_ylabel(metric_label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003 * (1 if metric_key != "holdout_mae" else -1),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("XGBoost performance by dataset", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "xgb_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_dir / 'xgb_comparison.png'}")

    return results


# ---------------------------------------------------------------------------
# Feature importance per augmentation
# ---------------------------------------------------------------------------

def augmentation_importance_reports(
    results: dict[str, dict],
    out_dir: Path,
    seed: int,
) -> None:
    sep("Feature importance — augmentation contributions")

    base_feat_set = set(_feat_cols(results["Base"]["df"]))

    # Collect group-level perm importance share across all augmented datasets
    group_shares: dict[str, dict[str, float]] = {}  # {label: {group: pct}}

    for label, res in results.items():
        if label == "Base":
            continue
        aug_key = _aug_name(label)
        aug_def  = AUGMENTATION_DEFS.get(aug_key, {})
        aug_cols = set(aug_def.get("cols", [])) & set(res["X"].columns)
        aug_color = aug_def.get("color", "#888888")

        X_test = res["X"].iloc[res["test_idx"]]
        y_test = res["y"].iloc[res["test_idx"]]

        print(f"\n  [{label}]  Computing permutation importance (n_repeats=10)…")
        perm = permutation_importance(
            res["model"], X_test, y_test, n_repeats=10, random_state=seed, n_jobs=1
        )
        imp_df = pd.DataFrame({
            "mdi":       res["model"].feature_importances_,
            "perm_mean": perm.importances_mean,
            "perm_std":  perm.importances_std,
        }, index=res["X"].columns).sort_values("perm_mean", ascending=False)

        aug_imp  = imp_df[imp_df.index.isin(aug_cols)]
        base_imp = imp_df[~imp_df.index.isin(aug_cols)]
        total    = imp_df["perm_mean"].sum()
        aug_share  = 100 * aug_imp["perm_mean"].sum() / total if total > 0 else 0.0
        base_share = 100 * base_imp["perm_mean"].sum() / total if total > 0 else 0.0

        group_shares[label] = {"Base features": base_share, aug_key: aug_share}

        print(f"    Base features : {base_share:.1f}% of total perm importance")
        print(f"    {aug_key:9s} features : {aug_share:.1f}%")

        # Top-15
        print(f"\n    {'Feature':40s}  {'Perm':>7s}  {'±':>6s}  {'MDI':>7s}  Source")
        print("    " + "─" * 72)
        for feat, row in imp_df.head(15).iterrows():
            src = aug_key if feat in aug_cols else "Base"
            print(f"    {feat:40s}  {row.perm_mean:7.4f}  {row.perm_std:6.4f}  "
                  f"{row.mdi:7.4f}  {src}")

        # Augmented-feature-only ranking
        if not aug_imp.empty:
            print(f"\n    {aug_key} features ranked:")
            print(f"    {'Feature':40s}  {'Perm':>7s}  {'±':>6s}  {'Overall rank':>12s}")
            print("    " + "─" * 68)
            feat_list = list(imp_df.index)
            for feat, row in aug_imp.iterrows():
                rank = feat_list.index(feat) + 1
                print(f"    {feat:40s}  {row.perm_mean:7.4f}  {row.perm_std:6.4f}  "
                      f"{'#' + str(rank):>12s}")

        # Figure: top-20 importance (augmented features highlighted) + pie of group share
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        top20 = imp_df.head(20)
        bar_colors = [aug_color if f in aug_cols else "#4878CF" for f in top20.index]
        y_pos = range(len(top20))
        axes[0].barh(list(y_pos), top20["perm_mean"].values[::-1],
                     xerr=top20["perm_std"].values[::-1],
                     color=bar_colors[::-1], ecolor="gray", capsize=3)
        axes[0].set_yticks(list(y_pos))
        axes[0].set_yticklabels(list(top20.index[::-1]), fontsize=8)
        axes[0].set_xlabel("Mean permutation importance (R² decrease)")
        axes[0].set_title(f"Top-20 features [{label}]\n"
                          f"({aug_color} = {aug_key}, blue = base)")

        pie_labels = ["Base features", f"{aug_key} features"]
        pie_vals   = np.array([base_share, aug_share])
        pie_colors = ["#4878CF", aug_color]
        pie_vals_clipped = np.maximum(pie_vals, 0)
        if pie_vals_clipped.sum() > 0:
            axes[1].pie(pie_vals_clipped, labels=pie_labels, colors=pie_colors,
                        autopct="%1.1f%%", startangle=90)
        else:
            axes[1].text(0.5, 0.5, "Net-negative importance\n(augmentation features hurt model)",
                         ha="center", va="center", transform=axes[1].transAxes, fontsize=10)
            axes[1].axis("off")
        axes[1].set_title(f"Permutation importance share\n[{label}]")

        plt.tight_layout()
        fname = f"importance_{aug_key.lower()}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"\n    Saved: {out_dir / fname}")

    # Cross-augmentation comparison bar chart (if more than one augmentation)
    aug_labels = [lb for lb in results if lb != "Base"]
    if len(aug_labels) >= 2:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(aug_labels))
        w = 0.35
        base_shares = [group_shares[lb]["Base features"] for lb in aug_labels]
        aug_sh      = [group_shares[lb].get(_aug_name(lb), 0) for lb in aug_labels]
        aug_colors  = [AUGMENTATION_DEFS.get(_aug_name(lb), {}).get("color", "#888") for lb in aug_labels]
        ax.bar(x - w/2, base_shares, w, label="Base features", color="#4878CF")
        for i, (lb, sh, col) in enumerate(zip(aug_labels, aug_sh, aug_colors)):
            ax.bar(x[i] + w/2, sh, w, color=col, label=f"{_aug_name(lb)} features")
        ax.set_xticks(x)
        ax.set_xticklabels(aug_labels)
        ax.set_ylabel("% of total permutation importance")
        ax.set_title("Augmentation feature importance share")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "importance_comparison.png", dpi=150)
        plt.close(fig)
        print(f"\n  Saved: {out_dir / 'importance_comparison.png'}")


# ---------------------------------------------------------------------------
# PCA: dimensionality and independent signal
# ---------------------------------------------------------------------------

def pca_comparison(
    datasets: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    sep("PCA — dimensionality and independent signal per augmentation")

    common_locs = set(datasets["Base"]["name_loc"])
    for df in datasets.values():
        common_locs &= set(df["name_loc"])

    scaler = StandardScaler()

    def _pca_stats(df: pd.DataFrame, cols: list[str], label: str) -> np.ndarray:
        X = df[cols].copy()
        for c in X.columns:
            X[c] = X[c].fillna(X[c].median())
        Xs = scaler.fit_transform(X)
        pca = PCA().fit(Xs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for t in (0.80, 0.90, 0.95):
            n = int(np.searchsorted(cumvar, t) + 1)
            print(f"  [{label}]  {int(t*100)}% variance → {n:2d} / {len(cols)} components")
        ev = pca.explained_variance_ratio_
        ev = ev[ev > 0]
        eff_rank = int(np.round(np.exp(-np.sum(ev * np.log(ev)))))
        print(f"  [{label}]  Effective rank: {eff_rank}")
        return cumvar

    cumvars: dict[str, np.ndarray] = {}
    base_feats = _feat_cols(datasets["Base"])
    aligned_base = datasets["Base"][datasets["Base"]["name_loc"].isin(common_locs)]

    print()
    cumvars["Base"] = _pca_stats(aligned_base, base_feats, "Base")

    for label, df in datasets.items():
        if label == "Base":
            continue
        aug_key  = _aug_name(label)
        aug_cols_present = [c for c in AUGMENTATION_DEFS.get(aug_key, {}).get("cols", [])
                            if c in df.columns]
        aligned = df[df["name_loc"].isin(common_locs)]
        all_feats = _feat_cols(aligned)
        print()
        cumvars[label] = _pca_stats(aligned, all_feats, label)

        # Internal structure of the augmentation block alone
        if aug_cols_present:
            X_aug = aligned[aug_cols_present].copy()
            for c in X_aug.columns:
                X_aug[c] = X_aug[c].fillna(X_aug[c].median())
            pca_aug = PCA().fit(scaler.fit_transform(X_aug))
            cumvar_aug = np.cumsum(pca_aug.explained_variance_ratio_)
            print(f"    {aug_key} block ({len(aug_cols_present)} features) alone:")
            for t in (0.80, 0.90):
                n = int(np.searchsorted(cumvar_aug, t) + 1)
                redundancy = (1 - n / len(aug_cols_present)) * 100
                print(f"      {int(t*100)}% variance → {n} / {len(aug_cols_present)} components "
                      f"(internal redundancy: {redundancy:.0f}%)")

    # Scree plot
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, cumvar in cumvars.items():
        color = PALETTE.get(label, "#888888")
        n = min(len(cumvar), 60)
        feats = _feat_cols(datasets[label])
        ax.plot(range(1, n + 1), cumvar[:n] * 100, label=f"{label} ({len(feats)} feats)",
                color=color, linewidth=1.8,
                linestyle="-" if label == "Base" else "--")
    for t in (80, 90, 95):
        ax.axhline(t, color="gray", linestyle=":", linewidth=0.8)
        ax.text(60 * 0.98, t + 0.5, f"{t}%", ha="right", fontsize=8, color="gray")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA scree comparison — all datasets")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "pca_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_dir / 'pca_comparison.png'}")


# ---------------------------------------------------------------------------
# Per-country R²
# ---------------------------------------------------------------------------

def per_country_comparison(
    results: dict[str, dict],
    out_dir: Path,
) -> None:
    sep("Per-country holdout R²")

    rows = []
    base_res = results["Base"]
    all_countries = sorted(base_res["df"][COUNTRY_COL].unique())

    for country in all_countries:
        for label, res in results.items():
            mask = res["df"].iloc[res["test_idx"]][COUNTRY_COL].values == country
            if mask.sum() < 5:
                continue
            y_t = res["y"].iloc[res["test_idx"]].values[mask]
            y_p = res["model"].predict(res["X"].iloc[res["test_idx"]])[mask]
            r2  = r2_score(y_t, y_p) if len(y_t) > 1 else float("nan")
            rows.append({"country": country, "model": label, "n": mask.sum(),
                         "r2": r2, "mae": mean_absolute_error(y_t, y_p)})

    res_df = pd.DataFrame(rows)
    labels = list(results.keys())

    # Print table
    header = f"  {'Country':8s}  {'n':>6s}" + "".join(f"  {'R²_'+lb:>10s}" for lb in labels)
    print(header)
    print("  " + "─" * (18 + 12 * len(labels)))
    for country in all_countries:
        sub = res_df[res_df["country"] == country]
        if sub.empty:
            continue
        n_val = sub[sub["model"] == "Base"]["n"].values
        n_str = f"{n_val[0]:6d}" if len(n_val) else "     -"
        row = f"  {country:8s}  {n_str}"
        for lb in labels:
            r2_row = sub[sub["model"] == lb]["r2"].values
            row += f"  {r2_row[0]:10.3f}" if len(r2_row) else f"  {'—':>10s}"
        print(row)

    # Bar chart
    pivot = res_df.pivot(index="country", columns="model", values="r2")
    n_models = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, 2 * len(all_countries)), 4))
    x = np.arange(len(all_countries))
    w = 0.8 / n_models
    for i, lb in enumerate(labels):
        vals = [pivot.loc[c, lb] if c in pivot.index and lb in pivot.columns
                else float("nan") for c in all_countries]
        offset = (i - n_models / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=lb,
               color=PALETTE.get(lb, "#888888"))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(all_countries)
    ax.set_ylabel("Holdout R²")
    ax.set_title("Per-country holdout R²")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "per_country_r2.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_dir / 'per_country_r2.png'}")


# ---------------------------------------------------------------------------
# Hydra config recommendations
# ---------------------------------------------------------------------------

def hydra_recommendations(
    datasets: dict[str, pd.DataFrame],
    results: dict[str, dict],
    data_dir: Path,
) -> None:
    sep("Hydra config recommendations")

    base = datasets["Base"]
    m_base = results["Base"]
    base_feats = _feat_cols(base)

    # Rank augmentations by CV R²
    ranking = sorted(
        [(lb, res) for lb, res in results.items()],
        key=lambda x: x[1]["cv_r2_mean"],
        reverse=True,
    )

    print("\nRANKING BY CV R²")
    print("────────────────")
    for rank, (lb, res) in enumerate(ranking, 1):
        dr2 = res["cv_r2_mean"] - m_base["cv_r2_mean"]
        delta = f"  (Δ={dr2:+.3f} vs Base)" if lb != "Base" else ""
        print(f"  #{rank}  {lb:14s}  CV R²={res['cv_r2_mean']:.3f}  "
              f"HO R²={res['holdout_r2']:.3f}  MAE={res['holdout_mae']:.3f} t/ha{delta}")

    best_label, best_res = ranking[0]
    best_dr2_cv = best_res["cv_r2_mean"] - m_base["cv_r2_mean"]
    best_dr2_ho = best_res["holdout_r2"] - m_base["holdout_r2"]

    # Tabular dim for each dataset (injected at runtime: +1 year, +6 Fourier; no country OH)
    print("\nEXPECTED TABULAR DIMENSIONS  (use_country_features=False)")
    print("────────────────────────────────────────────────────────")
    print(f"  {'Dataset':14s}  {'CSV feat_*':>10s}  {'+ injected':>10s}  {'= total':>8s}  CSV name")
    print("  " + "─" * 72)
    for lb, df in datasets.items():
        csv_feats = len(_feat_cols(df))
        injected = 7   # feat_year + 6 Fourier harmonics (no country one-hots)
        total = csv_feats + injected
        csv_name = f"model_ready_yield_africa_{'base' if lb == 'Base' else lb.lower().replace('+', '').replace(' ', '')}.csv"
        print(f"  {lb:14s}  {csv_feats:>10d}  {injected:>10d}  {total:>8d}  {csv_name}")

    print(f"""
HYDRA CONFIG SNIPPETS
─────────────────────
  1. Set DATA_DIR in .env:
       DATA_DIR=/Volumes/data_and_models_2/aether/data

  2. Select CSV per experiment (csv_name is already implemented in BaseDataset):
       # In any yield_africa data config:
       dataset:
         csv_name: model_ready_yield_africa_base.csv      # base
         csv_name: model_ready_yield_africa_ndvi.csv      # +NDVI
         csv_name: model_ready_yield_africa_agera5.csv    # +AgERA5

       # Or override at the command line:
       python src/train.py experiment=yield_africa_tabular_loco \\
         data.dataset.csv_name=model_ready_yield_africa_agera5.csv

  3. LOCO experiments — always keep:
       dataset:
         use_country_features: false   # prevents distribution shift for held-out country

  4. Update input_dim in yield_tabular_reg.yaml to match the chosen CSV's tabular_dim.
     The tabular_dim is printed at dataset init — confirm from the training log.

RECOMMENDATION
──────────────""")

    if best_label == "Base":
        print(f"\n  ✦ USE BASE CSV")
        print(f"  No augmentation improves CV R² by more than 0.01.")
    else:
        aug_key = _aug_name(best_label)
        meaningful = best_dr2_cv > 0.01 or best_dr2_ho > 0.01
        if meaningful:
            print(f"\n  ✦ USE {best_label.upper()} CSV")
            print(f"  Best augmentation: {best_label}  "
                  f"(ΔCV R²={best_dr2_cv:+.3f}, ΔHO R²={best_dr2_ho:+.3f}).")
            if len(ranking) > 2:
                second_lb, second_res = ranking[1]
                if second_lb != "Base":
                    second_dr2 = second_res["cv_r2_mean"] - m_base["cv_r2_mean"]
                    gap = best_res["cv_r2_mean"] - second_res["cv_r2_mean"]
                    print(f"  Runner-up: {second_lb} (ΔCV R²={second_dr2:+.3f}, "
                          f"gap to winner: {gap:.3f}).")
                    if gap < 0.01:
                        print(f"  Gap < 0.01 — consider combining {best_label} and {second_lb} "
                              f"if a combined CSV is available.")
        else:
            print(f"\n  ✦ MARGINAL — differences are small (ΔCV R²={best_dr2_cv:+.3f}).")
            print(f"  All augmentations perform similarly. Prefer base CSV for simplicity.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_csv",   required=True,
                        help="Path to model_ready_yield_africa_base.csv")
    parser.add_argument("--ndvi_csv",   default=None,
                        help="Path to model_ready_yield_africa_ndvi.csv (optional)")
    parser.add_argument("--agera5_csv", default=None,
                        help="Path to model_ready_yield_africa_agera5.csv (optional)")
    parser.add_argument("--merged_csv", default=None,
                        help="Path to model_ready_yield_africa_merged.csv (optional)")
    parser.add_argument("--out_dir",    default="data/yield_africa/analysis_augmentation",
                        help="Output directory for figures")
    parser.add_argument("--n_trees",    type=int, default=300,
                        help="Number of RF trees (default: 300)")
    parser.add_argument("--xgb_n_estimators", type=int, default=300,
                        help="Number of XGBoost estimators (default: 300)")
    parser.add_argument("--model",      choices=["rf", "xgb", "both"], default="both",
                        help="Which model(s) to run: rf, xgb, or both (default: both)")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument(
        "--complete_only",
        action="store_true",
        default=False,
        help=(
            "Restrict the RF comparison to rows where every augmented dataset has "
            "non-null values in its key augmentation columns (seasonal aggregates for "
            "NDVI; all columns for AgERA5).  Useful when downloads are still in progress "
            "and many rows are NaN-imputed, which would mask the true benefit of augmentation."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build ordered path dict — Base always first
    paths: dict[str, Path] = {"Base": Path(args.base_csv)}
    if args.ndvi_csv:
        paths["Base+NDVI"] = Path(args.ndvi_csv)
    if args.agera5_csv:
        paths["Base+AgERA5"] = Path(args.agera5_csv)
    if args.merged_csv:
        paths["Merged"] = Path(args.merged_csv)

    for label, p in paths.items():
        if not p.exists():
            print(f"ERROR: {label} CSV not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Infer data_dir for the recommendation snippets
    data_dir = Path(args.base_csv).parent.parent

    datasets = load_datasets(paths)
    coverage_report(datasets)
    feature_set_report(datasets)

    rf_results = None
    xgb_results = None

    if args.model in ("rf", "both"):
        rf_results = rf_comparison(datasets, args.n_trees, args.seed, out_dir, args.complete_only)
        augmentation_importance_reports(rf_results, out_dir, args.seed)
        per_country_comparison(rf_results, out_dir)

    if args.model in ("xgb", "both"):
        xgb_results = xgb_comparison(
            datasets, args.xgb_n_estimators, args.seed, out_dir, args.complete_only
        )
        aug_out_dir = out_dir / "xgb"
        aug_out_dir.mkdir(parents=True, exist_ok=True)
        augmentation_importance_reports(xgb_results, aug_out_dir, args.seed)
        per_country_comparison(xgb_results, aug_out_dir)

    pca_comparison(datasets, out_dir)

    # Use RF results for Hydra recommendations when available, else XGBoost.
    rec_results = rf_results if rf_results is not None else xgb_results
    hydra_recommendations(datasets, rec_results, data_dir)

    sep()
    print(f"Analysis complete. Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
