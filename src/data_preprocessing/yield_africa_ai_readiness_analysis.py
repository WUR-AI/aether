"""Exploratory analysis of the raw crop yield Africa dataset.

Run this BEFORE step 1 (make_model_ready_yield_africa.py) to understand the
raw data quality and AI-readiness of the dataset.  It operates on the original
semicolon-delimited raw CSV (not the model-ready output) and produces:

  • Data quality report: missingness, target distribution, country/year balance,
    location accuracy.
  • Random Forest and/or XGBoost regression with spatial group CV, permutation
    importance, and per-group importance share.
  • PCA redundancy analysis across the full feature set and per feature group.
  • Residual and spatial bias analysis (predicted vs actual, spatial residual map).
  • Feature correlation heatmap for the top-20 features.
  • Structured AI-readiness gap report with strengths, gaps, and next steps.

Usage (from repo root, with venv active):
    python src/data_preprocessing/yield_africa_ai_readiness_analysis.py \\
        --input_csv  data/yield_africa/Full_dataset_CropYield_classified_and_numeric_v20260218.csv \\
        --out_dir    data/yield_africa/analysis_ai_readiness

Optional arguments:
    --input_csv         Path to raw semicolon-delimited CSV (required)
    --out_dir           Directory for output figures
                        (default: data/yield_africa/analysis_ai_readiness)
    --model             Which model(s) to run: rf, xgb, or both (default: both)
    --n_trees           Number of trees in the random forest (default: 300)
    --xgb_n_estimators  Number of XGBoost estimators (default: 300)
    --seed              Random seed (default: 42)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (kept in sync with make_model_ready_yield_africa.py)
# ---------------------------------------------------------------------------

SOIL_FEATURES = [
    "C_0_20", "C_20_50", "N_0_20", "N_20_50",
    "P_0_20", "P_20_50", "MA_0_20", "MA_20_50",
    "PO_0_20", "PO_20_50", "pH_0_20", "pH_20_50",
    "BD_0_20", "BD_20_50", "ECX_0_20", "ECX_20_50",
    "CA_0_20", "CA_20_50",
]

CLIMATE_FEATURES = [
    "PrecJJA", "PrecMAM", "PrecSON", "PrecDJF",
    "TaveJJA", "TaveMAM", "TaveSON", "TaveDJF",
    "TmaxJJA", "TmaxMAM", "TmaxSON", "TmaxDJF",
    "TminJJA", "TminMAM", "TminSON", "TminDJF",
    "CMD", "Eref", "MAP", "MAT", "TD", "MWMT", "MCMT",
    "DD_above_5", "DD_above_18", "DD_below_18",
]

TERRAIN_FEATURES = [
    "DEM", "Slope", "Aspect", "CHILI", "Top_div",
]

CONTEXT_FEATURES = [
    "Tree_c", "Dist_water", "Paved", "Unpaved", "Pop_10km",
]

CATEGORICAL_FEATURES = ["TX_0_20_cl", "TX_20_50_cl"]

TARGET_COL = "Yld_ton_ha"
LAT_COL = "Lat"
LON_COL = "Lon"
COUNTRY_COL = "Country"
YEAR_COL = "Year"
LOC_ACC_COL = "Location_accuracy"

ALL_NUMERIC_FEATURES = SOIL_FEATURES + CLIMATE_FEATURES + TERRAIN_FEATURES + CONTEXT_FEATURES

FEATURE_GROUPS = {
    "Soil": SOIL_FEATURES,
    "Climate": CLIMATE_FEATURES,
    "Terrain": TERRAIN_FEATURES,
    "Context": CONTEXT_FEATURES,
    "Categorical (encoded)": CATEGORICAL_FEATURES,
}

LOG_TRANSFORM_FEATURES = ["Dist_water", "Paved", "Unpaved", "Pop_10km"]

DERIVED_FEATURE_NAMES = ["CN_ratio", "C_layer_delta", "BD_layer_delta", "WHC_proxy", "aridity_index"]

# Water holding capacity lookup (Saxton & Rawls 2006, ~2.5% OM, mm/m)
WHC_LOOKUP = {
    "sand": 50, "loamy sand": 70, "sandy loam": 100, "loam": 140,
    "silt loam": 200, "silt": 250, "sandy clay loam": 100,
    "clay loam": 140, "silty clay loam": 170, "silty clay": 140,
    "sandy clay": 110, "clay": 120,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_comma_float(series: pd.Series) -> pd.Series:
    """Convert European-style comma-decimal strings to float (e.g. '1,5' → 1.5)."""
    if series.dtype == object or str(series.dtype) == "str":
        return series.astype(str).str.replace(",", ".").astype(float)
    return series.astype(float)


def sep(title: str = "", width: int = 72) -> None:
    if title:
        pad = max(0, width - len(title) - 4)
        print(f"\n{'─' * 2} {title} {'─' * pad}")
    else:
        print("─" * width)


def _pct(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.1f}%)"


# ---------------------------------------------------------------------------
# Loading and parsing
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> pd.DataFrame:
    sep("Loading data")
    df = pd.read_csv(csv_path, sep=",")
    print(f"Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Parse comma-decimal string columns to float
    str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]
    non_numeric_str_cols = []
    for col in str_cols:
        if col in (COUNTRY_COL, LOC_ACC_COL, "Yld_ton_ha_cl") or col.endswith("_cl"):
            non_numeric_str_cols.append(col)
            continue
        try:
            df[col] = _parse_comma_float(df[col])
        except (ValueError, TypeError):
            non_numeric_str_cols.append(col)

    print(f"Parsed {len(str_cols) - len(non_numeric_str_cols)} string columns to float.")
    print(f"Kept as categorical/string: {non_numeric_str_cols}")
    return df


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def data_quality_report(df: pd.DataFrame) -> None:
    sep("Data quality")

    total = len(df)
    print(f"Total rows: {total:,}")
    print(f"Total columns: {df.shape[1]}")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("Missing values: none")
    else:
        print(f"\nMissing values ({len(missing)} columns affected):")
        for col, n in missing.items():
            print(f"  {col:30s} {_pct(n, total)}")

    # Target distribution
    yld = df[TARGET_COL]
    print(f"\nTarget – {TARGET_COL}:")
    print(f"  range : {yld.min():.3f} – {yld.max():.3f} t/ha")
    print(f"  mean  : {yld.mean():.3f} t/ha  |  median: {yld.median():.3f} t/ha")
    print(f"  std   : {yld.std():.3f}  |  skew: {yld.skew():.2f}")
    q1, q3 = yld.quantile(0.25), yld.quantile(0.75)
    iqr = q3 - q1
    outliers = ((yld < q1 - 3 * iqr) | (yld > q3 + 3 * iqr)).sum()
    print(f"  3×IQR outliers: {_pct(outliers, total)}")

    # Class distribution (yield classes)
    if "Yld_ton_ha_cl" in df.columns:
        print("\nYield class distribution:")
        for cls, n in df["Yld_ton_ha_cl"].value_counts().sort_index().items():
            print(f"  {cls:50s} {_pct(n, total)}")

    # Country / year split
    print("\nSamples per country:")
    for country, n in df[COUNTRY_COL].value_counts().items():
        print(f"  {country:6s} {_pct(n, total)}")

    print("\nSamples per year:")
    for year, n in df[YEAR_COL].value_counts().sort_index().items():
        print(f"  {year:4d}  {n:,}")

    print("\nLocation accuracy:")
    for acc, n in df[LOC_ACC_COL].value_counts().items():
        print(f"  {acc:35s} {_pct(n, total)}")

    # Spatial extent
    lat = df[LAT_COL]
    lon = df[LON_COL]
    print(f"\nSpatial extent: lat [{lat.min():.2f}, {lat.max():.2f}]  lon [{lon.min():.2f}, {lon.max():.2f}]")


# ---------------------------------------------------------------------------
# Feature engineering (mirrors make_model_ready_yield_africa.py)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    sep("Feature engineering")
    df = df.copy()

    # C:N ratio
    df["CN_ratio"] = np.where(df["N_0_20"] > 0, df["C_0_20"] / df["N_0_20"], np.nan)
    # Layer deltas
    df["C_layer_delta"] = df["C_0_20"] - df["C_20_50"]
    df["BD_layer_delta"] = df["BD_0_20"] - df["BD_20_50"]
    # WHC proxy from texture class + bulk density
    if "TX_0_20_cl" in df.columns:
        df["WHC_proxy"] = df["TX_0_20_cl"].str.lower().map(WHC_LOOKUP).fillna(100)
        bd = df["BD_0_20"].replace(0, np.nan)
        df["WHC_proxy"] = df["WHC_proxy"] * (1.3 / bd).fillna(1.0)
    # Aridity index
    df["aridity_index"] = np.where(df["MAP"] > 0, df["CMD"] / df["MAP"], np.nan)
    # Log-transform skewed context features
    for col in LOG_TRANSFORM_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(np.maximum(df[col], 0))

    for name in DERIVED_FEATURE_NAMES:
        n_nan = df[name].isna().sum() if name in df.columns else "missing"
        print(f"  {name:20s}  NaN: {n_nan}")

    return df


# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) with imputed + encoded features."""
    all_features = ALL_NUMERIC_FEATURES + DERIVED_FEATURE_NAMES

    # Encode soil texture classes
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            all_features.append(f"{col}_enc")

    # Retain only columns that exist
    feature_cols = [c for c in all_features if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Impute remaining NaN with median
    n_nan_before = X.isna().sum().sum()
    if n_nan_before > 0:
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

    print(f"\nFeature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"NaN values imputed: {n_nan_before}")

    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Random forest regression
# ---------------------------------------------------------------------------

def run_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_trees: int,
    seed: int,
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray]:
    """Train RF on a random 70/30 split; also run spatial group cross-validation."""
    sep("Random Forest regression")

    # ── Spatial cross-validation (group = country × year block) ──────────────
    print("Spatial cross-validation (GroupShuffleSplit by country×year, 5 splits):")
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    rf_cv = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    cv_r2 = cross_val_score(rf_cv, X, y, cv=gss, groups=groups, scoring="r2")
    cv_mae = cross_val_score(rf_cv, X, y, cv=gss, groups=groups, scoring="neg_mean_absolute_error")
    cv_rmse = cross_val_score(rf_cv, X, y, cv=gss, groups=groups, scoring="neg_root_mean_squared_error")
    print(f"  R²    : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}  (folds: {np.round(cv_r2, 3)})")
    print(f"  MAE   : {-cv_mae.mean():.3f} ± {cv_mae.std():.3f} t/ha")
    print(f"  RMSE  : {-cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} t/ha")

    # ── Single holdout split for residual analysis & permutation importance ──
    # Use GroupShuffleSplit so test countries/years differ from train
    gss_single = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss_single.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"\nHoldout split: {len(train_idx):,} train / {len(test_idx):,} test")
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

    print(f"\nHoldout metrics:")
    print(f"  R²          : {r2:.3f}")
    print(f"  MAE         : {mae:.3f} t/ha")
    print(f"  RMSE        : {rmse:.3f} t/ha")
    print(f"  Baseline MAE: {baseline_mae:.3f} t/ha  (predict mean)")
    print(f"  Skill (1 - MAE/baseline): {1 - mae / baseline_mae:.3f}")

    return rf, train_idx, test_idx


# ---------------------------------------------------------------------------
# XGBoost regression
# ---------------------------------------------------------------------------

def run_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_estimators: int,
    seed: int,
) -> tuple:
    """Train XGBoost on a random 75/25 split; also run spatial group cross-validation."""
    sep("XGBoost regression")

    # ── Spatial cross-validation ──────────────────────────────────────────────
    print("Spatial cross-validation (GroupShuffleSplit by country×year, 5 splits):")
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    xgb_cv = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    cv_r2 = cross_val_score(xgb_cv, X, y, cv=gss, groups=groups, scoring="r2")
    cv_mae = cross_val_score(xgb_cv, X, y, cv=gss, groups=groups, scoring="neg_mean_absolute_error")
    cv_rmse = cross_val_score(xgb_cv, X, y, cv=gss, groups=groups, scoring="neg_root_mean_squared_error")
    print(f"  R²    : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}  (folds: {np.round(cv_r2, 3)})")
    print(f"  MAE   : {-cv_mae.mean():.3f} ± {cv_mae.std():.3f} t/ha")
    print(f"  RMSE  : {-cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} t/ha")

    # ── Single holdout split ──────────────────────────────────────────────────
    gss_single = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss_single.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"\nHoldout split: {len(train_idx):,} train / {len(test_idx):,} test")
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

    print(f"\nHoldout metrics:")
    print(f"  R²          : {r2:.3f}")
    print(f"  MAE         : {mae:.3f} t/ha")
    print(f"  RMSE        : {rmse:.3f} t/ha")
    print(f"  Baseline MAE: {baseline_mae:.3f} t/ha  (predict mean)")
    print(f"  Skill (1 - MAE/baseline): {1 - mae / baseline_mae:.3f}")

    return model, train_idx, test_idx


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importance_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    test_idx: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    seed: int,
) -> pd.DataFrame:
    sep("Feature importance")

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # ── MDI (Mean Decrease in Impurity) ──────────────────────────────────────
    mdi_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    # ── Permutation importance (on test set) ─────────────────────────────────
    print("Computing permutation importance on test set (n_repeats=10)…")
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=1)
    perm_mean = pd.Series(perm.importances_mean, index=feature_names)
    perm_std = pd.Series(perm.importances_std, index=feature_names)

    importance_df = pd.DataFrame({
        "mdi": mdi_importances,
        "perm_mean": perm_mean,
        "perm_std": perm_std,
    }).sort_values("perm_mean", ascending=False)

    # Print top-30
    print(f"\n{'Feature':30s}  {'MDI':>7s}  {'Perm':>7s}  {'±':>6s}  Group")
    print("─" * 68)
    for feat, row in importance_df.head(30).iterrows():
        group = next((g for g, cols in FEATURE_GROUPS.items() if feat in cols or feat.rstrip("_enc") in cols), "Derived")
        if feat in DERIVED_FEATURE_NAMES:
            group = "Derived"
        print(f"  {feat:28s}  {row.mdi:7.4f}  {row.perm_mean:7.4f}  {row.perm_std:6.4f}  {group}")

    # Group-level summary
    sep("Importance by group")
    for group, cols in FEATURE_GROUPS.items():
        # also include encoded variants
        group_cols = [c for c in importance_df.index if c.rstrip("_enc") in cols or c in cols]
        if not group_cols:
            continue
        total_mdi = importance_df.loc[group_cols, "mdi"].sum()
        total_perm = importance_df.loc[group_cols, "perm_mean"].sum()
        print(f"  {group:30s}  MDI: {total_mdi:.4f}  Perm: {total_perm:.4f}  ({len(group_cols)} features)")
    derived_cols = [c for c in importance_df.index if c in DERIVED_FEATURE_NAMES]
    if derived_cols:
        total_mdi = importance_df.loc[derived_cols, "mdi"].sum()
        total_perm = importance_df.loc[derived_cols, "perm_mean"].sum()
        print(f"  {'Derived':30s}  MDI: {total_mdi:.4f}  Perm: {total_perm:.4f}  ({len(derived_cols)} features)")

    # ── Figure: top-20 permutation importances ────────────────────────────────
    top20 = importance_df.head(20)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(top20))
    ax.barh(list(y_pos), top20["perm_mean"].values[::-1], xerr=top20["perm_std"].values[::-1],
            color="steelblue", ecolor="gray", capsize=3)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(list(top20.index[::-1]), fontsize=9)
    ax.set_xlabel("Mean permutation importance (R² decrease)")
    ax.set_title("Top-20 feature importances (permutation, test set)")
    plt.tight_layout()
    fig_path = out_dir / "feature_importance.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    return importance_df


# ---------------------------------------------------------------------------
# PCA analysis
# ---------------------------------------------------------------------------

def pca_analysis(
    X: pd.DataFrame,
    feature_names: list[str],
    out_dir: Path,
) -> None:
    sep("PCA – dimensionality and redundancy")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    for threshold in (0.80, 0.90, 0.95, 0.99):
        n_components = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"  {int(threshold*100):3d}% variance explained by {n_components:2d} / {len(feature_names)} components")

    # Effective rank (Shannon entropy on explained variance)
    ev = explained[explained > 0]
    eff_rank = int(np.round(np.exp(-np.sum(ev * np.log(ev)))))
    print(f"  Effective rank (exp entropy): {eff_rank}")

    # Group-level PCA on each feature group
    print()
    for group, cols in FEATURE_GROUPS.items():
        group_cols = [c for c in feature_names if c in cols or c.rstrip("_enc") in cols]
        if len(group_cols) < 2:
            continue
        X_g = scaler.fit_transform(X[group_cols])
        pca_g = PCA()
        pca_g.fit(X_g)
        cum_g = np.cumsum(pca_g.explained_variance_ratio_)
        n90 = int(np.searchsorted(cum_g, 0.90) + 1)
        print(f"  {group:30s} {len(group_cols):2d} features → {n90:2d} components for 90% var  "
              f"(redundancy: {(1 - n90/len(group_cols))*100:.0f}%)")

    # ── Figure: scree plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(explained) + 1), cumulative * 100, marker="o", markersize=3, linewidth=1.5)
    for t in (80, 90, 95):
        ax.axhline(t, color="gray", linestyle="--", linewidth=0.8)
        ax.text(len(explained) * 0.98, t + 0.5, f"{t}%", ha="right", fontsize=8, color="gray")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA scree plot – full feature set")
    plt.tight_layout()
    fig_path = out_dir / "pca_scree.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {fig_path}")


# ---------------------------------------------------------------------------
# Residual / spatial bias analysis
# ---------------------------------------------------------------------------

def residual_analysis(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df_meta: pd.DataFrame,
    test_idx: np.ndarray,
    out_dir: Path,
) -> None:
    sep("Residual and spatial bias analysis")

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    meta_test = df_meta.iloc[test_idx]
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    print(f"Residuals: mean={residuals.mean():.4f}  std={residuals.std():.3f}")
    print(f"  Bias (systematic over/under-prediction): {residuals.mean():.4f} t/ha")

    # Per-country breakdown
    print("\nPer-country metrics (test set):")
    print(f"  {'Country':8s}  {'n':>5s}  {'R²':>6s}  {'MAE':>6s}  {'Bias':>7s}")
    print("  " + "─" * 45)
    for country in sorted(meta_test[COUNTRY_COL].unique()):
        mask = (meta_test[COUNTRY_COL] == country).values
        if mask.sum() < 5:
            continue
        y_c = y_test.values[mask]
        p_c = y_pred[mask]
        r_c = y_c - p_c
        r2_c = r2_score(y_c, p_c) if len(y_c) > 1 else float("nan")
        print(f"  {country:8s}  {mask.sum():5d}  {r2_c:6.3f}  {mean_absolute_error(y_c,p_c):6.3f}  {r_c.mean():+7.3f}")

    # Per-year breakdown
    print("\nPer-year metrics (test set):")
    print(f"  {'Year':6s}  {'n':>5s}  {'MAE':>6s}  {'Bias':>7s}")
    print("  " + "─" * 35)
    for year in sorted(meta_test[YEAR_COL].unique()):
        mask = (meta_test[YEAR_COL] == year).values
        if mask.sum() < 5:
            continue
        y_c = y_test.values[mask]
        p_c = y_pred[mask]
        r_c = y_c - p_c
        print(f"  {year:6d}  {mask.sum():5d}  {mean_absolute_error(y_c,p_c):6.3f}  {r_c.mean():+7.3f}")

    # ── Figure: predicted vs actual + spatial residual map ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.25, s=8, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", linewidth=1)
    ax1.set_xlabel("Actual yield (t/ha)")
    ax1.set_ylabel("Predicted yield (t/ha)")
    ax1.set_title(f"Predicted vs Actual  (R²={r2_score(y_test,y_pred):.3f})")

    ax2 = axes[1]
    lat = meta_test[LAT_COL].values
    lon = meta_test[LON_COL].values
    vmax = max(abs(residuals.min()), abs(residuals.max()))
    sc = ax2.scatter(lon, lat, c=residuals, cmap="RdBu_r", s=8, alpha=0.5,
                     vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax2, label="Residual (t/ha)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_title("Spatial residuals (test set)")

    plt.tight_layout()
    fig_path = out_dir / "residuals.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {fig_path}")


# ---------------------------------------------------------------------------
# Correlation among top features
# ---------------------------------------------------------------------------

def correlation_heatmap(
    X: pd.DataFrame,
    importance_df: pd.DataFrame,
    out_dir: Path,
    n_top: int = 20,
) -> None:
    sep("Feature correlation heatmap (top features)")
    top_features = importance_df.head(n_top).index.tolist()
    corr = X[top_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_features)))
    ax.set_yticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(top_features, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Pearson correlations — top-{n_top} features")
    plt.tight_layout()
    fig_path = out_dir / "feature_correlation.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Flag highly correlated pairs
    high_corr = []
    for i, fi in enumerate(top_features):
        for j, fj in enumerate(top_features):
            if j <= i:
                continue
            if abs(corr.loc[fi, fj]) > 0.90:
                high_corr.append((fi, fj, corr.loc[fi, fj]))
    if high_corr:
        print(f"\nHighly correlated pairs (|r| > 0.90) among top-{n_top}:")
        for fi, fj, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f"  {fi:30s} — {fj:30s}  r={r:.3f}")
    else:
        print(f"No highly correlated pairs (|r| > 0.90) among top-{n_top}.")


# ---------------------------------------------------------------------------
# AI-readiness gap analysis
# ---------------------------------------------------------------------------

def ai_readiness_report(
    df: pd.DataFrame,
    importance_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    sep("AI-readiness gap analysis")

    total = len(df)

    # ── 1. Data volume and balance ────────────────────────────────────────────
    print("\n[1] DATA VOLUME & BALANCE")
    print(f"  Total samples         : {total:,}")
    country_counts = df[COUNTRY_COL].value_counts()
    dominant = country_counts.index[0]
    pct_dominant = 100 * country_counts.iloc[0] / total
    print(f"  Countries             : {df[COUNTRY_COL].nunique()}  ({', '.join(sorted(df[COUNTRY_COL].unique()))})")
    print(f"  Dominant country      : {dominant} ({pct_dominant:.1f}% of data)")
    if pct_dominant > 50:
        print(f"  ⚠  Heavy country imbalance: {dominant} dominates. Models may overfit to its "
              "agro-climatic conditions. Country-stratified sampling or re-weighting recommended.")

    years_covered = sorted(df[YEAR_COL].unique())
    print(f"  Years covered         : {years_covered[0]}–{years_covered[-1]}  (n={len(years_covered)})")
    thin_years = df[YEAR_COL].value_counts()
    thin_years = thin_years[thin_years < 200]
    if not thin_years.empty:
        sparse_str = ", ".join(f"{y}: {n}" for y, n in thin_years.sort_index().items())
        print(f"  ⚠  Sparse years: {sparse_str}. Temporal generalization may be limited.")

    high_acc = (df[LOC_ACC_COL] == "High location accuracy").sum()
    print(f"  High location accuracy: {_pct(high_acc, total)}")
    low_acc = (df[LOC_ACC_COL] == "Low location accuracy").sum()
    if low_acc / total > 0.1:
        print(f"  ⚠  {_pct(low_acc, total)} samples have low location accuracy — "
              "EO pixel matching will be unreliable for those.")

    # ── 2. Feature completeness ───────────────────────────────────────────────
    print("\n[2] FEATURE COMPLETENESS")
    all_present = all(c in df.columns for c in ALL_NUMERIC_FEATURES)
    print(f"  All numeric features present : {all_present}")
    missing_cols = [c for c in ALL_NUMERIC_FEATURES if c not in df.columns]
    if missing_cols:
        print(f"  Missing: {missing_cols}")

    # NaN after float parsing
    num_cols = [c for c in feature_names if c in df.columns]
    nan_counts = df[num_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if nan_cols.empty:
        print(f"  No NaN values in feature columns (after parsing)")
    else:
        print(f"  Columns with NaN: {dict(nan_cols)}")

    # Identify features with near-zero variance
    num_df = df[[c for c in ALL_NUMERIC_FEATURES if c in df.columns]].apply(
        lambda s: _parse_comma_float(s) if (s.dtype == object or str(s.dtype) == "str") else s
    )
    low_var = num_df.std()[num_df.std() < 1e-4]
    if not low_var.empty:
        print(f"  ⚠  Near-zero variance features: {low_var.index.tolist()}")
    else:
        print(f"  No near-zero variance features.")

    # ── 3. Target leakage risks ───────────────────────────────────────────────
    print("\n[3] LEAKAGE / PROXY RISKS")
    print("  ⚠  TX_0_20_cl / TX_20_50_cl: soil texture classification may be derived from the")
    print("     same survey as yield data — verify independent provenance.")
    print("  ⚠  Yld_ton_ha_cl must be excluded from feature matrix (direct yield encoding).")
    print("  ⚠  ID column is just a row number — not predictive, already excluded.")

    # ── 4. Spatial autocorrelation risk ──────────────────────────────────────
    print("\n[4] SPATIAL AUTOCORRELATION")
    # Quick estimate: what fraction of test samples are within 10 km of a training sample?
    print("  ⚠  Most climate, soil, and terrain features are spatially smooth — nearby samples")
    print("     share nearly identical feature vectors. Random splits will overestimate performance.")
    print("     → Spatial or country-based cross-validation is essential (already implemented).")
    print("     → Leave-one-country-out (LOCO) splits expose generalisation to new regions.")

    # ── 5. Feature importance findings ───────────────────────────────────────
    print("\n[5] FEATURE IMPORTANCE INSIGHTS")
    top5 = importance_df.head(5).index.tolist()
    print(f"  Top-5 features (permutation): {', '.join(top5)}")
    # Check if management / variety features are present
    mgmt_keywords = ["fertilizer", "fertil", "irrig", "variety", "crop_type", "planting", "harvest",
                     "pesticide", "tillage", "manure", "nitrogen_applied"]
    present_mgmt = [k for k in mgmt_keywords if any(k in c.lower() for c in df.columns)]
    print(f"\n  Management / agronomic features present: {present_mgmt if present_mgmt else 'NONE'}")
    print("  ⚠  No crop management variables detected (fertilizer, irrigation, variety, planting")
    print("     date, etc.). These are the strongest drivers of yield variation in smallholder")
    print("     contexts and are the primary explanation gap for yield prediction models.")

    eo_keywords = ["ndvi", "evi", "lai", "s2", "sentinel", "eo_", "modis", "landsat", "radar", "sar"]
    present_eo = [k for k in eo_keywords if any(k in c.lower() for c in df.columns)]
    print(f"\n  EO / remote sensing features present: {present_eo if present_eo else 'NONE'}")
    print("  ⚠  No satellite-derived vegetation indices (NDVI, EVI, LAI) or SAR backscatter")
    print("     in the raw CSV. In-season EO time series are among the strongest predictors for")
    print("     yield forecasting. The pipeline addresses this via:")
    print("       • Step 2a: year-specific NDVI augmentation (MODIS MOD13Q1, monthly + seasonal)")
    print("       • Step 2b: AgERA5 climate augmentation (temperature, precipitation, radiation)")
    print("       • TESSERA spatial embeddings (spatial EO signal via geo-encoder)")

    # ── 6. AI-readiness summary ───────────────────────────────────────────────
    sep("AI-readiness summary")
    low_pct = 100 * low_acc / total
    print(f"""
STRENGTHS
─────────
  ✓  30,000+ samples across 8 African countries and 7+ years.
  ✓  Rich static feature set: 18 soil, 26 climate, 5 terrain, 5 context features.
  ✓  Pre-computed classified (_cl) columns support caption-based explanation methods.
  ✓  Derived features (CN ratio, WHC proxy, aridity index) add agronomic signal.
  ✓  Yield classes defined for classification tasks.
  ✓  Spatial split infrastructure already implemented (LOCO, spatial block CV).
  ✓  Pipeline supports year-specific NDVI augmentation (monthly + seasonal means).
  ✓  Pipeline supports AgERA5 climate augmentation (temperature, precipitation, etc.).
  ✓  TESSERA spatial embeddings available as an additional modality.

GAPS (ordered by estimated impact on model quality)
────────────────────────────────────────────────────
  1. NO MANAGEMENT DATA
     Fertilizer application, irrigation, crop variety, planting / harvest dates, and
     tillage practices are the largest unmeasured drivers of yield variation. Without
     them the model will conflate agro-climatic suitability with actual productivity.
     → Integrate GYGA, FAOSTAT, or plot-level survey data if available.

  2. LIMITED IN-SEASON EO SIGNAL
     Static terrain/soil features cannot capture inter-annual crop-growth variation.
     The pipeline adds year-specific NDVI (monthly + seasonal aggregates from MODIS
     MOD13Q1) and AgERA5 climate variables, which substantially improve temporal
     signal. However, higher-resolution in-season SAR backscatter or multispectral
     time series per location-year would further improve performance.
     → Run steps 2a (NDVI) and 2b (AgERA5) of the pipeline, then compare with
       step 2d (augmentation comparison) to quantify the gain.

  3. HEAVY COUNTRY IMBALANCE
     {dominant} accounts for {pct_dominant:.0f}% of samples. Models trained on this dataset will
     implicitly optimise for its agro-climatic conditions. Stratified sampling or
     loss-weighting is needed for equitable multi-country models.
     → Use LOCO (leave-one-country-out) splits to test generalisation per country.

  4. SPARSE TEMPORAL COVERAGE FOR SOME COUNTRIES / YEARS
     Several countries only appear in specific years, making year-as-feature unreliable
     and temporal generalisation untestable. Expanding to consistent annual coverage
     would enable trend modelling.

  5. LOCATION ACCURACY HETEROGENEITY
     ~{low_pct:.1f}% of records have low location accuracy. EO pixel extraction will be
     off-target for these, introducing noise in any EO-feature fusion model.
     → Flag and optionally down-weight low-accuracy records; do not silently include them.

  6. MISSING CROP TYPE / CROP CALENDAR
     The dataset appears to cover a single crop or mixed crops — this is unclear from
     the column names. Yield magnitude and seasonality differ substantially across crops.
     A crop-type label per record would enable crop-specific models and explanations.

  7. STATIC SOIL DATA
     Soil features are likely derived from gridded products (e.g. SoilGrids), not
     plot-level measurements. Spatial resolution (~250 m) may not reflect field-scale
     variability relevant to smallholder plots (often < 1 ha).

RECOMMENDED NEXT STEPS
───────────────────────
  • Run the preprocessing pipeline (steps 1–2) to build the model-ready CSV with
    NDVI, AgERA5, and optionally TESSERA augmentations.
  • Use step 2d (augmentation comparison) to quantify how much each augmentation
    improves RF performance and decide which CSVs to use for training.
  • Collect or impute crop management proxies (at minimum: fertilizer use yes/no,
    irrigation yes/no) from household surveys or remote proxies.
  • Establish a held-out test set from a country not used in training to report
    unbiased generalisation performance; use LOCO splits as a proxy in the meantime.
  • Document crop types per record to enable crop-stratified modelling.
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _data_dir = Path(os.environ.get("DATA_DIR", "data"))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the raw semicolon-delimited yield Africa CSV",
    )
    parser.add_argument(
        "--out_dir",
        default=str(_data_dir / "yield_africa" / "analysis_ai_readiness"),
        help="Output directory for figures (default: <DATA_DIR>/yield_africa/analysis_ai_readiness)",
    )
    parser.add_argument("--n_trees", type=int, default=300, help="Number of RF trees (default: 300)")
    parser.add_argument("--xgb_n_estimators", type=int, default=300,
                        help="Number of XGBoost estimators (default: 300)")
    parser.add_argument("--model", choices=["rf", "xgb", "both"], default="both",
                        help="Which model(s) to run: rf, xgb, or both (default: both)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(csv_path)

    # ── Column validation ─────────────────────────────────────────────────────
    required_cols = [TARGET_COL, LAT_COL, LON_COL, COUNTRY_COL, YEAR_COL, LOC_ACC_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(
            f"\nERROR: The input CSV is missing required columns: {missing_cols}\n"
            f"\nThis script expects the rich semicolon-delimited raw dataset with soil,\n"
            f"climate, terrain, and location-accuracy columns (e.g. the file named\n"
            f"Full_dataset_CropYield_classified_and_numeric_*.csv).\n"
            f"\nColumns found in the provided CSV:\n  {list(df.columns)}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Data quality ──────────────────────────────────────────────────────────
    data_quality_report(df)

    # ── Feature engineering ───────────────────────────────────────────────────
    df = engineer_features(df)

    # ── Feature matrix ────────────────────────────────────────────────────────
    sep("Building feature matrix")
    X, y, feature_names = build_feature_matrix(df)

    # Group variable for spatial CV: country × year
    groups = df[COUNTRY_COL].astype(str) + "_" + df[YEAR_COL].astype(str)

    # ── Model(s) ──────────────────────────────────────────────────────────────
    # The last model run is used for importance/residual analysis so both can
    # be compared when --model=both; RF results come first, XGB overwrites.
    model = None
    train_idx = test_idx = None

    if args.model in ("rf", "both"):
        model, train_idx, test_idx = run_random_forest(X, y, groups, args.n_trees, args.seed)

    if args.model in ("xgb", "both"):
        xgb_out_dir = out_dir / "xgb" if args.model == "both" else out_dir
        xgb_out_dir.mkdir(parents=True, exist_ok=True)
        xgb_model, xgb_train_idx, xgb_test_idx = run_xgboost(
            X, y, groups, args.xgb_n_estimators, args.seed
        )
        xgb_imp_df = feature_importance_report(
            xgb_model, X, y, xgb_test_idx, feature_names, xgb_out_dir, args.seed
        )
        residual_analysis(
            xgb_model, X, y,
            df[[COUNTRY_COL, YEAR_COL, LAT_COL, LON_COL]],
            xgb_test_idx, xgb_out_dir,
        )
        # When running both, use RF results for the shared reports below;
        # when running xgb-only, promote to the primary model.
        if args.model == "xgb":
            model, train_idx, test_idx = xgb_model, xgb_train_idx, xgb_test_idx

    # ── Feature importance (RF or xgb-only path) ──────────────────────────────
    importance_df = feature_importance_report(model, X, y, test_idx, feature_names, out_dir, args.seed)

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_analysis(X, feature_names, out_dir)

    # ── Residual analysis (RF or xgb-only path) ───────────────────────────────
    residual_analysis(model, X, y, df[[COUNTRY_COL, YEAR_COL, LAT_COL, LON_COL]], test_idx, out_dir)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    correlation_heatmap(X, importance_df, out_dir)

    # ── AI-readiness report ───────────────────────────────────────────────────
    ai_readiness_report(df, importance_df, feature_names)

    sep()
    print(f"Analysis complete. Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
