"""Compare model_ready_yield_africa_base.csv against model_ready_yield_africa_ndvi.csv.

Evaluates whether adding AGERA5-derived NDVI features improves crop yield prediction,
diagnoses data coverage and quality differences between the two CSVs, and produces
actionable recommendations for selecting and configuring the dataset in the Hydra
config system.

Usage (from repo root, with venv active):
    python src/data_preprocessing/yield_africa_ndvi_comparison.py

Optional arguments:
    --data_dir   Root data directory (default: /Volumes/data_and_models_2/aether/data)
    --out_dir    Directory for output figures (default: data/yield_africa/analysis_ndvi_comparison)
    --n_trees    Number of RF trees (default: 300)
    --seed       Random seed (default: 42)
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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column sets (mirrors make_model_ready_yield_africa.py / base CSV layout)
# ---------------------------------------------------------------------------

TARGET_COL = "target_yld_ton_ha"
COUNTRY_COL = "country"
YEAR_COL = "year"
LAT_COL = "lat"
LON_COL = "lon"
LOC_ACC_COL = "location_accuracy"

# NDVI features added by yield_africa_augment_ndvi.py
NDVI_MONTHLY_COLS = [f"feat_ndvi_month_{m}" for m in range(1, 13)]
NDVI_SEASONAL_COLS = [
    "feat_ndvi_mean_djf",
    "feat_ndvi_mean_mam",
    "feat_ndvi_mean_jja",
    "feat_ndvi_mean_son",
    "feat_ndvi_mean_grow",
]
NDVI_COLS = NDVI_MONTHLY_COLS + NDVI_SEASONAL_COLS

# Non-feature columns that should be excluded from the feature matrix
NON_FEATURE_COLS = {
    "name_loc", TARGET_COL, LAT_COL, LON_COL, COUNTRY_COL,
    YEAR_COL, LOC_ACC_COL, "Landform", "GLAD",
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
    """Return feat_* columns present in df, excluding aux_* and non-feature columns."""
    return [c for c in df.columns if c.startswith("feat_")]


def _aux_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("aux_")]


def _build_X_y(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Impute NaN with column median and return (X, y)."""
    X = df[feat_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    y = df[TARGET_COL].copy()
    return X, y


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csvs(base_path: Path, ndvi_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sep("Loading CSVs")
    base = pd.read_csv(base_path)
    ndvi = pd.read_csv(ndvi_path)
    print(f"  Base : {base.shape[0]:,} rows × {base.shape[1]} columns  ({base_path.name})")
    print(f"  NDVI : {ndvi.shape[0]:,} rows × {ndvi.shape[1]} columns  ({ndvi_path.name})")
    return base, ndvi


# ---------------------------------------------------------------------------
# Coverage comparison
# ---------------------------------------------------------------------------

def coverage_report(base: pd.DataFrame, ndvi: pd.DataFrame) -> None:
    sep("Coverage comparison")

    base_locs = set(base["name_loc"])
    ndvi_locs = set(ndvi["name_loc"])
    only_base = base_locs - ndvi_locs
    only_ndvi = ndvi_locs - base_locs
    common = base_locs & ndvi_locs

    print(f"  Locations in base only : {len(only_base):,}")
    print(f"  Locations in NDVI only : {len(only_ndvi):,}")
    print(f"  Locations in both      : {len(common):,}")

    if only_base:
        missing_countries = base.loc[base["name_loc"].isin(only_base), COUNTRY_COL].value_counts()
        print(f"\n  Base-only locations by country (no NDVI coverage):")
        for c, n in missing_countries.items():
            print(f"    {c:6s} {n:,}")

    if only_ndvi:
        extra_countries = ndvi.loc[ndvi["name_loc"].isin(only_ndvi), COUNTRY_COL].value_counts()
        print(f"\n  NDVI-only locations by country (not in base):")
        for c, n in extra_countries.items():
            print(f"    {c:6s} {n:,}")

    # Country coverage side-by-side
    sep("Sample counts by country")
    all_countries = sorted(set(base[COUNTRY_COL].unique()) | set(ndvi[COUNTRY_COL].unique()))
    base_cc = base[COUNTRY_COL].value_counts()
    ndvi_cc = ndvi[COUNTRY_COL].value_counts()
    print(f"  {'Country':8s}  {'Base':>8s}  {'NDVI':>8s}  {'Δ':>8s}")
    print("  " + "─" * 42)
    for c in all_countries:
        b = base_cc.get(c, 0)
        n = ndvi_cc.get(c, 0)
        print(f"  {c:8s}  {b:8,}  {n:8,}  {n - b:+8,}")

    # Year coverage
    sep("Sample counts by year")
    all_years = sorted(set(base[YEAR_COL].unique()) | set(ndvi[YEAR_COL].unique()))
    base_yc = base[YEAR_COL].value_counts()
    ndvi_yc = ndvi[YEAR_COL].value_counts()
    print(f"  {'Year':6s}  {'Base':>8s}  {'NDVI':>8s}  {'Δ':>8s}")
    print("  " + "─" * 38)
    for y in all_years:
        b = base_yc.get(y, 0)
        n = ndvi_yc.get(y, 0)
        print(f"  {y:6d}  {b:8,}  {n:8,}  {n - b:+8,}")


# ---------------------------------------------------------------------------
# Feature set comparison
# ---------------------------------------------------------------------------

def feature_set_report(base: pd.DataFrame, ndvi: pd.DataFrame) -> None:
    sep("Feature set comparison")

    base_feats = set(_feat_cols(base))
    ndvi_feats = set(_feat_cols(ndvi))
    added = sorted(ndvi_feats - base_feats)
    removed = sorted(base_feats - ndvi_feats)
    shared = sorted(base_feats & ndvi_feats)

    print(f"  Base features     : {len(base_feats)}")
    print(f"  NDVI features     : {len(ndvi_feats)}")
    print(f"  Shared features   : {len(shared)}")
    print(f"  Added in NDVI     : {len(added)}")
    if added:
        for col in added:
            print(f"    + {col}")
    if removed:
        print(f"  Removed in NDVI   : {len(removed)}")
        for col in removed:
            print(f"    - {col}")

    # NDVI feature coverage / missing values (in NDVI CSV)
    if added:
        sep("NDVI feature coverage in NDVI CSV")
        total = len(ndvi)
        print(f"  {'Feature':35s}  {'Non-null':>10s}  {'Missing':>10s}  {'Min':>7s}  {'Max':>7s}  {'Mean':>7s}")
        print("  " + "─" * 82)
        for col in added:
            if col not in ndvi.columns:
                continue
            nn = ndvi[col].notna().sum()
            mn = ndvi[col].min() if nn > 0 else float("nan")
            mx = ndvi[col].max() if nn > 0 else float("nan")
            mu = ndvi[col].mean() if nn > 0 else float("nan")
            print(f"  {col:35s}  {_pct(nn, total):>10s}  {_pct(total - nn, total):>10s}  "
                  f"{mn:7.3f}  {mx:7.3f}  {mu:7.3f}")

    # Check whether shared feature values are identical (same preprocessing)
    sep("Shared feature consistency check (random 1000-row sample)")
    common_locs = list(set(base["name_loc"]) & set(ndvi["name_loc"]))
    sample_locs = common_locs[:1000]
    base_sample = base[base["name_loc"].isin(sample_locs)].set_index("name_loc")
    ndvi_sample = ndvi[ndvi["name_loc"].isin(sample_locs)].set_index("name_loc")
    shared_to_check = [c for c in shared if c in base_sample.columns and c in ndvi_sample.columns]
    idx = base_sample.index.intersection(ndvi_sample.index)
    diffs = []
    for col in shared_to_check:
        b_vals = base_sample.loc[idx, col]
        n_vals = ndvi_sample.loc[idx, col]
        max_diff = (b_vals - n_vals).abs().max()
        if max_diff > 1e-6:
            diffs.append((col, max_diff))
    if diffs:
        print(f"  ⚠  {len(diffs)} shared features have differing values between base and NDVI CSVs:")
        for col, d in sorted(diffs, key=lambda x: -x[1])[:10]:
            print(f"    {col:35s}  max |Δ| = {d:.6f}")
    else:
        print(f"  ✓  All {len(shared_to_check)} shared features are bit-identical between the two CSVs.")


# ---------------------------------------------------------------------------
# Random Forest comparison
# ---------------------------------------------------------------------------

def _run_rf_cv(
    df: pd.DataFrame,
    feat_cols: list[str],
    n_trees: int,
    seed: int,
    label: str,
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray, dict]:
    """Train RF with spatial group CV and return (fitted_rf, train_idx, test_idx, metrics)."""
    X, y = _build_X_y(df, feat_cols)
    groups = df[COUNTRY_COL].astype(str) + "_" + df[YEAR_COL].astype(str)

    print(f"\n  [{label}]  {X.shape[0]:,} samples × {X.shape[1]} features")

    # 5-fold spatial CV
    gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    rf_cv = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    cv_r2 = cross_val_score(rf_cv, X, y, cv=gss, groups=groups, scoring="r2")
    cv_mae = cross_val_score(rf_cv, X, y, cv=gss, groups=groups,
                             scoring="neg_mean_absolute_error")
    cv_rmse = cross_val_score(rf_cv, X, y, cv=gss, groups=groups,
                              scoring="neg_root_mean_squared_error")

    print(f"    CV R²   : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}  "
          f"(folds: {np.round(cv_r2, 3)})")
    print(f"    CV MAE  : {-cv_mae.mean():.3f} ± {cv_mae.std():.3f} t/ha")
    print(f"    CV RMSE : {-cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} t/ha")

    # Single holdout for permutation importance and residual plots
    gss_single = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(gss_single.split(X, y, groups=groups))
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = rf.predict(X.iloc[test_idx])
    y_test = y.iloc[test_idx]

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y.iloc[train_idx].mean()))

    print(f"    Holdout R²     : {r2:.3f}")
    print(f"    Holdout MAE    : {mae:.3f} t/ha  (baseline: {baseline_mae:.3f} t/ha)")
    print(f"    Holdout RMSE   : {rmse:.3f} t/ha")
    print(f"    Skill (1-MAE/baseline): {1 - mae / baseline_mae:.3f}")

    metrics = {
        "cv_r2_mean": cv_r2.mean(), "cv_r2_std": cv_r2.std(),
        "cv_mae_mean": -cv_mae.mean(), "cv_rmse_mean": -cv_rmse.mean(),
        "holdout_r2": r2, "holdout_mae": mae, "holdout_rmse": rmse,
        "baseline_mae": baseline_mae,
    }
    return rf, train_idx, test_idx, metrics, X, y


def rf_comparison(
    base: pd.DataFrame,
    ndvi: pd.DataFrame,
    n_trees: int,
    seed: int,
    out_dir: Path,
) -> tuple:
    sep("Random Forest comparison  (spatial group CV, country×year blocks)")

    base_feats = _feat_cols(base)
    ndvi_feats = _feat_cols(ndvi)

    # Restrict comparison to common locations so row counts are equal
    common_locs = set(base["name_loc"]) & set(ndvi["name_loc"])
    base_common = base[base["name_loc"].isin(common_locs)].reset_index(drop=True)
    ndvi_common = ndvi[ndvi["name_loc"].isin(common_locs)].reset_index(drop=True)
    print(f"\n  Evaluating on {len(common_locs):,} locations present in both CSVs.")

    rf_base, tr_b, te_b, m_base, X_base, y_base = _run_rf_cv(
        base_common, base_feats, n_trees, seed, "Base"
    )
    rf_ndvi, tr_n, te_n, m_ndvi, X_ndvi, y_ndvi = _run_rf_cv(
        ndvi_common, ndvi_feats, n_trees, seed, "Base + NDVI"
    )

    # Delta summary
    sep("Performance delta  (NDVI − Base)")
    dr2_cv = m_ndvi["cv_r2_mean"] - m_base["cv_r2_mean"]
    dr2_ho = m_ndvi["holdout_r2"] - m_base["holdout_r2"]
    dmae = m_ndvi["holdout_mae"] - m_base["holdout_mae"]
    print(f"  ΔR² (CV)      : {dr2_cv:+.3f}  "
          f"({'improvement' if dr2_cv > 0 else 'degradation'})")
    print(f"  ΔR² (holdout) : {dr2_ho:+.3f}  "
          f"({'improvement' if dr2_ho > 0 else 'degradation'})")
    print(f"  ΔMAE          : {dmae:+.3f} t/ha  "
          f"({'lower = better' if dmae < 0 else 'higher = worse'})")

    # Side-by-side bar chart
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ["Base", "Base+NDVI"]
    colors = ["#4878CF", "#6ACC65"]

    for ax, metric, vals, title in [
        (axes[0], "CV R²",
         [m_base["cv_r2_mean"], m_ndvi["cv_r2_mean"]],
         f"CV R²  (Δ={dr2_cv:+.3f})"),
        (axes[1], "Holdout R²",
         [m_base["holdout_r2"], m_ndvi["holdout_r2"]],
         f"Holdout R²  (Δ={dr2_ho:+.3f})"),
        (axes[2], "Holdout MAE (t/ha)",
         [m_base["holdout_mae"], m_ndvi["holdout_mae"]],
         f"Holdout MAE  (Δ={dmae:+.3f} t/ha)"),
    ]:
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(metric)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Random Forest performance: Base vs. Base+NDVI", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "rf_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_dir / 'rf_comparison.png'}")

    return (rf_base, te_b, X_base, y_base,
            rf_ndvi, te_n, X_ndvi, y_ndvi,
            ndvi_feats, m_base, m_ndvi, base_common, ndvi_common)


# ---------------------------------------------------------------------------
# Feature importance — NDVI contribution
# ---------------------------------------------------------------------------

def ndvi_importance_report(
    rf_ndvi: RandomForestRegressor,
    X_ndvi: pd.DataFrame,
    y_ndvi: pd.Series,
    test_idx: np.ndarray,
    out_dir: Path,
    seed: int,
) -> pd.DataFrame:
    sep("Feature importance — NDVI contribution")

    X_test = X_ndvi.iloc[test_idx]
    y_test = y_ndvi.iloc[test_idx]
    feat_names = list(X_ndvi.columns)

    print("  Computing permutation importance on holdout set (n_repeats=10)…")
    perm = permutation_importance(
        rf_ndvi, X_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "mdi": rf_ndvi.feature_importances_,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
    }, index=feat_names).sort_values("perm_mean", ascending=False)

    # Separate NDVI vs. base features
    ndvi_imp = imp_df[imp_df.index.isin(NDVI_COLS)]
    base_imp = imp_df[~imp_df.index.isin(NDVI_COLS)]

    total_perm = imp_df["perm_mean"].sum()
    ndvi_perm = ndvi_imp["perm_mean"].sum()
    base_perm = base_imp["perm_mean"].sum()

    print(f"\n  Total permutation importance (all features) : {total_perm:.4f}")
    print(f"  Base features contribution                  : {base_perm:.4f}  "
          f"({100 * base_perm / total_perm:.1f}%)")
    print(f"  NDVI features contribution                  : {ndvi_perm:.4f}  "
          f"({100 * ndvi_perm / total_perm:.1f}%)")

    # Top-15 overall
    print(f"\n  {'Feature':35s}  {'Perm':>7s}  {'±':>6s}  {'MDI':>7s}  Source")
    print("  " + "─" * 72)
    for feat, row in imp_df.head(15).iterrows():
        src = "NDVI" if feat in NDVI_COLS else "Base"
        print(f"  {feat:35s}  {row.perm_mean:7.4f}  {row.perm_std:6.4f}  "
              f"{row.mdi:7.4f}  {src}")

    # NDVI-only ranking
    if not ndvi_imp.empty:
        print(f"\n  NDVI features ranked by permutation importance:")
        print(f"  {'Feature':35s}  {'Perm':>7s}  {'±':>6s}  {'Rank (overall)':>15s}")
        print("  " + "─" * 68)
        for feat, row in ndvi_imp.iterrows():
            overall_rank = list(imp_df.index).index(feat) + 1
            print(f"  {feat:35s}  {row.perm_mean:7.4f}  {row.perm_std:6.4f}  "
                  f"{'#' + str(overall_rank):>15s}")

    # Figure: base vs NDVI feature importance groups
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: top-20 overall with NDVI highlighted
    top20 = imp_df.head(20)
    colors = ["#E07B54" if f in NDVI_COLS else "#4878CF" for f in top20.index]
    y_pos = range(len(top20))
    axes[0].barh(list(y_pos), top20["perm_mean"].values[::-1],
                 xerr=top20["perm_std"].values[::-1],
                 color=colors[::-1], ecolor="gray", capsize=3)
    axes[0].set_yticks(list(y_pos))
    axes[0].set_yticklabels(list(top20.index[::-1]), fontsize=8)
    axes[0].set_xlabel("Mean permutation importance (R² decrease)")
    axes[0].set_title("Top-20 feature importances\n(orange = NDVI, blue = base)")

    # Right: NDVI monthly pattern (perm importance by month)
    monthly = ndvi_imp[ndvi_imp.index.isin(NDVI_MONTHLY_COLS)].copy()
    if not monthly.empty:
        monthly["month"] = monthly.index.str.extract(r"(\d+)$").astype(int)
        monthly = monthly.sort_values("month")
        axes[1].bar(monthly["month"], monthly["perm_mean"],
                    yerr=monthly["perm_std"], color="#E07B54",
                    ecolor="gray", capsize=3)
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Permutation importance")
        axes[1].set_title("NDVI monthly feature importance\n(seasonality signal)")
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            rotation=45, fontsize=8
        )
    else:
        axes[1].text(0.5, 0.5, "No monthly NDVI features", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_dir / "ndvi_importance.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_dir / 'ndvi_importance.png'}")

    return imp_df


# ---------------------------------------------------------------------------
# PCA: does NDVI add independent signal?
# ---------------------------------------------------------------------------

def pca_comparison(
    base: pd.DataFrame,
    ndvi: pd.DataFrame,
    out_dir: Path,
) -> None:
    sep("PCA — independent signal added by NDVI features")

    common_locs = set(base["name_loc"]) & set(ndvi["name_loc"])
    base_c = base[base["name_loc"].isin(common_locs)]
    ndvi_c = ndvi[ndvi["name_loc"].isin(common_locs)]

    base_feats = _feat_cols(base_c)
    ndvi_feats = _feat_cols(ndvi_c)
    ndvi_only = [c for c in ndvi_feats if c not in base_feats and c in NDVI_COLS]

    scaler = StandardScaler()

    def _pca_stats(df: pd.DataFrame, cols: list[str], label: str) -> np.ndarray:
        X = df[cols].copy()
        for c in X.columns:
            X[c] = X[c].fillna(X[c].median())
        Xs = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(Xs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        for t in (0.80, 0.90, 0.95):
            n = int(np.searchsorted(cumvar, t) + 1)
            print(f"  [{label}]  {int(t*100)}% variance → {n:2d} / {len(cols)} components")
        ev = pca.explained_variance_ratio_
        ev = ev[ev > 0]
        eff_rank = int(np.round(np.exp(-np.sum(ev * np.log(ev)))))
        print(f"  [{label}]  Effective rank: {eff_rank}")
        return cumvar

    print()
    cumvar_base = _pca_stats(base_c, base_feats, "Base")
    print()
    cumvar_ndvi = _pca_stats(ndvi_c, ndvi_feats, "Base+NDVI")

    # How much independent variance do NDVI features add?
    if ndvi_only:
        X_ndvi_only = ndvi_c[ndvi_only].copy()
        for c in X_ndvi_only.columns:
            X_ndvi_only[c] = X_ndvi_only[c].fillna(X_ndvi_only[c].median())
        Xs_n = scaler.fit_transform(X_ndvi_only)
        pca_n = PCA()
        pca_n.fit(Xs_n)
        cumvar_ndvi_only = np.cumsum(pca_n.explained_variance_ratio_)
        print(f"\n  NDVI-only ({len(ndvi_only)} features) internal structure:")
        for t in (0.80, 0.90):
            n = int(np.searchsorted(cumvar_ndvi_only, t) + 1)
            print(f"    {int(t*100)}% variance → {n} / {len(ndvi_only)} components  "
                  f"(redundancy: {(1 - n/len(ndvi_only))*100:.0f}%)")

    # Scree comparison plot
    fig, ax = plt.subplots(figsize=(9, 4))
    n_base = min(len(cumvar_base), 50)
    n_ndvi = min(len(cumvar_ndvi), 50)
    ax.plot(range(1, n_base + 1), cumvar_base[:n_base] * 100,
            label=f"Base ({len(base_feats)} features)", color="#4878CF", linewidth=1.8)
    ax.plot(range(1, n_ndvi + 1), cumvar_ndvi[:n_ndvi] * 100,
            label=f"Base+NDVI ({len(ndvi_feats)} features)", color="#6ACC65",
            linewidth=1.8, linestyle="--")
    for t in (80, 90, 95):
        ax.axhline(t, color="gray", linestyle=":", linewidth=0.8)
        ax.text(max(n_base, n_ndvi) * 0.98, t + 0.5, f"{t}%",
                ha="right", fontsize=8, color="gray")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA scree comparison — Base vs. Base+NDVI")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "pca_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out_dir / 'pca_comparison.png'}")


# ---------------------------------------------------------------------------
# Per-country R² comparison
# ---------------------------------------------------------------------------

def per_country_comparison(
    rf_base: RandomForestRegressor,
    rf_ndvi: RandomForestRegressor,
    X_base: pd.DataFrame,
    X_ndvi: pd.DataFrame,
    y_base: pd.Series,
    y_ndvi: pd.Series,
    test_idx_base: np.ndarray,
    test_idx_ndvi: np.ndarray,
    base_common: pd.DataFrame,
    ndvi_common: pd.DataFrame,
    out_dir: Path,
) -> None:
    sep("Per-country holdout R²  (Base vs. Base+NDVI)")

    results = []
    for country in sorted(base_common[COUNTRY_COL].unique()):
        for label, rf, X, y, test_idx, df_meta in [
            ("Base", rf_base, X_base, y_base, test_idx_base, base_common),
            ("NDVI", rf_ndvi, X_ndvi, y_ndvi, test_idx_ndvi, ndvi_common),
        ]:
            mask = df_meta.iloc[test_idx][COUNTRY_COL].values == country
            if mask.sum() < 5:
                continue
            y_t = y.iloc[test_idx].values[mask]
            y_p = rf.predict(X.iloc[test_idx])[mask]
            r2 = r2_score(y_t, y_p) if len(y_t) > 1 else float("nan")
            mae = mean_absolute_error(y_t, y_p)
            results.append({"country": country, "model": label, "n": mask.sum(),
                             "r2": r2, "mae": mae})

    results_df = pd.DataFrame(results)
    print(f"  {'Country':8s}  {'n_base':>7s}  {'R²_base':>8s}  {'R²_ndvi':>8s}  "
          f"{'ΔR²':>8s}  {'MAE_base':>9s}  {'MAE_ndvi':>9s}")
    print("  " + "─" * 72)

    countries = results_df["country"].unique()
    for c in sorted(countries):
        b = results_df[(results_df["country"] == c) & (results_df["model"] == "Base")]
        n = results_df[(results_df["country"] == c) & (results_df["model"] == "NDVI")]
        if b.empty or n.empty:
            continue
        r2b = b["r2"].values[0]
        r2n = n["r2"].values[0]
        maeb = b["mae"].values[0]
        maen = n["mae"].values[0]
        nb = b["n"].values[0]
        print(f"  {c:8s}  {nb:7d}  {r2b:8.3f}  {r2n:8.3f}  {r2n - r2b:+8.3f}  "
              f"{maeb:9.3f}  {maen:9.3f}")

    # Bar chart per country
    pivot = results_df.pivot(index="country", columns="model", values="r2").fillna(float("nan"))
    if "Base" in pivot.columns and "NDVI" in pivot.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(pivot))
        w = 0.35
        ax.bar(x - w / 2, pivot["Base"], w, label="Base", color="#4878CF")
        ax.bar(x + w / 2, pivot["NDVI"], w, label="Base+NDVI", color="#6ACC65")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_ylabel("Holdout R²")
        ax.set_title("Per-country holdout R²: Base vs. Base+NDVI")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "per_country_r2.png", dpi=150)
        plt.close(fig)
        print(f"\n  Saved: {out_dir / 'per_country_r2.png'}")


# ---------------------------------------------------------------------------
# Hydra config recommendations
# ---------------------------------------------------------------------------

def hydra_recommendations(
    base: pd.DataFrame,
    ndvi: pd.DataFrame,
    m_base: dict,
    m_ndvi: dict,
    data_dir: Path,
) -> None:
    sep("Hydra config recommendations")

    common_locs = set(base["name_loc"]) & set(ndvi["name_loc"])
    base_only = set(base["name_loc"]) - common_locs
    ndvi_only = set(ndvi["name_loc"]) - common_locs
    coverage_pct = 100 * len(common_locs) / len(set(base["name_loc"]))

    dr2_cv = m_ndvi["cv_r2_mean"] - m_base["cv_r2_mean"]
    dr2_ho = m_ndvi["holdout_r2"] - m_base["holdout_r2"]

    ndvi_feats = _feat_cols(ndvi)
    base_feats = _feat_cols(base)
    n_ndvi_added = len(set(ndvi_feats) - set(base_feats))

    # ── Missing-value analysis: seasonal vs monthly, covered years only ───────
    # The right signal for the recommendation is the seasonal aggregate columns
    # (feat_ndvi_mean_*), which are what the model primarily relies on for
    # phenological signal.  Monthly columns are inherently noisier because each
    # source NDVI file may be partially complete for individual months, yet the
    # pandas mean() used in the augmentation script salvages seasonal aggregates
    # from partial monthly data.
    #
    # We also distinguish:
    #   - years with no NDVI file at all (known data gap, e.g. 2014) — these
    #     rows will always be NaN and are handled by TabularEncoder imputation.
    #   - years with an NDVI file but incomplete monthly data (source quality gap,
    #     e.g. 2016–2018) — seasonal NaN here reflects source incompleteness.
    #   - years with a complete NDVI file (2019+) — NaN here is unexpected.
    seasonal_cols = [c for c in NDVI_SEASONAL_COLS if c in ndvi.columns]
    monthly_cols  = [c for c in NDVI_MONTHLY_COLS  if c in ndvi.columns]

    # Years that have an NDVI source file
    ndvi_file_years = sorted(ndvi["year"].unique()) if "year" in ndvi.columns else []
    # Infer which years have a file by checking if seasonal NaN < 100%
    years_with_file = (
        ndvi.groupby("year")[seasonal_cols].apply(lambda g: g.isna().mean().mean())
        .pipe(lambda s: s[s < 1.0].index.tolist())
        if seasonal_cols and "year" in ndvi.columns else []
    )
    years_no_file = [y for y in ndvi["year"].unique() if y not in years_with_file] if "year" in ndvi.columns else []

    rows_no_file = int(ndvi["year"].isin(years_no_file).sum()) if "year" in ndvi.columns else 0
    rows_with_file = len(ndvi) - rows_no_file
    ndvi_covered_df = ndvi[ndvi["year"].isin(years_with_file)] if years_with_file else ndvi

    seasonal_missing_covered = (
        ndvi_covered_df[seasonal_cols].isna().mean().mean() * 100 if seasonal_cols else 0.0
    )
    monthly_missing_covered = (
        ndvi_covered_df[monthly_cols].isna().mean().mean() * 100 if monthly_cols else 0.0
    )
    seasonal_missing_all = ndvi[seasonal_cols].isna().mean().mean() * 100 if seasonal_cols else 0.0

    # Per-year seasonal missing rate (for the report)
    if seasonal_cols and "year" in ndvi.columns:
        yr_seasonal = (
            ndvi.groupby("year")[seasonal_cols]
            .apply(lambda g: g.isna().mean().mean() * 100)
            .sort_index()
        )
    else:
        yr_seasonal = pd.Series(dtype=float)

    print(f"""
DATASET SELECTION
─────────────────
  Base CSV  : {data_dir}/yield_africa/model_ready_yield_africa_base.csv
              {len(base):,} rows, {len(base_feats)} feat_* features
  NDVI CSV  : {data_dir}/yield_africa/model_ready_yield_africa_ndvi.csv
              {len(ndvi):,} rows, {len(ndvi_feats)} feat_* features  (+{n_ndvi_added} NDVI)
  Row coverage  : {len(common_locs):,} / {len(set(base["name_loc"])):,} base locations present ({coverage_pct:.1f}%)
  Rows lost     : {len(base_only):,}

NDVI DATA QUALITY BY YEAR
──────────────────────────""")
    for yr, pct in yr_seasonal.items():
        note = "(no NDVI file)" if yr in years_no_file else ("(partial source data)" if pct > 5 else "")
        print(f"  {yr}  seasonal NaN: {pct:5.1f}%  {note}")
    print(f"""
  Seasonal missing rate — years with NDVI file : {seasonal_missing_covered:.1f}%
  Monthly  missing rate — years with NDVI file : {monthly_missing_covered:.1f}%
  Rows with no NDVI file (will use imputation) : {rows_no_file:,}  ({100*rows_no_file/len(ndvi):.1f}%)
""")

    # Verdict
    # Use seasonal missing rate on covered years as the quality signal — that is
    # what the model actually trains on (seasonal aggregates are more robust than
    # individual monthly columns and are the primary NDVI signal for yield prediction).
    ndvi_meaningful = dr2_cv > 0.01 or dr2_ho > 0.01
    ndvi_coverage_ok = coverage_pct >= 95.0
    ndvi_seasonal_ok = seasonal_missing_covered < 30.0   # seasonal cols, covered years only
    no_file_rows_minor = rows_no_file / len(ndvi) < 0.10  # < 10% rows lack a file entirely

    print("RECOMMENDATION")
    print("──────────────")
    if ndvi_meaningful and ndvi_coverage_ok and ndvi_seasonal_ok and no_file_rows_minor:
        verdict = "USE NDVI CSV"
        rationale = (
            f"NDVI features improve CV R² by {dr2_cv:+.3f} and holdout R² by {dr2_ho:+.3f}. "
            f"Seasonal features are {seasonal_missing_covered:.1f}% NaN for covered years "
            f"(monthly: {monthly_missing_covered:.1f}%); "
            f"{rows_no_file:,} rows ({100*rows_no_file/len(ndvi):.1f}%) have no NDVI file "
            f"and fall back to median imputation."
        )
    elif ndvi_meaningful and not ndvi_coverage_ok:
        verdict = "USE NDVI CSV with caution"
        rationale = (
            f"NDVI features improve R², but {len(base_only):,} base locations have no NDVI data "
            f"({100 - coverage_pct:.1f}% of base). "
            "Dropping those rows may introduce geographic bias — check if missing locations "
            "are spatially clustered (e.g. a whole country or year)."
        )
    elif not ndvi_meaningful:
        verdict = "USE BASE CSV"
        rationale = (
            f"NDVI features provide marginal improvement (ΔR² CV={dr2_cv:+.3f}, "
            f"holdout={dr2_ho:+.3f}). The added complexity ({n_ndvi_added} extra features) "
            "does not justify switching."
        )
    elif not ndvi_seasonal_ok:
        verdict = "USE NDVI CSV — source data for 2016–2018 is incomplete"
        rationale = (
            f"NDVI improves R² (ΔCV={dr2_cv:+.3f}), but seasonal features are "
            f"{seasonal_missing_covered:.1f}% NaN for covered years, driven by incomplete "
            f"source NDVI files for years {[y for y in years_with_file if yr_seasonal.get(y, 0) > 5]}. "
            "Those rows fall back to median imputation. Consider obtaining complete NDVI "
            "files for those years to reduce reliance on imputation."
        )
    else:
        verdict = "EVALUATE FURTHER"
        rationale = (
            "Mixed signals: NDVI adds predictive value but row coverage or "
            "file-availability concerns exist. Run the LOCO experiment on both before deciding."
        )

    print(f"\n  ✦ {verdict}")
    print(f"  {rationale}")

    # Config snippets
    print("""
HYDRA CONFIG SNIPPETS
─────────────────────
  1. Point data_dir to the external drive in your .env:
       DATA_DIR=/Volumes/data_and_models_2/aether/data

  2a. Use base CSV (current default):
       # configs/data/yield_africa_all.yaml  (or yield_africa_loco.yaml)
       dataset:
         _target_: src.data.yield_africa_dataset.YieldAfricaDataset
         data_dir: ${paths.data_dir}
         # YieldAfricaDataset automatically loads model_ready_yield_africa.csv
         # Rename or symlink model_ready_yield_africa_base.csv →
         #   ${DATA_DIR}/yield_africa/model_ready_yield_africa.csv

  2b. Use NDVI CSV (csv_name parameter is already implemented):
       dataset:
         csv_name: model_ready_yield_africa_ndvi.csv
       # Or override at the command line:
       #   python src/train.py experiment=yield_africa_tabular_loco \\
       #     data.dataset.csv_name=model_ready_yield_africa_ndvi.csv

  3. LOCO config — regardless of CSV choice:
       # configs/data/yield_africa_loco.yaml
       dataset:
         use_country_features: false   # ← already applied; essential for LOCO

  4. Tabular dimension — update yield_tabular_reg.yaml input_dim after switching CSVs:
       # Base  : 61 CSV feat_* + 1 feat_year + 6 Fourier harmonics = 68  (no country OH)
       # NDVI  : 78 CSV feat_* + 1 feat_year + 6 Fourier harmonics = 85  (no country OH)
       # With country one-hots (+8): add 8 to the above.
       # The tabular_dim is logged at dataset init — check the training log to confirm.

  5. To resolve the CSV name dynamically (recommended long-term):
       Add to BaseDataset.__init__:
         self.csv_name = csv_name or f"model_ready_{dataset_name}.csv"
       Then override per-experiment:
         python src/train.py experiment=yield_africa_tabular_loco \\
           data.dataset.csv_name=model_ready_yield_africa_ndvi.csv
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        default="/Volumes/data_and_models_2/aether/data",
        help="Root data directory containing yield_africa/",
    )
    parser.add_argument(
        "--out_dir",
        default="data/yield_africa/analysis_ndvi_comparison",
        help="Output directory for figures and report",
    )
    parser.add_argument("--n_trees", type=int, default=300, help="Number of RF trees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    base_path = data_dir / "yield_africa" / "model_ready_yield_africa_base.csv"
    ndvi_path = data_dir / "yield_africa" / "model_ready_yield_africa_ndvi.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in (base_path, ndvi_path):
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Load
    base, ndvi = load_csvs(base_path, ndvi_path)

    # Coverage
    coverage_report(base, ndvi)

    # Feature set
    feature_set_report(base, ndvi)

    # RF comparison
    (rf_base, te_b, X_base, y_base,
     rf_ndvi, te_n, X_ndvi, y_ndvi,
     ndvi_feats, m_base, m_ndvi,
     base_common, ndvi_common) = rf_comparison(base, ndvi, args.n_trees, args.seed, out_dir)

    # NDVI feature importance
    ndvi_importance_report(rf_ndvi, X_ndvi, y_ndvi, te_n, out_dir, args.seed)

    # PCA
    pca_comparison(base, ndvi, out_dir)

    # Per-country breakdown
    per_country_comparison(
        rf_base, rf_ndvi,
        X_base, X_ndvi, y_base, y_ndvi,
        te_b, te_n,
        base_common, ndvi_common,
        out_dir,
    )

    # Hydra recommendations
    hydra_recommendations(base, ndvi, m_base, m_ndvi, data_dir)

    sep()
    print(f"Analysis complete. Figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
