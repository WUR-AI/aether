"""Merge augmented yield Africa CSVs into a single combined CSV.

Combines the base model-ready CSV with any combination of NDVI and AgERA5
augmented CSVs.  Shared columns (all base feat_* columns, plus name_loc, lat,
lon, year, country, target_*, aux_*) appear exactly once in the output.
Augmentation-specific columns are added alongside without duplication.

Usage:
    # Base + NDVI + AgERA5 (full merge)
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

Optional arguments:
    --base_csv    Path to model_ready_yield_africa.csv (required)
    --ndvi_csv    Path to model_ready_yield_africa_ndvi.csv (optional)
    --agera5_csv  Path to model_ready_yield_africa_agera5.csv (optional)
    --out_csv     Output path for merged CSV
                  (default: model_ready_yield_africa_merged.csv beside --base_csv)
    --how         Join type applied when augmentation rows don't fully cover base:
                  'inner' keeps only rows present in all provided CSVs,
                  'left' keeps all base rows and fills missing augmentation
                  columns with NaN (default: left)
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

JOIN_KEYS = ["name_loc", "year"]


# ---------------------------------------------------------------------------
# Helpers
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
    print(f"  {label:12s}: {df.shape[0]:,} rows × {df.shape[1]} cols  "
          f"({len(_feat_cols(df))} feat_*)  [{path.name}]")
    return df


def _augmentation_only_cols(aug_df: pd.DataFrame, base_cols: set[str]) -> list[str]:
    """Return columns in aug_df that are not already in base_cols.

    Always includes sentinel columns like 'agera5_fetched' even though they
    don't start with 'feat_', as long as they aren't in the base.
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

    Only the columns that are genuinely new in each augmentation CSV are
    attached; any column already present in the running result is skipped.
    """
    result = base_df.copy()
    result_cols = set(result.columns)

    for label, aug_df in augmentations:
        sep(f"Merging {label}")

        new_cols = _augmentation_only_cols(aug_df, result_cols)
        if not new_cols:
            print(f"  No new columns found in {label} — skipping.")
            continue

        print(f"  New columns to add ({len(new_cols)}): "
              + ", ".join(new_cols[:8])
              + (" …" if len(new_cols) > 8 else ""))

        # Coverage check before merge
        base_keys = set(zip(result["name_loc"], result["year"]))
        aug_keys  = set(zip(aug_df["name_loc"],  aug_df["year"]))
        only_base = base_keys - aug_keys
        only_aug  = aug_keys  - base_keys
        if only_base:
            print(f"  {len(only_base):,} base rows have no match in {label} "
                  f"{'(will be NaN)' if how == 'left' else '(dropped by inner join)'}")
        if only_aug:
            print(f"  {len(only_aug):,} {label} rows have no match in base (ignored)")

        # Select only join keys + new columns from the augmentation dataframe
        merge_cols = JOIN_KEYS + new_cols
        # Guard: join keys might not be in new_cols list, ensure no duplicates
        merge_subset = aug_df[merge_cols].copy()

        result = result.merge(merge_subset, on=JOIN_KEYS, how=how, suffixes=("", f"_{label.lower()}"))

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
        c for c in result.columns
        if c not in base_df.columns and not c.startswith("feat_")
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
        print(f"  {label}: {complete:,} / {len(result):,} rows fully populated "
              f"({100 * complete / len(result):.1f}%)")

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
    parser.add_argument("--base_csv",   required=True,
                        help="Path to model_ready_yield_africa.csv (base)")
    parser.add_argument("--ndvi_csv",   default=None,
                        help="Path to model_ready_yield_africa_ndvi.csv (optional)")
    parser.add_argument("--agera5_csv", default=None,
                        help="Path to model_ready_yield_africa_agera5.csv (optional)")
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
    args = parser.parse_args()

    base_path = Path(args.base_csv)
    if not base_path.exists():
        print(f"ERROR: base CSV not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    if args.ndvi_csv is None and args.agera5_csv is None:
        print("ERROR: at least one of --ndvi_csv or --agera5_csv must be provided.",
              file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out_csv) if args.out_csv else base_path.parent / "model_ready_yield_africa_merged.csv"

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

    sep("Writing output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, float_format="%.7f")
    print(f"  Saved: {out_path.resolve()}")
    print(f"  Shape: {result.shape[0]:,} rows × {result.shape[1]} columns")

    sep()
    print("Merge complete.")


if __name__ == "__main__":
    main()
