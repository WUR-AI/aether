"""Rename existing tessera tiles to include a year suffix.

Existing naming: tessera_{name_loc}.npy
New naming:      tessera_{name_loc}_{year}.npy

Run this script once before deploying Tasks 2-7 of the dual-year tessera plan.
Tiles already carrying a four-digit year suffix are skipped automatically, so
the script is safe to re-run.

Usage
-----
    # Preview what would be renamed (default — no files touched)
    python src/data_preprocessing/yield_africa_tessera_rename.py \\
        --data_dir /path/to/external/data

    # Actually rename
    python src/data_preprocessing/yield_africa_tessera_rename.py \\
        --data_dir /path/to/external/data --no-dry-run
"""

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

log = logging.getLogger(__name__)

DATASET_NAME = "yield_africa"
MODEL_READY_CSV = f"model_ready_{DATASET_NAME}.csv"

# Matches a four-digit year at the end of the stem, e.g. _2020 in tessera_RWA_24803_2020
_YEAR_SUFFIX_RE = re.compile(r"_\d{4}$")


def rename_tessera_tiles(data_dir: str, dry_run: bool = True) -> None:
    dataset_dir = Path(data_dir) / DATASET_NAME
    csv_path = dataset_dir / MODEL_READY_CSV
    tessera_dir = dataset_dir / "eo" / "tessera"

    if not csv_path.exists():
        raise FileNotFoundError(f"Model-ready CSV not found: {csv_path}")
    if not tessera_dir.exists():
        raise FileNotFoundError(f"Tessera directory not found: {tessera_dir}")

    df = pd.read_csv(csv_path)
    name_loc_to_year: dict[str, int] = dict(zip(df["name_loc"], df["year"].astype(int)))

    n_renamed = 0
    n_already_done = 0
    n_unmatched = 0

    for src in sorted(tessera_dir.glob("tessera_*.npy")):
        name_loc = src.stem.removeprefix("tessera_")

        if _YEAR_SUFFIX_RE.search(name_loc):
            n_already_done += 1
            continue

        year = name_loc_to_year.get(name_loc)
        if year is None:
            log.warning("No year found in CSV for %s — leaving untouched", src.name)
            n_unmatched += 1
            continue

        dst = src.with_name(f"tessera_{name_loc}_{year}.npy")
        print(f"  {'[dry-run] ' if dry_run else ''}rename  {src.name}  ->  {dst.name}")
        if not dry_run:
            src.rename(dst)
        n_renamed += 1

    print(
        f"\nSummary: {n_renamed} to rename, "
        f"{n_already_done} already up-to-date, "
        f"{n_unmatched} unmatched."
    )
    if dry_run and n_renamed > 0:
        print("Re-run with --no-dry-run to apply changes.")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Rename tessera_<name_loc>.npy tiles to tessera_<name_loc>_<year>.npy. "
            "Dry-run is the default — pass --no-dry-run to apply changes."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root data directory (same as paths.data_dir in configs).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preview renames without touching files (default: True). Pass --no-dry-run to apply.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — no files will be renamed. Pass --no-dry-run to apply.\n")

    rename_tessera_tiles(data_dir=args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
