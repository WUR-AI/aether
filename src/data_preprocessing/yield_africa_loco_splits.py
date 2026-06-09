"""Generate leave-one-country-out (LOCO) split files for the yield_africa dataset.

Location: src/data_preprocessing/yield_africa_loco_splits.py

For each held-out country one `.pth` file is written to
`{data_dir}/yield_africa/splits/split_loco_{COUNTRY}.pth`.

Split layout
------------
- test  : all records from the held-out country
- train : 80 % of records from the remaining countries (random, seeded)
- val   : 20 % of records from the remaining countries (random, seeded)

The files are consumed by BaseDataModule when `split_mode: from_file` and
`saved_split_file_name: split_loco_{COUNTRY}.pth`.

Usage
-----
    python src/data_preprocessing/yield_africa_loco_splits.py --data_dir data/
    python src/data_preprocessing/yield_africa_loco_splits.py --data_dir data/ --country KEN
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

# All countries present in the full dataset (must match _ALL_COUNTRIES in
# yield_africa_dataset.py so that the feature encoding is consistent).
ALL_COUNTRIES = ["BF", "BUR", "ETH", "KEN", "MAL", "RWA", "TAN", "ZAM"]

DATASET_NAME = "yield_africa"
MODEL_READY_CSV = f"model_ready_{DATASET_NAME}.csv"


def make_loco_split(
    df: pd.DataFrame,
    test_country: str,
    val_fraction: float = 0.2,
    seed: int = 12345,
) -> dict:
    """Return a split-indices dict for one held-out country.

    :param df: full model-ready dataframe (must contain 'country' and 'name_loc')
    :param test_country: country code to hold out as the test set
    :param val_fraction: fraction of the non-test pool to use for validation
    :param seed: random seed for the train/val shuffle
    :return: dict with 'train_indices', 'val_indices', 'test_indices' as pd.Series of name_locs
    """
    test_mask = df["country"] == test_country
    test_locs = df.loc[test_mask, "name_loc"].reset_index(drop=True)

    remaining = df.loc[~test_mask, "name_loc"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    shuffled = remaining.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = int(len(shuffled) * val_fraction)
    val_locs = shuffled.iloc[:n_val]
    train_locs = shuffled.iloc[n_val:]

    return {
        "train_indices": train_locs,
        "val_indices": val_locs,
        "test_indices": test_locs,
    }


def generate_splits(
    data_dir: str,
    countries: list[str] | None = None,
    val_fraction: float = 0.2,
    seed: int = 12345,
) -> None:
    """Generate and save LOCO split files for the requested countries.

    :param data_dir: root data directory (same as `paths.data_dir` in configs)
    :param countries: list of country codes to generate splits for; None means all
    :param val_fraction: fraction of non-test data to use for validation
    :param seed: random seed
    """
    dataset_dir = Path(data_dir) / DATASET_NAME
    csv_path = dataset_dir / MODEL_READY_CSV
    splits_dir = dataset_dir / "splits"

    if not csv_path.exists():
        raise FileNotFoundError(f"Model-ready CSV not found: {csv_path}")

    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "country" not in df.columns or "name_loc" not in df.columns:
        raise ValueError("CSV must contain 'country' and 'name_loc' columns")

    available = sorted(df["country"].unique().tolist())
    targets = countries if countries is not None else available

    for country in targets:
        if country not in available:
            log.warning(f"Country '{country}' not found in CSV (available: {available}), skipping")
            continue

        split = make_loco_split(df, country, val_fraction=val_fraction, seed=seed)
        n_train = len(split["train_indices"])
        n_val = len(split["val_indices"])
        n_test = len(split["test_indices"])

        out_path = splits_dir / f"split_loco_{country}.pth"
        torch.save(split, out_path)

        log.info(
            f"  {country}: train={n_train}, val={n_val}, test={n_test} " f"-> {out_path.name}"
        )
        print(
            f"  Saved split_loco_{country}.pth  " f"(train={n_train}, val={n_val}, test={n_test})"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate leave-one-country-out split files for yield_africa."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Root data directory (same as paths.data_dir in configs). Default: data/",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Single country code to generate a split for. Omit to generate all.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fraction of non-test records used for validation. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for the train/val shuffle. Default: 12345",
    )
    args = parser.parse_args()

    countries = [args.country] if args.country else None
    print(
        f"Generating LOCO splits  data_dir={args.data_dir}  "
        f"countries={countries or 'all'}  val_fraction={args.val_fraction}  seed={args.seed}"
    )
    generate_splits(
        data_dir=args.data_dir,
        countries=countries,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
