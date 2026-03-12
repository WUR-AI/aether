"""Generate spatial-cluster split files for the yield_africa dataset.

Location: src/data_preprocessing/yield_africa_spatial_splits.py

Uses DBSCAN with a haversine distance metric to group nearby field locations
into clusters, then assigns whole clusters to train/val/test so that no
geographically close points straddle a split boundary.

One `.pth` file is written per distance threshold to
`{data_dir}/yield_africa/splits/split_spatial_{distance_km}km.pth`.

Split layout
------------
- train : ~70 % of records (cluster-aligned)
- val   : ~15 % of records (cluster-aligned)
- test  : ~15 % of records (cluster-aligned)

Proportions are approximate because whole clusters are kept intact.

The files are consumed by BaseDataModule when `split_mode: from_file` and
`saved_split_file_name: split_spatial_{distance_km}km.pth`.

Usage
-----
    # Generate the default set of splits (10 km, 25 km, 50 km)
    python src/data_preprocessing/yield_africa_spatial_splits.py --data_dir data/

    # Generate a single split at a specific distance
    python src/data_preprocessing/yield_africa_spatial_splits.py --data_dir data/ --distance_km 25

    # Generate multiple distances in one run
    python src/data_preprocessing/yield_africa_spatial_splits.py --data_dir data/ --distance_km 10 25 50

Notes
-----
- DBSCAN uses sklearn's built-in haversine metric with a BallTree spatial index
  and n_jobs=-1, which is significantly faster than a Python geodesic lambda.
  Haversine vs. true geodesic error is < 0.1% at distances up to ~100 km.
- `min_samples=2` means a pair of fields within `distance_km` of each other
  forms a cluster; isolated fields each become their own singleton cluster.
- All clusters are kept intact across the split boundary, so the test set
  contains no locations geographically close to any training location.
"""

import argparse
import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

DATASET_NAME = "yield_africa"
MODEL_READY_CSV = f"model_ready_{DATASET_NAME}.csv"

# Default distances to generate when no --distance_km is supplied.
DEFAULT_DISTANCES_KM = [10, 25, 50]

# Split proportions (must sum to 1.0).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Fixed random seed for GroupShuffleSplit.
SEED = 12345


def make_spatial_split(
    df: pd.DataFrame,
    distance_m: int,
    train_val_test_split: tuple[float, float, float] = (TRAIN_FRAC, VAL_FRAC, TEST_FRAC),
    seed: int = SEED,
) -> dict:
    """Return a split-indices dict using DBSCAN spatial clustering.

    :param df: full model-ready dataframe (must contain 'lat', 'lon', 'name_loc')
    :param distance_m: DBSCAN eps in metres — pairs of fields closer than this
        value are assigned to the same cluster
    :param train_val_test_split: (train, val, test) proportions, must sum to 1.0
    :param seed: random seed for GroupShuffleSplit
    :return: dict with 'train_indices', 'val_indices', 'test_indices' as
        pd.Series of name_loc strings, plus 'clusters' as a numpy array of
        cluster labels (same length as df)
    """
    # Deduplicate to unique (lat, lon) locations before clustering.
    # yield_africa has ~9 rows per location (one per year); running DBSCAN on all
    # rows produces giant clusters whose row counts are unequal, causing
    # GroupShuffleSplit (which splits by cluster count) to produce badly skewed
    # train/val/test proportions.  Clustering unique locations and propagating
    # the split back to all rows fixes this.
    unique_locs = df.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
    n_unique = len(unique_locs)
    n_total = len(df)
    if n_unique < n_total:
        print(
            f"  Deduplicating: {n_unique} unique locations from {n_total} rows "
            f"(~{n_total / n_unique:.1f} rows/location)."
        )

    # Convert (lat, lon) degrees to radians for sklearn's haversine metric.
    # haversine returns arc length on the unit sphere, so eps must be in radians.
    # Error vs. true geodesic is < 0.1% at distances up to ~100 km.
    _EARTH_RADIUS_M = 6_371_000
    coords_rad = np.radians(np.array([unique_locs["lat"].values, unique_locs["lon"].values]).T)
    eps_rad = distance_m / _EARTH_RADIUS_M

    print(
        f"  Running DBSCAN (eps={distance_m / 1000:.1f} km, haversine, "
        f"n={n_unique} locations, n_jobs=-1)..."
    )
    t0 = time.time()
    clustering = DBSCAN(
        eps=eps_rad,
        metric="haversine",
        algorithm="ball_tree",
        min_samples=2,
        n_jobs=-1,
    ).fit(coords_rad)
    print(f"  DBSCAN done in {time.time() - t0:.1f}s.")

    # Noise points (label -1) each become their own unique cluster so that
    # GroupShuffleSplit can assign them individually to a split partition.
    clusters = copy.deepcopy(clustering.labels_)
    next_label = int(np.max(clusters)) + 1
    for i, label in enumerate(clusters):
        if label == -1:
            clusters[i] = next_label
            next_label += 1

    n_clusters = len(np.unique(clusters))
    n_noise = int(np.sum(clustering.labels_ == -1))
    print(f"  Clustering done: {n_clusters} location clusters ({n_noise} singleton noise points).")

    train_prop, val_prop, test_prop = train_val_test_split

    # Greedy size-aware cluster assignment.
    #
    # GroupShuffleSplit splits by cluster *count*, not by sample count.  When the
    # cluster size distribution is heavily skewed (a few mega-clusters + many
    # tiny 2-location clusters), this produces badly imbalanced splits.
    #
    # Instead: shuffle clusters for randomness, sort by size descending, then
    # assign each cluster to whichever split is furthest below its sample-count
    # target.  Each cluster goes to exactly one split, so there is no overlap.
    rng = np.random.default_rng(seed)
    unique_clusters, cluster_sizes = np.unique(clusters, return_counts=True)

    # Shuffle first so ties are broken randomly, then sort by descending size.
    shuffle_order = rng.permutation(len(unique_clusters))
    unique_clusters = unique_clusters[shuffle_order]
    cluster_sizes = cluster_sizes[shuffle_order]
    size_order = np.argsort(-cluster_sizes)
    unique_clusters = unique_clusters[size_order]
    cluster_sizes = cluster_sizes[size_order]

    target_train = n_unique * train_prop
    target_val = n_unique * val_prop
    target_test = n_unique * test_prop
    train_clusters, val_clusters, test_clusters = [], [], []
    count_train, count_val, count_test = 0, 0, 0

    for cluster_id, size in zip(unique_clusters, cluster_sizes):
        deficit_train = target_train - count_train
        deficit_val = target_val - count_val
        deficit_test = target_test - count_test
        if deficit_train >= deficit_val and deficit_train >= deficit_test:
            train_clusters.append(cluster_id)
            count_train += size
        elif deficit_val >= deficit_test:
            val_clusters.append(cluster_id)
            count_val += size
        else:
            test_clusters.append(cluster_id)
            count_test += size

    train_loc_mask = np.isin(clusters, train_clusters)
    val_loc_mask = np.isin(clusters, val_clusters)
    test_loc_mask = np.isin(clusters, test_clusters)

    # Sanity checks: every location assigned, no cluster in multiple splits.
    assert train_loc_mask.sum() + val_loc_mask.sum() + test_loc_mask.sum() == n_unique
    assert len(set(train_clusters) & set(val_clusters)) == 0
    assert len(set(train_clusters) & set(test_clusters)) == 0
    assert len(set(val_clusters) & set(test_clusters)) == 0

    print(
        f"  Split (locations): train={train_loc_mask.sum()}, "
        f"val={val_loc_mask.sum()}, test={test_loc_mask.sum()}"
    )

    # Propagate location-level split assignments back to all rows by (lat, lon).
    train_latlon = set(
        zip(unique_locs.loc[train_loc_mask, "lat"], unique_locs.loc[train_loc_mask, "lon"])
    )
    val_latlon = set(
        zip(unique_locs.loc[val_loc_mask, "lat"], unique_locs.loc[val_loc_mask, "lon"])
    )
    test_latlon = set(
        zip(unique_locs.loc[test_loc_mask, "lat"], unique_locs.loc[test_loc_mask, "lon"])
    )
    row_latlon = list(zip(df["lat"], df["lon"]))
    train_mask = np.array([ll in train_latlon for ll in row_latlon])
    val_mask = np.array([ll in val_latlon for ll in row_latlon])
    test_mask = np.array([ll in test_latlon for ll in row_latlon])

    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == n_total, (
        "Not all rows were assigned to a split — check for (lat, lon) values that "
        "don't match any unique location after deduplication."
    )

    name_locs = df["name_loc"].reset_index(drop=True)
    return {
        "train_indices": name_locs[train_mask].reset_index(drop=True),
        "val_indices": name_locs[val_mask].reset_index(drop=True),
        "test_indices": name_locs[test_mask].reset_index(drop=True),
        "clusters": clusters,
    }


def generate_splits(
    data_dir: str,
    distances_km: list[int] | None = None,
    seed: int = SEED,
) -> None:
    """Generate and save spatial-cluster split files for the requested distances.

    :param data_dir: root data directory (same as `paths.data_dir` in configs)
    :param distances_km: list of DBSCAN cluster distances in kilometres; None
        uses DEFAULT_DISTANCES_KM
    :param seed: random seed for GroupShuffleSplit
    """
    if distances_km is None:
        distances_km = DEFAULT_DISTANCES_KM

    dataset_dir = Path(data_dir) / DATASET_NAME
    csv_path = dataset_dir / MODEL_READY_CSV
    splits_dir = dataset_dir / "splits"

    if not csv_path.exists():
        raise FileNotFoundError(f"Model-ready CSV not found: {csv_path}")

    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for col in ("lat", "lon", "name_loc"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column")

    print(f"Loaded {len(df)} rows from {csv_path}")

    for dist_km in distances_km:
        dist_m = dist_km * 1000
        print(f"\nGenerating spatial split at {dist_km} km ({dist_m} m)...")

        split = make_spatial_split(df, distance_m=dist_m, seed=seed)
        n_train = len(split["train_indices"])
        n_val = len(split["val_indices"])
        n_test = len(split["test_indices"])

        out_name = f"split_spatial_{dist_km}km.pth"
        out_path = splits_dir / out_name
        torch.save(split, out_path)

        print(
            f"  Saved {out_name}  "
            f"(train={n_train}, val={n_val}, test={n_test}, "
            f"total={n_train + n_val + n_test}/{len(df)})"
        )
        log.info(
            f"  {dist_km}km: train={n_train}, val={n_val}, test={n_test} -> {out_name}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate spatial-cluster split files for yield_africa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Root data directory (same as paths.data_dir in configs). Default: data/",
    )
    parser.add_argument(
        "--distance_km",
        type=int,
        nargs="+",
        default=None,
        metavar="KM",
        help=(
            "Cluster distance threshold(s) in km. "
            f"Omit to generate the default set: {DEFAULT_DISTANCES_KM} km."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for GroupShuffleSplit. Default: {SEED}",
    )
    args = parser.parse_args()

    distances = args.distance_km  # None means use defaults
    print(
        f"Generating spatial splits  data_dir={args.data_dir}  "
        f"distances_km={distances or DEFAULT_DISTANCES_KM}  seed={args.seed}"
    )
    generate_splits(
        data_dir=args.data_dir,
        distances_km=distances,
        seed=args.seed,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
