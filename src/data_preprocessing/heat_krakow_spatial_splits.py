import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def generate_spatial_stratified_split(
    csv_path: str,
    output_pth_path: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 12345,
):
    """Generate and save a spatially and target-stratified train/val/test split.

    Executes a two-stage stratification:
    1. Clusters coordinates via KMeans into 8 geographical spatial zones.
    2. Quantizes the 'target_lst' variable into 8 uniform distribution buckets.
    3. Merges these into a composite key, cleans singleton strata, and splits the location identifiers into train/val/test subsets.

    The final splits are stored as a mapped PyTorch serialized pandas Series.

    Parameters
    ----------
    csv_path : str
    Path to the source CSV containing 'lat', 'lon', 'target_lst', and 'name_loc'.
        output_pth_path : str
    Destination file path to save the serialized PyTorch (.pth) object.
        train_ratio : float, default 0.70
    Proportion of the dataset allocated to the training set.
        val_ratio : float, default 0.15
    Proportion of the dataset allocated to the validation set.
        test_ratio : float, default 0.15
    Proportion of the dataset allocated to the test set.
        seed : int, default 12345
    Random seed for KMeans initialization and stratified shuffling stability.

    Returns
    -------
    None
    Saves a pd.Series to `output_pth_path` mapping split tags to location IDs.
    """
    # Load  Kraków CSV
    print(f"Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Spatial Binning: Group coordinates into 8 geographic clusters
    print("Clustering spatial locations...")
    coords = df[["lat", "lon"]].values
    kmeans = KMeans(n_clusters=8, random_state=seed, n_init="auto")
    df["spatial_zone"] = kmeans.fit_predict(coords)

    # Target Binning: Segment LST temperatures into 8 quantile buckets
    print("Binning target variable distributions...")
    df["target_bucket"] = pd.qcut(df["target_lst"], q=8, labels=False, duplicates="drop")

    # Combine into a joint Strata Key
    df["composite_strata"] = df["spatial_zone"].astype(str) + "_" + df["target_bucket"].astype(str)

    # Clean single-sample strata to prevent train_test_split from crashing
    strata_counts = df["composite_strata"].value_counts()
    singletons = strata_counts[strata_counts < 2].index
    if len(singletons) > 0:
        # Merge singletons into the largest stratum as a safety fallback
        largest_stratum = strata_counts.idxmax()
        df["composite_strata"] = df["composite_strata"].replace(singletons, largest_stratum)

    # Execute First Split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_idx, temp_idx = train_test_split(
        df.index.tolist(), test_size=temp_ratio, stratify=df["composite_strata"], random_state=seed
    )

    # Execute Second Split: Split Temp into Validation and Test
    relative_test_ratio = test_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=df.loc[temp_idx, "composite_strata"],
        random_state=seed,
    )

    # Map indices to unique location identifiers (name_loc)
    train_locs = df.loc[train_idx, "name_loc"].tolist()
    val_locs = df.loc[val_idx, "name_loc"].tolist()
    test_locs = df.loc[test_idx, "name_loc"].tolist()

    # Build the Pandas Series
    print("Converting dictionary structures to an indexed pd.Series...")
    index_labels = (
        ["train_indices"] * len(train_locs)
        + ["val_indices"] * len(val_locs)
        + ["test_indices"] * len(test_locs)
    )
    name_loc_values = train_locs + val_locs + test_locs
    split_series = pd.Series(data=name_loc_values, index=index_labels)

    # Save to a .pth file
    os.makedirs(os.path.dirname(output_pth_path), exist_ok=True)
    torch.save(split_series, output_pth_path)

    # Print Diagnostics to verify distribution percentages
    print("\n--- Split Generation Complete ---")
    print(f"Train samples : {len(train_locs)} ({len(train_locs)/len(df):.1%})")
    print(f"Val samples   : {len(val_locs)} ({len(val_locs)/len(df):.1%})")
    print(f"Test samples  : {len(test_locs)} ({len(test_locs)/len(df):.1%})")
    print(f"Saved split object type: {type(split_series)}")
    print(f"File successfully written to -> {output_pth_path}")


# --- Execution Example ---
if __name__ == "__main__":
    CSV_INPUT = "/aether/data/heat_krakow/model_ready_heat_krakow.csv"
    PTH_OUTPUT = "/aether/data/heat_krakow/splits.pth"

    generate_spatial_stratified_split(CSV_INPUT, PTH_OUTPUT)
