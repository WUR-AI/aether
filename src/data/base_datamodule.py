import copy
import os
import time
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, random_split

from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.base_dataset import BaseDataset
from src.data.collate_fns import collate_fn
from src.data_preprocessing.data_utils import create_timestamp
from src.utils.errors import IllegalArgumentCombination


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 64,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        dataset_name: str = "base",
        split_mode: str = "random",
        save_split: bool = False,
        saved_split_file_name: str | None = None,
        caption_builder: BaseCaptionBuilder = None,
        seed: int = 12345,
        spatial_split_distance_m: int = 1000,
    ) -> None:
        """Datamodule class which handles dataset splits and batching.

        :param dataset: a use case and model configuration specific dataset
        :param batch_size: batch size for model training, validation and testing
        :param train_val_test_split: proportion of dataset to use for training, validation and
            testing
        :param num_workers: number of workers for dataloader
        :param pin_memory: pin memory for dataloader
        :param persistent_workers: keep DataLoader workers alive between epochs
        :param persistent_workers:
        :param dataset_name: dataset name
        :param split_mode: data split mode: random/from_file
        :param save_split: if to save split file
        :param saved_split_file_name: file name to save split file
        :param caption_builder: instance of BaseCaptionBuilder for generating textual captions
        :param spatial_split_distance_m: minimum distance in metres between clusters when
            split_mode is 'spatial_clusters'. Default 1000 m.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset: BaseDataset = dataset

        # Caption generation
        self.use_collate_fn: bool = self.dataset.use_aux_data
        if self.use_collate_fn:
            assert caption_builder is not None, "Caption_builder cannot be None"
            self.caption_builder = caption_builder
            self.caption_builder.sync_with_dataset(self.dataset)
            self.concept_configs = caption_builder.concepts

        self.split_data()

    @property
    def tabular_dim(self):
        return self.dataset.tabular_dim

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        return self.dataset.num_classes

    def setup(self, stage: str = "fit") -> None:
        """Obtaining batch size and data splits.

        Called by model trainer (trainer.fit()).
        """

        # Set up the dataset (download requested modalities)
        self.dataset.setup()

    @property
    def batch_size_per_device(self) -> None:
        """Divide batch size by the number of devices."""
        if self.trainer is None:
            return self.hparams.batch_size

        if self.hparams.batch_size % self.trainer.world_size != 0:
            raise RuntimeError(
                f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
            )

        return self.hparams.batch_size // self.trainer.world_size

    def split_data(self) -> None:
        """Split data into train, val and test.

        Either calculated here or loaded from file (random or dbscan clustered). Can be saved to
        file.
        """
        split_data_from_inds = True

        if self.hparams.split_mode == "random":
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(self.hparams.seed),
            )
            split_data_from_inds = False  # already split data
            print(
                f"Dataset was randomly split with proportions: {self.hparams.train_val_test_split}"
            )
            if self.hparams.save_split:
                split_indices = {
                    "train_indices": self.data_train.dataset.df.name_loc,
                    "val_indices": self.data_val.dataset.df.name_loc,
                    "test_indices": self.data_test.dataset.df.name_loc,
                }

        elif self.hparams.split_mode == "from_df":
            assert hasattr(
                self.dataset.df, "split"
            ), "Dataset dataframe must have a 'split' column for 'from_df' split mode."
            train_indices = self.dataset.df[self.dataset.df.split == "train"].index
            val_indices = self.dataset.df[self.dataset.df.split == "val"].index
            test_indices = self.dataset.df[self.dataset.df.split == "test"].index

        elif self.hparams.split_mode == "spatial_clusters":
            min_dist = self.hparams.spatial_split_distance_m
            coords = np.array([self.dataset.df.lat, self.dataset.df.lon]).T
            n = len(coords)
            print(
                f"Splitting {n} samples into spatial clusters "
                f"(eps={min_dist / 1000:.1f} km, haversine, n_jobs=-1)..."
            )
            # Convert (lat, lon) degrees to radians for sklearn's haversine metric.
            # haversine returns arc length on the unit sphere, so eps must be in radians.
            _EARTH_RADIUS_M = 6_371_000
            coords_rad = np.radians(coords)
            eps_rad = min_dist / _EARTH_RADIUS_M
            t0 = time.time()
            clustering = DBSCAN(
                eps=eps_rad,
                metric="haversine",
                algorithm="ball_tree",
                min_samples=2,
                n_jobs=-1,
            ).fit(coords_rad)
            print(f"DBSCAN done in {time.time() - t0:.1f}s. Creating splits...")
            # Non-clustered points are labeled -1. Change to new cluster label.
            clusters = copy.deepcopy(clustering.labels_)
            new_cl = np.max(clusters) + 1
            for i, cl in enumerate(clusters):
                if cl == -1:
                    clusters[i] = new_cl
                    new_cl += 1

            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.hparams.train_val_test_split[2],
                random_state=self.hparams.seed,
            )
            train_val_indices, test_indices = next(
                gss.split(np.arange(len(coords)), groups=clusters)
            )
            gss_2 = GroupShuffleSplit(
                n_splits=1,
                test_size=(
                    self.hparams.train_val_test_split[1]
                    / (self.hparams.train_val_test_split[0] + self.hparams.train_val_test_split[1])
                ),
                random_state=self.hparams.seed,
            )
            tmp_train_indices, tmp_val_indices = next(
                gss_2.split(train_val_indices, groups=clusters[train_val_indices])
            )
            train_indices = train_val_indices[tmp_train_indices]
            val_indices = train_val_indices[tmp_val_indices]
            clusters_train = clusters[train_indices]
            clusters_val = clusters[val_indices]
            clusters_test = clusters[test_indices]
            # assert no overlap in indices:
            assert len(np.intersect1d(train_indices, val_indices)) == 0, np.intersect1d(
                train_indices, val_indices
            )
            assert len(np.intersect1d(train_indices, test_indices)) == 0, np.intersect1d(
                train_indices, test_indices
            )
            assert len(np.intersect1d(val_indices, test_indices)) == 0, np.intersect1d(
                val_indices, test_indices
            )

            # assert no overlap in clusters:
            assert len(np.intersect1d(clusters_train, clusters_val)) == 0, np.intersect1d(
                clusters_train, clusters_val
            )
            assert len(np.intersect1d(clusters_train, clusters_test)) == 0, np.intersect1d(
                clusters_train, clusters_test
            )
            assert len(np.intersect1d(clusters_val, clusters_test)) == 0, np.intersect1d(
                clusters_val, clusters_test
            )

            print(
                f"Created {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test indices using DBSCAN spatial clustering with {min_dist} m minimum distance between clusters."
            )
            if self.hparams.save_split:
                split_indices = {
                    "train_indices": self.dataset.df.name_loc[train_indices],
                    "val_indices": self.dataset.df.name_loc[val_indices],
                    "test_indices": self.dataset.df.name_loc[test_indices],
                    "clusters": clusters,
                }

        elif self.hparams.split_mode == "from_file":
            if self.hparams.saved_split_file_name is None:
                raise IllegalArgumentCombination(
                    "saved_split_file_name must be provided when split_mode is 'from_file'"
                )

            self.hparams.save_split = False  # don't save split when loading from file

            # get indices from file
            self.saved_split_file_path = os.path.join(
                self.hparams.dataset.data_dir, "splits", self.hparams.saved_split_file_name
            )
            split_indices = self.load_split_indices(self.saved_split_file_path)
            train_indices = split_indices["train_indices"]
            val_indices = split_indices["val_indices"]
            test_indices = split_indices.get("test_indices", None)

            if not isinstance(train_indices, pd.Series):
                raise NotImplementedError("Expected a pd series of name_locs for data splits.")
            if not isinstance(val_indices, pd.Series):
                raise NotImplementedError("Expected a pd series of name_locs for data splits.")
            if test_indices is not None and not isinstance(test_indices, pd.Series):
                raise NotImplementedError("Expected a pd series of name_locs for data splits.")

            train_indices = np.where(self.dataset.df["name_loc"].isin(train_indices))[0]
            val_indices = np.where(self.dataset.df["name_loc"].isin(val_indices))[0]
            if test_indices is not None:
                test_indices = np.where(self.dataset.df["name_loc"].isin(test_indices))[0]

            print(f"Dataset was split using indices from file: {self.saved_split_file_path}")
        else:
            raise NotImplementedError(
                f"{self.hparams.train_val_test_split} split mode not implemented."
            )

        if split_data_from_inds:
            self.data_train = torch.utils.data.Subset(self.dataset, train_indices)
            self.data_train.dataset.mode = "train"
            self.data_val = torch.utils.data.Subset(self.dataset, val_indices)
            self.data_val.dataset.mode = "val"

            if test_indices is not None:
                self.data_test = torch.utils.data.Subset(self.dataset, test_indices)
                self.data_test.dataset.mode = "test"
            else:
                self.data_test = None

        if self.hparams.save_split:
            self.save_split_indices(split_indices)

        self._compute_tabular_normalisation_stats()
        self._compute_target_normalisation_stats()

    def _compute_tabular_normalisation_stats(self) -> None:
        """Compute per-feature mean and std on the training split for use by TabularEncoder.

        Statistics are stored as ``self.tabular_normalisation_stats = (mean, std)`` —
        both float32 tensors of shape ``(tabular_dim,)`` — or ``None`` if the dataset
        has no tabular features.  The std is clamped to 1 for constant features so that
        division is always safe.
        """
        self.tabular_normalisation_stats = None

        if not getattr(self.dataset, "use_features", False):
            return
        feat_names = getattr(self.dataset, "feat_names", None)
        if not feat_names:
            return

        train_indices = self.data_train.indices
        train_df = self.dataset.df.iloc[train_indices][feat_names]

        mean = train_df.mean(axis=0).values
        std = train_df.std(axis=0).values
        std = np.where(std == 0, 1.0, std)  # avoid division by zero for constant features

        self.tabular_normalisation_stats = (
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32),
        )

    def _compute_target_normalisation_stats(self) -> None:
        """Compute per-target mean and std on the training split.

        Statistics are stored as ``self.target_normalisation_stats = (mean, std)`` —
        both float32 tensors of shape ``(num_targets,)`` — or ``None`` if the dataset
        has no targets.  The std is clamped to 1 for constant targets.
        """
        self.target_normalisation_stats = None

        if not getattr(self.dataset, "use_target_data", False):
            return
        target_names = getattr(self.dataset, "target_names", None)
        if not target_names:
            return

        train_indices = self.data_train.indices
        train_df = self.dataset.df.iloc[train_indices][target_names]

        mean = train_df.mean(axis=0).values
        std = train_df.std(axis=0).values
        std = np.where(std == 0, 1.0, std)  # avoid division by zero for constant targets

        self.target_normalisation_stats = (
            torch.tensor(mean, dtype=torch.float32),
            torch.tensor(std, dtype=torch.float32),
        )

    def save_split_indices(self, split_indices: dict[str, Any] | dict):
        """Save split indices into file."""
        self.split_dir = os.path.join(self.hparams.dataset.data_dir, "splits")
        os.makedirs(self.split_dir, exist_ok=True)

        timestamp = create_timestamp()
        torch.save(
            split_indices,
            os.path.join(
                self.split_dir,
                f"split_indices_{self.hparams.dataset_name}_{timestamp}.pth",
            ),
        )
        print(f"Saved split indices to split_indices_{timestamp}.pth")

    def load_split_indices(self, filepath: str = None) -> dict:
        """Load split indices from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError("Split indices file does not exist: {filepath}")

        split_indices = torch.load(filepath, weights_only=False)
        assert (
            "train_indices" in split_indices and "val_indices" in split_indices
        ), "Split indices file must contain 'train_indices' and 'val_indices'"

        # TODO: is this ever used?
        n_in_splits = len(split_indices["train_indices"]) + len(split_indices["val_indices"])
        if "test_indices" in split_indices:
            n_in_splits += len(split_indices["test_indices"])

        return split_indices

    def setup_conceptcaption_validation_parameters(
        self,
        use_saved_threshold_if_available=True,
        overwrite_existing_thresholds=False,
        save_newly_computed_threshold=True,
        compute_train_threshold=True,
        verbose=1,
    ):
        """Set up the contrastive retrieval evaluation by computing dynamic k baselines and
        initializing the validation object.

        Parameters:
        - use_saved_threshold_if_available: whether to use saved thresholds if available
        - overwrite_existing_thresholds: whether to overwrite existing thresholds with newly computed ones
        - save_newly_computed_threshold: whether to save newly computed thresholds to a new concept configs file
        - compute_train_threshold: whether to compute thresholds only using the train set
        - verbose: whether to print the dynamic ks and their baselines
        - verbose: whether to print the dynamic ks and their baselines
        """
        if not (
            hasattr(self, "data_train") or hasattr(self, "data_val") or hasattr(self, "data_test")
        ):
            raise RuntimeError(
                "Data splits not found. Make sure to call this method after split_data() which creates the data splits."
            )

        assert (
            compute_train_threshold
        ), "Currently the only implementation of computing thresholds is by computing them on the train set only, and then using these for the validation and test set too."
        if overwrite_existing_thresholds:
            use_saved_threshold_if_available = False
            save_newly_computed_threshold = True  # for reproducibility.

        self.concepts = [c["concept_caption"] for c in self.concept_configs]
        self.concept_names = [
            f"{c['col'].replace('aux_', '')}_{'max' if c.get('is_max') else 'min'}"
            for c in self.concept_configs
        ]
        list_concept_ids_drop = []
        new_thresholds_computed = False
        self.dynamic_k_baselines = {}
        for dataset_name in [
            "train",
            "val",
            "test",
        ]:  # ensure 'train' is first for use_train_threshold logic!
            if not hasattr(self, f"data_{dataset_name}"):
                continue

            tmp_ds = getattr(self, f"data_{dataset_name}")
            n_ds = len(tmp_ds)
            self.dynamic_k_baselines[dataset_name] = {}

            if use_saved_threshold_if_available and all(
                (
                    c.get("theta_k") is not None
                    and c.get(f"accuracy_baseline_{dataset_name}") is not None
                    and c.get("is_max") is not None
                )
                for c in self.concept_configs
            ):  # no need to compute if all concepts have saved thresholds and baselines
                for i_c, c in enumerate(self.concept_configs):
                    c_name = self.concept_names[i_c]
                    theta_k = c["theta_k"]
                    self.dynamic_k_baselines[dataset_name][c_name] = self.concept_configs[i_c][
                        f"accuracy_baseline_{dataset_name}"
                    ]

                    if verbose:
                        print(
                            f"Concept '{self.concept_names[i_c]}' in {dataset_name} set: is_max={c['is_max']}, saved theta_k={theta_k:.6f}, saved baseline={self.dynamic_k_baselines[dataset_name][c_name]}%)"
                        )

            else:
                print(
                    f"WARNING: No saved thresholds and baselines found for some or all concepts for {dataset_name}, computing new ones. This may take a while..."
                )
                print(
                    "To speed up this computation, make sure to run this method with a dataloader that has only the coordinates and aux data (no other EO data)."
                )
                new_thresholds_computed = True
                if save_newly_computed_threshold:
                    print(
                        "The threshold values will be written to a new concept configs file if they are computed anew."
                    )
                else:
                    print(
                        "Consider setting save_newly_computed_threshold=True to store the computed thresholds and avoid recomputation in the future."
                    )
                # Iterate through dataset once to get aux values for all concepts (to avoid multiple iterations if multiple concepts). Best done with coords only dataset for speed!
                aux_vals_per_concept = {i: [] for i in range(len(self.concept_configs))}
                for item in tmp_ds:
                    aux_data = item["aux"]["aux"]
                    for i_c, c in enumerate(self.concept_configs):
                        aux_col_id = c["id"]
                        aux_vals_per_concept[i_c].append(aux_data[aux_col_id])

                # Compute per concept
                for i_c, c in enumerate(self.concept_configs):
                    c_name = self.concept_names[i_c]
                    aux_vals_current_ds = aux_vals_per_concept[i_c]

                    if dataset_name == "train":
                        if overwrite_existing_thresholds:
                            theta_k = self.find_elbow_point(
                                aux_vals_current_ds
                            )  # only compute if not present
                        else:
                            theta_k = c.get("theta_k") or self.find_elbow_point(
                                aux_vals_current_ds
                            )  # only compute if not present
                        self.concept_configs[i_c]["theta_k"] = float(
                            theta_k
                        )  # assign new theta_k to concept_configs for later use in validation
                    else:
                        theta_k = self.concept_configs[i_c]["theta_k"]

                    n_baseline_max = sum(aux_val >= theta_k for aux_val in aux_vals_current_ds)
                    n_baseline_min = sum(aux_val <= theta_k for aux_val in aux_vals_current_ds)

                    if n_baseline_max < n_baseline_min:
                        if not c.get("is_max", True):
                            print(
                                f"Concept {c_name} has n_baseline_max < n_baseline_min but is_max is False. Therefore it will NOT be used/stored. Please check the concept configs or the computed theta_k for this concept."
                            )
                            if i_c not in list_concept_ids_drop:
                                list_concept_ids_drop.append(i_c)
                        n_baseline = n_baseline_max
                        _is_max = True
                    else:
                        if c.get("is_max", False):
                            print(
                                f"Concept {c_name} has n_baseline_max >= n_baseline_min but is_max is True. Therefore it will NOT be used/stored. Please check the concept configs or the computed theta_k for this concept."
                            )
                            if i_c not in list_concept_ids_drop:
                                list_concept_ids_drop.append(i_c)
                        n_baseline = n_baseline_min
                        _is_max = False
                    if "is_max" not in c:
                        print(
                            f"Concept {c_name} does not have 'is_max' specified. Setting is_max to {_is_max} based on whether n_baseline_max ({n_baseline_max}) is smaller than n_baseline_min ({n_baseline_min})."
                        )
                        self.concept_configs[i_c]["is_max"] = _is_max
                        self.concept_names[i_c] = (
                            f"{c['col'].replace('aux_', '')}_{'max' if _is_max else 'min'}"
                        )

                    if n_baseline == n_ds:
                        n_baseline = (
                            n_ds - 1
                        )  # to avoid having a baseline of 100% (will still yield index score of 1)
                        if dataset_name == "train":
                            theta_k = (
                                min(aux_vals_current_ds) + 1e-6
                                if c["is_max"]
                                else max(aux_vals_current_ds) - 1e-6
                            )

                    if verbose:
                        print(
                            f"Concept '{self.concept_names[i_c]}' in {dataset_name} set: is_max={c['is_max']}, original theta_k={self.concept_configs[i_c]['theta_k']:.6f}, new theta_k={theta_k:.6f}, baseline={n_baseline}/{n_ds} ({n_baseline / n_ds * 100:.1f}%)"
                        )
                    self.dynamic_k_baselines[dataset_name][c_name] = n_baseline / n_ds * 100
                    self.concept_configs[i_c][f"accuracy_baseline_{dataset_name}"] = float(
                        self.dynamic_k_baselines[dataset_name][c_name]
                    )

                if len(list_concept_ids_drop) > 0 and dataset_name == "test":
                    print(
                        f"Dropping concepts with ids {list_concept_ids_drop} and names {[self.concept_names[i] for i in list_concept_ids_drop]} from evaluation due to mismatch between is_max and whether n_baseline_max or n_baseline_min is smaller."
                    )
                    self.concept_configs = [
                        c
                        for i, c in enumerate(self.concept_configs)
                        if i not in list_concept_ids_drop
                    ]
                    self.concepts = [c["concept_caption"] for c in self.concept_configs]
                    self.concept_names = [
                        f"{c['col'].replace('aux_', '')}_{'max' if c['is_max'] else 'min'}"
                        for c in self.concept_configs
                    ]
                    self.dynamic_k_baselines[dataset_name] = {
                        c_name: baseline
                        for i, (c_name, baseline) in enumerate(
                            self.dynamic_k_baselines[dataset_name].items()
                        )
                        if i not in list_concept_ids_drop
                    }

        if (
            save_newly_computed_threshold and new_thresholds_computed
        ):  # only save after computing on train set, and only if we are computing new thresholds (not just using saved ones), to avoid overwriting with the same values or with values computed on val/test set
            self.caption_builder.store_concept_thresholds(self.concept_configs, update_self=True)
        elif new_thresholds_computed:
            self.caption_builder.update_concept_thresholds(self.concept_configs)

        return None

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            persistent_workers=(
                bool(self.hparams.persistent_workers) if self.hparams.num_workers > 0 else False
            ),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=(
                self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None
            ),
            shuffle=True,
            collate_fn=(
                partial(
                    collate_fn,
                    mode="train",
                    caption_builder=self.caption_builder,
                )
                if self.use_collate_fn
                else None
            ),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                bool(self.hparams.persistent_workers) if self.hparams.num_workers > 0 else False
            ),
            prefetch_factor=(
                self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None
            ),
            shuffle=False,
            collate_fn=(
                partial(
                    collate_fn,
                    mode="val",
                    caption_builder=self.caption_builder,
                )
                if self.use_collate_fn
                else None
            ),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=(
                bool(self.hparams.persistent_workers) if self.hparams.num_workers > 0 else False
            ),
            prefetch_factor=(
                self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None
            ),
            shuffle=False,
            collate_fn=(
                partial(
                    collate_fn,
                    mode="test",
                    caption_builder=self.caption_builder,
                )
                if self.use_collate_fn
                else None
            ),
        )

    @staticmethod
    def find_elbow_point(vals):
        """Vals is a list of tensor values."""
        with torch.no_grad():
            vals = torch.tensor(vals).cpu().numpy()
            vals = vals[~np.isnan(vals)]  # remove NaN values

            vals = np.sort(vals)
            vals = vals[vals > vals[0]]
            x = np.arange(len(vals)) / len(vals)
            y = vals
            if x[0] == x[-1]:  # all values are the same
                print(
                    "All values are the same, returning the value itself as elbow point.", vals[0]
                )
                return vals[0]
            slope = (y[-1] - y[0]) / (x[-1] - x[0])  # diagonal from first to last point
            intercept = y[0] - slope * x[0]
            orthogonal_slope = -1 / slope

            intercepts_orthogonal = y - orthogonal_slope * x
            intersection_diagonal_orthogonal = (intercepts_orthogonal - intercept) / (
                slope - orthogonal_slope
            )
            distances = np.sqrt(
                (x - intersection_diagonal_orthogonal) ** 2 + (y - (slope * x + intercept)) ** 2
            )  # distance to diagonal
            elbow_index = np.argmax(distances)
            elbow_point = y[elbow_index]
            return elbow_point


if __name__ == "__main__":
    _ = BaseDataModule(None, None, None, None, None, None, None, None, None, None, None)
