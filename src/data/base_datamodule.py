from functools import partial
from typing import Tuple, Any, Dict, List
import os, copy
import torch
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from geopy.distance import distance as geodist  # avoid naming confusion

from src.data_preprocessing.data_utils import create_timestamp
from src.data.base_dataset import BaseDataset
from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.collate_fns import collate_fn
from src.utils.errors import IllegalArgumentCombination


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 64,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        num_workers: int = 0,
        pin_memory: bool = False,
        dataset_name: str = 'base',
        split_mode: str = 'random',
        save_split: bool = False,
        split_dir: str = None,
        saved_split_file_name: str | None = None,
        caption_builder: BaseCaptionBuilder = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset: BaseDataset = dataset
        self.batch_size_per_device: int = batch_size
        self.use_collate_fn: bool = True if self.dataset.use_aux_data else False
        if self.use_collate_fn:
            self.caption_builder = caption_builder
            self.caption_builder.sync_with_dataset(self.dataset)

        self.setup()

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    def setup(self, stage: str = 'fit') -> None:
        self.setup_batch_size_per_device()
        self.split_data()

    def setup_batch_size_per_device(self) -> None:
        """Divide batch size by the number of devices."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def split_data(self) -> None:
        '''Split data into train, val and test. Either calculated here or loaded from file (random or dbscan clustered).
        Can be saved to file.'''
        split_data_from_inds = True # TODO: what is this for?

        if self.hparams.split_mode == "random":
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            split_data_from_inds = False  # already split data
            print(f'Dataset was randomly split with proportions: {self.hparams.train_val_test_split}')
            if self.hparams.save_split:
                split_indices = {
                    'train_indices': self.data_train.dataset.df.id,
                    'val_indices': self.data_val.dataset.df.id,
                    'test_indices': self.data_test.dataset.df.id,
                }

        elif self.hparams.split_mode == "spatial_clusters":
            print('Splitting dataset using spatial clusters. This can take a while...')
            coords = np.array([self.dataset.df.lat, self.dataset.df.lon]).T
            if len(coords) > 2000:
                print('Warning: DBSCAN clustering on more than 2000 samples may be slow. Maybe set n_jobs in DBScan?')
            ## 4000 m distance between points. Use geodist to calculate true distance.
            min_dist = 4000
            clustering = DBSCAN(eps=min_dist, metric=lambda u, v: geodist(u, v).meters, min_samples=2).fit(coords)
            print('Clustering done. Creating splits and saving.')
            ## Non-clustered points are labeled -1. Change to new cluster label.
            clusters = copy.deepcopy(clustering.labels_)
            new_cl = np.max(clusters) + 1
            for i, cl in enumerate(clusters):
                if cl == -1:
                    clusters[i] = new_cl
                    new_cl += 1

            gss = GroupShuffleSplit(n_splits=1, test_size=self.hparams.train_val_test_split[2], random_state=0)
            train_val_indices, test_indices = next(gss.split(np.arange(len(coords)), groups=clusters))
            gss_2 = GroupShuffleSplit(
                n_splits=1,
                test_size=(
                    self.hparams.train_val_test_split[1]
                    / (self.hparams.train_val_test_split[0] + self.hparams.train_val_test_split[1])
                ),
                random_state=0,
            )
            tmp_train_indices, tmp_val_indices = next(
                gss_2.split(train_val_indices, groups=clusters[train_val_indices])
            )
            train_indices = train_val_indices[tmp_train_indices]
            val_indices = train_val_indices[tmp_val_indices]
            clusters_train = clusters[train_indices]
            clusters_val = clusters[val_indices]
            clusters_test = clusters[test_indices]
            ## assert no overlap in indices:
            assert len(np.intersect1d(train_indices, val_indices)) == 0, np.intersect1d(train_indices, val_indices)
            assert len(np.intersect1d(train_indices, test_indices)) == 0, np.intersect1d(train_indices, test_indices)
            assert len(np.intersect1d(val_indices, test_indices)) == 0, np.intersect1d(val_indices, test_indices)

            ## assert no overlap in clusters:
            assert len(np.intersect1d(clusters_train, clusters_val)) == 0, np.intersect1d(clusters_train, clusters_val)
            assert len(np.intersect1d(clusters_train, clusters_test)) == 0, np.intersect1d(
                clusters_train, clusters_test
            )
            assert len(np.intersect1d(clusters_val, clusters_test)) == 0, np.intersect1d(clusters_val, clusters_test)

            print(
                f'Created {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test indices using DBSCAN spatial clustering with {min_dist} m minimum distance between clusters.'
            )
            if self.hparams.save_split:
                split_indices = {
                    'train_indices': self.dataset.df.id[train_indices],
                    'val_indices': self.dataset.df.id[val_indices],
                    'test_indices': self.dataset.df.id[test_indices],
                    'clusters': clusters,
                }

        elif self.hparams.split_mode == "from_file":
            if self.hparams.saved_split_file_name is None:
                raise IllegalArgumentCombination(f"saved_split_file_name must be provided when split_mode is 'from_file'")

            self.hparams.save_split = False  ## don't save split when loading from file

            # get indices from file
            self.saved_split_file_path = os.path.join(self.hparams.split_dir, self.hparams.saved_split_file_name)
            split_indices = self.load_split_indices(self.saved_split_file_path)
            train_indices = split_indices['train_indices']
            val_indices = split_indices['val_indices']
            test_indices = split_indices.get('test_indices', None)

            if type(train_indices) != pd.Series:
                raise NotImplementedError('Expected a pd series of ids for data splits.')
            if type(val_indices) != pd.Series:
                raise NotImplementedError('Expected a pd series of ids for data splits.')
            if test_indices is not None and type(test_indices) != pd.Series:
                raise NotImplementedError('Expected a pd series of ids for data splits.')

            train_indices = np.where(self.dataset.df['id'].isin(train_indices))[0]
            val_indices = np.where(self.dataset.df['id'].isin(val_indices))[0]
            if test_indices is not None:
                test_indices = np.where(self.dataset.df['id'].isin(test_indices))[0]

            print(f'Dataset was split using indices from file: {self.saved_split_file_path}')
        else:
            raise NotImplementedError(f'{self.hparams.train_val_test_split} split mode not implemented.')

        if split_data_from_inds:
            self.data_train = torch.utils.data.Subset(self.dataset, train_indices)
            self.data_train.dataset.mode = 'train'
            self.data_val = torch.utils.data.Subset(self.dataset, val_indices)
            self.data_val.dataset.mode = 'val'

            if test_indices is not None:
                self.data_test = torch.utils.data.Subset(self.dataset, test_indices)
                self.data_test.dataset.mode = 'test'
            else:
                self.data_test = None

        if self.hparams.save_split:
            self.save_split_indices(split_indices)

    def save_split_indices(self, split_indices: dict[str, Any] | dict):
        assert (self.hparams.split_dir is not None), "split_dir must be provided when saving a new data split."
        assert os.path.exists(self.hparams.split_dir), f"Directory to save split indices does not exist: {self.hparams.split_dir}"
        assert type(split_indices) == dict, "split_indices must be a dictionary to be saved."

        timestamp = create_timestamp()
        torch.save(
            split_indices,
            os.path.join(
                self.hparams.split_dir,
                f'split_indices_{self.hparams.dataset_name}_{timestamp}.pth',
            ),
        )
        print(f'Saved split indices to split_indices_{timestamp}.pth')

    def load_split_indices(self, filepath: str = None) -> dict:
        """Load split indices from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError('Split indices file does not exist: {filepath}')

        split_indices = torch.load(filepath, weights_only=False)
        assert (
            'train_indices' in split_indices and 'val_indices' in split_indices
        ), "Split indices file must contain 'train_indices' and 'val_indices'"

        # TODO: is this ever used?
        n_in_splits = len(split_indices['train_indices']) + len(split_indices['val_indices'])
        if 'test_indices' in split_indices:
            n_in_splits += len(split_indices['test_indices'])

        return split_indices

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=(
                partial(collate_fn, mode='train', caption_builder=self.caption_builder) if self.use_collate_fn else None
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
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            shuffle=False,
            collate_fn=(
                partial(collate_fn, mode='val', caption_builder=self.caption_builder) if self.use_collate_fn else None
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
            persistent_workers=True if self.hparams.num_workers > 0 else False,
            shuffle=False,
            collate_fn=(
                partial(collate_fn, mode='test', caption_builder=self.caption_builder) if self.use_collate_fn else None
            ),
        )


if __name__ == "__main__":
    _ = BaseDataModule(None, None, None, None, None, None, None, None, None, None, None)
