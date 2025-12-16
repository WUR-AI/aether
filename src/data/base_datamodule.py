from functools import partial
from typing import Tuple, Any, Dict, List
import os
import torch
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
import pandas as pd
import numpy as np
from src.data.base_dataset import BaseDataset
from src.models.components.collate_fns import collate_fn
from src.utils.errors import IllegalArgumentCombination


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            dataset: BaseDataset,
            batch_size: int = 64,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            num_workers: int = 0,
            pin_memory: bool = False,
            split_mode: str = 'random',
            save_split: bool = False,
            filepath_split_indices_load: str | None = None,
            filepath_split_indices_save: str | None = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset: BaseDataset = dataset
        self.batch_size_per_device: int = batch_size
        self.use_collate_fn: bool = True if self.dataset.use_aux_data else False

        self.setup()

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    def setup_batch_size_per_device(self) -> None:
        """Divide batch size by the number of devices."""
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def setup(self, stage: str = 'fit') -> None:
        self.setup_batch_size_per_device()

        if self.hparams.split_mode == "random":
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            print(f'Dataset was randomly split with proportions: {self.hparams.train_val_test_split}')
        elif self.hparams.split_mode == "from_file":
            assert self.hparams.filepath_split_indices is not None, IllegalArgumentCombination(f"filepath_split_indices must be provided when split_mode is 'from_file'")
            self.hparams.save_split = False  ## don't save split when loading from file
            split_indices = self.load_split_indices(self.hparams.filepath_split_indices_load)
            train_indices = split_indices['train_indices']
            val_indices = split_indices['val_indices']
            test_indices = split_indices.get('test_indices', None)
            
            if type(train_indices) != pd.Series: raise NotImplementedError('Expected a pd series of ids for data splits.')
            if type(val_indices) != pd.Series: raise NotImplementedError('Expected a pd series of ids for data splits.')
            if test_indices is not None and type(test_indices) != pd.Series: raise NotImplementedError('Expected a pd series of ids for data splits.')

            train_indices = np.where(self.dataset.df['id'].isin(train_indices))[0]
            self.data_train = torch.utils.data.Subset(self.dataset, train_indices)
            self.data_train.dataset.mode = 'train'
            val_indices = np.where(self.dataset.df['id'].isin(val_indices))[0]
            self.data_val = torch.utils.data.Subset(self.dataset, val_indices)
            self.data_val.dataset.mode = 'val'

            if test_indices is not None:
                test_indices = np.where(self.dataset.df['id'].isin(test_indices))[0]
                self.data_test = torch.utils.data.Subset(self.dataset, test_indices)
                self.data_test.dataset.mode = 'test'
            else:
                self.data_test = None
            print(f'Dataset was split using indices from file: {self.hparams.filepath_split_indices_load}')
        else:
            raise NotImplementedError(f'{self.hparams.train_val_test_split} split mode not implemented.')

        if self.hparams.save_split:
            assert self.hparams.filepath_split_indices_save is not None, "filepath_split_indices_save must be provided when saving a new data split."
            assert os.path.exists(os.path.dirname(self.hparams.filepath_split_indices_save)), f"Directory to save split indices does not exist: {os.path.dirname(self.hparams.filepath_split_indices_save)}"
            split_indices = {
                'train_indices': self.data_train.get('id'),
                'val_indices': self.data_val.get('id'),
                'test_indices': self.data_test.get('id')
            }
            timestamp = du.create_timestamp()
            torch.save(split_indices, os.path.join(self.hparams.filepath_split_indices_save, f'split_indices_{self.dataset_name}_{timestamp}.pth'))
            print(f'Saved split indices to split_indices_{timestamp}.pth')

    def load_split_indices(self, filepath: str = None) -> dict:
        """Load split indices from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError('Split indices file does not exist: {filepath}')

        split_indices = torch.load(filepath, weights_only=False)
        assert 'train_indices' in split_indices and 'val_indices' in split_indices, "Split indices file must contain 'train_indices' and 'val_indices'"

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
            collate_fn=partial(collate_fn, mode='train') if self.use_collate_fn else None,
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
            collate_fn=partial(collate_fn, mode='val') if self.use_collate_fn else None,
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
            collate_fn=partial(collate_fn, mode='test') if self.use_collate_fn else None,
        )

if __name__ == "__main__":
    _ = BaseDataModule(None, None, None, None, None, None, None)
