from typing import Tuple, Any

import torch
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule

from src.data.base_dataset import BaseDataset

class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            dataset: BaseDataset,
            batch_size: int = 64,
            train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            num_workers: int = 0,
            pin_memory: bool = False,
            split_mode: str = 'random'
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset: BaseDataset = dataset
        self.batch_size_per_device: int = batch_size

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

    def setup(self, stage:str = 'fit') -> None:
        self.setup_batch_size_per_device()

        if self.hparams.split_mode == "random":
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            print(f'Dataset was randomly split with proportions: {self.hparams.train_val_test_split}')
        else:
            raise NotImplementedError(f'{self.hparams.train_val_test_split} split mode not implemented.')

        #     split_indices = {
        #         'train_indices': self.data_train.get('id'),
        #         'val_indices': self.data_val.get('id'),
        #         'test_indices': self.data_test.get('id')
        #     }
        # else:
        #     raise NotImplementedError
        #
        # timestamp = cdu.create_timestamp()
        # torch.save(split_indices, os.path.join(X, 'split_indices_{self.dataset_name}_{timestamp}.pth'))
        # print(f'Saved split indices to split_indices_{timestamp}.pth')

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
        )

if __name__ == "__main__":
    _ = BaseDataModule(None, None, None, None, None, None)
