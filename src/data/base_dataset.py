import os
from abc import ABC, abstractmethod
from typing import Any, Dict, final

import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: bool = False,
        dataset_name: str = "BaseDataset",
        seed: int = 12345,
        mode: str = "train",
        cache_dir: str = None,
    ) -> None:
        """Interface for any use case dataset.

        It is built on a model-ready csv file containing as columns:
        - lon, lat coordinates
        - target column(s)
        - auxiliary data columns
        - id column, essential for data splits.

        Dataset should return target and auxiliary data columns if requested, (`use_target_data`, `use_aux_data` parameters).
        The requested training modality(-ies) are specified through `modalities` parameter.

        :param path_csv: path to model ready csv file
        :param modalities: a list of modalities needed as EO data (for EO encoder)
        :param use_target_data: if target values should be returned
        :param use_aux_data: if auxiliary values should be returned
        :param dataset_name: dataset name
        :param seed: random seed
        :param mode: train/val/test mode of the dataset
        """

        # Set attributes
        self.data_dir = os.path.join(data_dir, dataset_name)
        assert os.path.exists(self.data_dir), f"{self.data_dir} does not exist."
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        assert os.path.exists(self.cache_dir), f"{self.cache_dir} does not exist."

        # read and shuffle df
        path_csv = os.path.join(self.data_dir, f"model_ready_{dataset_name}.csv")
        assert os.path.exists(path_csv), f"{path_csv} does not exist."
        self.df: pd.DataFrame = pd.read_csv(path_csv)
        # self.df = shuffle(self.df, random_state=seed)  # shuffling is already done in datamodules
        self.seed = seed

        # more precise dataset name (with modalities)
        self.dataset_name: str = dataset_name + "_" + "_".join(modalities)

        self.modalities: dict = modalities

        self.use_target_data: bool = use_target_data
        self.use_aux_data: bool = use_aux_data

        # Set placeholders
        self.num_classes: int | None = None
        self.target_names: list[str] | None = None
        self.aux_names: list[str] | None = None
        self.records: Dict[str] | None = None
        self.mode: str = mode  # 'train', 'val', 'test'

    @final
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.records)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a single item from the dataset."""
        pass

    def setup(self):
        pass
