import os
from abc import ABC, abstractmethod
from typing import Any, Dict, final

import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        path_csv: str,
        modalities: list[str],
        use_target_data: bool = True,
        use_aux_data: bool = False,
        dataset_name: str = "BaseDataset",
        seed: int = 12345,
        mode: str = "train",
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

        # read and shuffle df
        assert os.path.exists(path_csv), f"{path_csv} does not exist."
        self.df: pd.DataFrame = pd.read_csv(path_csv)
        self.df = shuffle(self.df, random_state=seed)
        self.seed = seed

        # Set attributes
        self.dataset_name: str = dataset_name + "_" + "_".join(modalities)

        self.modalities: list[str] = modalities
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
