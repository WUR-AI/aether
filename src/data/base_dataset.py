import os
from abc import ABC, abstractmethod
from typing import final, Dict, Any

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
            dataset_name:str='BaseDataset',
            random_state=42,
            mode: str = 'train'
    ) -> None:

        # read and shuffle df
        assert os.path.exists(path_csv), f'{path_csv} does not exist.'
        self.df: pd.DataFrame = pd.read_csv(path_csv)
        self.df = shuffle(self.df, random_state=random_state)

        # Set attributes
        self.dataset_name: str = dataset_name + '_'+ '_'.join(modalities)

        self.modalities: list[str] = modalities
        self.use_target_data: bool = use_target_data
        self.use_aux_data: bool = use_aux_data

        # Set placeholders
        self.num_classes: int | None = None
        self.target_names: list[str] | None = None
        self.aux_names: list[str] | None = None
        self.records: Dict[str] | None = None

        self.mode = mode  # 'train', 'val', 'test'

    @final
    def __len__(self) -> int:
        return len(self.records)

    @abstractmethod
    def __getitem__(
            self,
            idx: int
    ) ->  Dict[str, Any]:
        pass