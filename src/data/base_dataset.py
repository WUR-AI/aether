import os
from abc import ABC, abstractmethod
from typing import Any, Dict, final

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import src.data_preprocessing.data_utils as du


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
        implemented_mod: set[str] = None,
        mock: bool = False,
    ) -> None:
        """Interface for any use case dataset.

        It is built on a model-ready csv file containing as columns:
        - lon, lat coordinates
        - target column(s)
        - auxiliary data columns
        - id column, essential for data splits.

        Dataset should return target and auxiliary data columns if requested, (`use_target_data`, `use_aux_data` parameters).
        The requested training modality(-ies) are specified through `modalities` parameter.

        :param data_dir: data directory
        :param modalities: a list of modalities needed as EO data (for EO encoder)
        :param use_target_data: if target values should be returned
        :param use_aux_data: if auxiliary values should be returned
        :param dataset_name: dataset name
        :param seed: random seed
        :param mode: train/val/test mode of the dataset
        :param cache_dir: directory to save cached data
        :param implemented_mod: implemented modalities for each dataset
        :param mock: whether to mock csv file
        """

        if mock:
            dataset_name = "mock"

        # Modalities
        self.implemented_mod = implemented_mod
        self.modalities: dict = modalities
        for mod in self.modalities.keys():
            if mod not in self.implemented_mod:
                raise ValueError(f"{mod} not in implemented modalities.")
        # more precise dataset name (with modalities)
        self.dataset_name: str = dataset_name + "_" + "_".join(modalities)

        # Set data attributes
        self.registry_path = os.path.join(data_dir, "registry.txt")
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        for d in [self.data_dir, self.cache_dir]:
            os.makedirs(d, exist_ok=True)

        # Read model ready csv df
        path_csv = os.path.join(self.data_dir, f"model_ready_{dataset_name}.csv")
        assert os.path.exists(path_csv), f"{path_csv} does not exist."
        self.df: pd.DataFrame = pd.read_csv(path_csv)

        # Other attributes or placeholders
        self.seed = seed
        self.num_classes = None
        self.mode: str = mode  # 'train', 'val', 'test'
        self.use_target_data: bool = use_target_data
        self.use_aux_data: bool = use_aux_data
        self.records: dict[str, Any] = self.get_records()

    @final
    def get_records(self) -> dict[str, Any]:
        """Gets record dictionary from the dataframe based on what is needed for the model (aux,
        target columns, modality paths)"""

        # Placeholder for filtered columns
        columns = ["name_loc"]

        # Modality columns
        for modality, params in self.modalities.items():
            if modality == "coords":
                columns.extend(["lat", "lon"])
            else:
                # Add paths
                self.add_modality_paths_to_df(modality, params["format"])
                columns.append(f"{modality}_path")

        # Include targets
        if self.use_target_data:
            self.target_names = [c for c in self.df.columns if "target_" in c]
            columns.extend(self.target_names)
            self.num_classes = len(self.target_names)

        # Include aux data
        if self.use_aux_data:
            self.aux_names = [c for c in self.df.columns if "aux_" in c]
            columns.extend(self.aux_names)

        return self.df.loc[:, columns].to_dict("records")

    @final
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.records)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a single item from the dataset."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Setups the whole dataset, makes available data of requested modalities."""
        pass

    @final
    def add_modality_paths_to_df(self, modality: str, extension: str) -> None:
        """Add modality path column to df.

        :param modality: modality name
        :param extension: file extension
        :return: None
        """
        # Directory path
        path = f"{self.data_dir}/eo/{modality}/"

        # Df column name
        col = f"{modality}_path"

        # Write paths
        self.df[col] = None
        for i, row in self.df.iterrows():
            file_path = path + f"{modality}_{row.name_loc}.{extension}"
            self.df.loc[i, col] = file_path

    @final
    def setup_tessera(self) -> None:
        """Download full dataset or the missing Tessera dataset.

        Right now retrieval is through GeoTessera API
        """

        print("\n\nSetting up Tessera data...\n\n")
        from geotessera import GeoTessera

        from src.data_preprocessing.tessera_embeds import (
            get_tessera_embeds,
            tessera_from_df,
        )

        year = self.modalities["tessera"].get(
            "year", KeyError('Missing parameter "year" for Tessera modality')
        )
        size = self.modalities["tessera"].get(
            "size", KeyError('Missing parameter "size" for Tessera modality')
        )

        # Check if data is already available
        dst_dir = os.path.join(self.data_dir, "eo/tessera")

        # If data does not exist or is empty → full download
        if not os.path.exists(dst_dir) or len(os.listdir(dst_dir)) == 0:
            os.makedirs(dst_dir, exist_ok=True)

            tessera_from_df(
                self.df,
                data_dir=dst_dir,
                year=year,
                tile_size=size,
                cache_dir=self.cache_dir,
            )

            # TODO: if we compile the dataset and use zenodo (or sth else) then change to pooch downloading/loading
            # TODO: in case of zenodo use may need to be moved to UC dataset subclasses
            # or self.setup_tessera_from_pooch() <- per children class implementation

        else:
            # Download missing rows (if any)
            avail_files = os.listdir(dst_dir)
            gt = None
            for rec in self.records:
                fname = os.path.basename(rec["tessera_path"])
                if fname not in avail_files:
                    print(f"Retrieving missing Tessera data: {fname}")
                    gt = gt or GeoTessera(cache_dir=self.cache_dir)
                    get_tessera_embeds(rec.lon, rec.lat, rec.name_loc, year, dst_dir, size)

    @final
    def setup_aef(self) -> None:
        """Download full dataset or the missing AEF tiles.

        Right now retrieval is through GEE API
        """

        print("\n\nSetting up AEF data...\n\n")

        dst_dir = os.path.join(self.data_dir, "eo/aef")

        # TODO aef retrieval?
        # TODO: in case of zenodo use may need to be moved to UC dataset subclasses
        # or self.setup_aef_from_pooch() <- per children class implementation

    @final
    def pooch_setup(self) -> None:
        """Initialises pooch connection and loads registry."""
        import pooch

        # Initialise pooch client
        self.pooch_cli = pooch.create(
            path=os.path.join(self.cache_dir, self.data_dir),
            base_url="",
            registry=None,
        )

        # Add registry with all datasets, hashes and urls
        self.pooch_cli.load_registry(self.registry_path)

    @final
    def load_npy(self, filepath: str) -> torch.Tensor:
        """Loads numpy array from file as a tensor."""
        im = np.load(filepath).transpose(2, 0, 1)
        return torch.from_numpy(im).float()

    @final
    def load_aef(self, filepath: str):
        """Loads AEF data from file as a tensor."""

        im = du.load_tiff(filepath, datatype="np")
        if np.isinf(im).any():
            im = np.clip(im, a_min=-0.5, a_max=0.5)
        return torch.tensor(im).float()
