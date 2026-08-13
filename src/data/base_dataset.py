import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, final

import numpy as np
import pandas as pd
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from src.data_preprocessing.tessera_embeds import NoTileError, PartialTileError
from src.utils.data_utils import center_crop_npy
from src.utils.errors import MissingConfigurationError, MissingDataError

log = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: Dict[str, List[str] | str] | str | None = None,
        dataset_name: str | List[str] = "BaseDataset",
        seed: int = 12345,
        cache_dir: str = None,
        implemented_mod: set[str] = None,
        mock: bool = False,
        use_features: bool = True,
        csv_name: str = None,
        dtype: str = "float32",
        return_name_loc: bool = False,
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
        :param modalities: a dict of modalities needed as EO data (for EO encoder) (e.g., {"coords": None, "s2": {"channels": "rgb", "preprocessing": "zscored"}})
        :param use_target_data: if target values should be returned
        :param use_aux_data: if auxiliary values should be returned
        :param dataset_name: dataset name
        :param seed: random seed
        :param cache_dir: directory to save cached data
        :param implemented_mod: implemented modalities for each dataset
        :param mock: whether to mock csv file
        :param use_features: if tabular feat_* columns should be included. Default True.
        :param dtype: global dtype (used if not specified for each modality individually), also used for aux, target
        """

        if mock:
            dataset_name = "mock"

        # Dtype
        assert getattr(torch, dtype), KeyError(f"Requested dtype {dtype} is not supported.")
        self.dtype: str = getattr(torch, dtype)

        # Modalities
        self.implemented_mod = implemented_mod
        self.modalities = modalities

        # Check modalities and set dtypes
        for mod, configs in self.modalities.items():
            if mod not in self.implemented_mod:
                raise ValueError(f"{mod} not in implemented modalities.")

            if configs is not None:
                m_dtype = configs.get("dtype", dtype)
                self.modalities[mod]["dtype"] = m_dtype  # Overwrite if dtype was not specified
                log.info(f"Dtype of {mod} set to {m_dtype}")
            else:
                m_dtype = dtype
                self.modalities[mod] = {"dtype": m_dtype}

        # Set data attributes
        self.registry_path = os.path.join(data_dir, "registry.txt")
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Read model ready csv df
        csv_filename = csv_name or f"model_ready_{dataset_name}.csv"
        path_csv = os.path.join(self.data_dir, csv_filename)
        assert os.path.exists(
            path_csv
        ), f"{path_csv} does not exist. (Expecting {csv_filename} to exist in {self.data_dir})"
        self.df = pd.read_csv(path_csv)

        # Other attributes or placeholders
        self.pooch_cli = None
        self.num_classes = None
        self.tabular_dim = None
        self.seed = seed
        self.use_target_data = use_target_data
        self.use_features = use_features

        self.configure_use_aux(use_aux_data)

        self.configure_use_feats(use_features)

        # More precise dataset name (with modalities)
        if isinstance(dataset_name, list):
            dataset_name = "+".join(dataset_name)
        self.dataset_name: str = dataset_name + "_" + "_".join(modalities)

        self.columns: List[Any] = self.get_columns()
        self.records: List[Any] = []

        self.return_name_loc: bool = return_name_loc
        self._setup_flag = False
        self._ignore_single_missing_data_points = True

    def configure_use_feats(self, use_features):
        if isinstance(use_features, DictConfig):
            self.use_features = OmegaConf.to_container(use_features, resolve=True)
        elif use_features is True or use_features == "all":
            self.use_features = {
                "pattern": "^feat_.*",
                #     'columns' : []
            }
        else:
            self.use_features = None

    def configure_use_aux(self, use_aux_data):
        if isinstance(use_aux_data, DictConfig):
            self.use_aux_data = OmegaConf.to_container(use_aux_data, resolve=True)
        elif isinstance(use_aux_data, dict):
            self.use_aux_data = use_aux_data
        elif use_aux_data is True or use_aux_data == "all":
            self.use_aux_data = {
                "aux": {
                    "pattern": "^aux_(?!.*top).*",
                    #     'columns' : []
                },
                "top": {
                    "pattern": "^aux_.*top.*",
                    #     'columns' : []
                },
            }
        else:
            self.use_aux_data = None

    @final
    def get_columns(self) -> List[str]:
        """Gets record dictionary from the dataframe based on what is needed for the model (aux,
        target columns, modality paths)"""

        # Placeholder for filtered columns
        columns = ["name_loc"]

        # Modality columns
        for modality, params in self.modalities.items():
            if modality == "coords":
                columns.extend(["lat", "lon"])
            elif modality in ["aef_avr", "tessera_avr"]:
                continue
            else:
                # Add paths
                self.add_modality_paths_to_df(
                    modality,
                    params.get("format"),
                )
                columns.append(f"{modality}_path")

        # Include targets
        if self.use_target_data:
            self.target_names = [c for c in self.df.columns if "target_" in c]
            columns.extend(self.target_names)
            self.num_classes = len(self.target_names)

        # Include aux data
        if self.use_aux_data is not None:
            for k, val in self.use_aux_data.items():
                if val.get("pattern"):
                    pattern = re.compile(val["pattern"])
                    aux_names = [x for x in self.df.columns if pattern.match(x)]
                else:
                    aux_names = val.get(
                        "columns",
                        ValueError('use_aux_data should have "pattern" or "columns" defined'),
                    )
                self.use_aux_data[k] = aux_names
                columns.extend(aux_names)

        # Include tabular features
        if self.use_features:
            if "pattern" in self.use_features:
                pattern = re.compile(self.use_features["pattern"])
                feat_names = [x for x in self.df.columns if pattern.match(x)]
            else:
                feat_names = self.use_features.get(
                    "columns",
                    ValueError('use_features should have "pattern" or "columns" defined'),
                )
            self.feat_names = feat_names
            self._feat_norm_setup()
            columns.extend(feat_names)

            self.tabular_dim = len(self.feat_names)  # drop any duplicates

        return list(set(columns))

    def _feat_norm_setup(self):
        """If statistics files provided for the features, read them into self.feat_stats
        parameter."""

        if "stats_file" in self.use_features:
            with open(self.use_features["stats_file"]) as json_data:
                d = json.load(json_data)

            means = [d[f]["mean"] for f in self.feat_names]
            stds = [d[f]["std"] if d[f]["std"] > 1e-8 else 1.0 for f in self.feat_names]

            self._feat_mean = torch.tensor(means, dtype=torch.float32)
            self._feat_std = torch.tensor(stds, dtype=torch.float32)
        else:
            self._feat_mean = None
            self._feat_std = None

    def get_records(self):
        return self.df.loc[:, self.columns].to_dict("records")

    @final
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.records)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a single item from the dataset."""
        pass

    @final
    def setup(self) -> None:
        """Setups the whole dataset, makes available data of requested modalities and filters out
        records for any location missing any modality data,"""
        if not self._setup_flag:
            self._setup()  # to be implemented for each UC

            self.records = self.get_records()
            self._setup_flag = True

    def _setup(self):
        pass

    @final
    def add_modality_paths_to_df(self, modality: str, extension: str) -> None:
        """Add modality path column to df.

        :param modality: modality name
        :param extension: file extension
        :return: None
        """
        assert extension in ["tif", "npy"], MissingConfigurationError(
            f"Please specify a file extension for {modality}"
        )
        # Directory path
        path = f"{self.data_dir}/eo/{modality}/"

        # Df column name
        col = f"{modality}_path"

        # Write paths
        self.df = pd.concat(
            [
                self.df,
                self.df["name_loc"]
                .apply(lambda loc: path + f"{modality}_{loc}.{extension}")
                .rename(col),
            ],
            axis=1,
        )

    @final
    def setup_tessera(self) -> None:
        """Download full dataset or the missing Tessera dataset.

        Right now retrieval is through GeoTessera API
        """

        logging.info("Setting up Tessera data...")
        download_missing_tiles = False

        # Check if data is already available
        dst_dir = os.path.join(self.data_dir, "eo/tessera")

        year = self.modalities["tessera"].get(
            "year", KeyError('Missing parameter "year" for Tessera modality')
        )
        size = self.modalities["tessera"].get(
            "size", KeyError('Missing parameter "size" for Tessera modality')
        )
        version = self.modalities["tessera"].get("version") or "v1.1"

        # If data does not exist or is empty → full download
        if not os.path.exists(dst_dir) or len(os.listdir(dst_dir)) == 0:
            from src.data_preprocessing.tessera_embeds import tessera_from_df

            if download_missing_tiles:
                os.makedirs(dst_dir, exist_ok=True)

                tessera_from_df(
                    self.df,
                    data_dir=dst_dir,
                    year=year,
                    tile_size=size,
                    cache_dir=self.cache_dir,
                    version=version,
                )
                if self._ignore_single_missing_data_points:
                    mask = self.df["tessera_path"].apply(
                        lambda p: os.path.basename(p) in avail_files
                    )
                    self.df = self.df[mask]
                    log.info(
                        f"Dropped {(~mask).sum()} locations because they had missing tessera tiles."
                    )
                else:
                    raise MissingDataError(
                        "Please download the missing Tessera tiles from src/data_preprocessing/tessera_embeds"
                    )

                # TODO: if we compile the dataset and use zenodo (or sth else) then change to pooch downloading/loading
                # TODO: in case of zenodo use may need to be moved to UC dataset subclasses
                # or self.setup_tessera_from_pooch() <- per children class implementation
            else:
                raise MissingDataError(
                    "Please download the Tessera tiles from src/data_preprocessing/tessera_embeds"
                )

        # Download missing rows (if any)
        else:
            log.info("Checking missing Tessera tiles...")
            avail_files = set(os.listdir(dst_dir))
            mask = self.df["tessera_path"].apply(lambda p: os.path.basename(p) in avail_files)
            if mask.all():
                return  # all data is available
            if download_missing_tiles:
                log.warning("May download tessera tiles filled with 0a")
                from geotessera import GeoTessera

                from src.data_preprocessing.tessera_embeds import (
                    get_tessera_embeds,
                    tessera_from_df,
                )

                gt = GeoTessera(cache_dir=self.cache_dir)

                missing_df = self.df[~mask]
                # Try downloading missing tiles for each location
                for _, row in missing_df.iterrows():
                    fname = os.path.basename(row["tessera_path"])
                    log.info(f"Retrieving missing Tessera data: {fname}")
                    lon, lat = row.lon.item(), row.lat.item()
                    try:
                        get_tessera_embeds(
                            lon,
                            lat,
                            row["name_loc"],
                            year=year,
                            save_dir=dst_dir,
                            tile_size=size,
                            tessera_con=gt,
                        )
                    except NoTileError or PartialTileError as e:
                        if self._ignore_single_missing_data_points:
                            log.info(f"Tile for {fname} could not be retrieved. Error: {e}")
                        else:
                            raise e
                mask = self.df["tessera_path"].apply(lambda p: os.path.basename(p) in avail_files)
                self.df = self.df[mask]
                log.info(
                    f"Dropped {(~mask).sum()} locations because they had missing tessera tiles."
                )

            elif self._ignore_single_missing_data_points:
                self.df = self.df[mask]
                log.info(
                    f"Dropped {(~mask).sum()} locations because they had missing tessera tiles."
                )
            else:
                raise MissingDataError(
                    "Please download the missing Tessera tiles from src/data_preprocessing/tessera_embeds"
                )

    @final
    def setup_aef(self) -> None:
        """Download full dataset or the missing AEF tiles.

        Right now retrieval is through GEE API
        """

        log.info("Setting up AEF data...")

        dst_dir = os.path.join(self.data_dir, "eo/aef")
        avail_files = os.listdir(dst_dir)

        mask = self.df["aef_path"].apply(lambda p: os.path.basename(p) in avail_files)

        if mask.all():
            return
        elif (~mask).any() and self._ignore_single_missing_data_points:
            self.df = self.df[mask]
            log.info(f"Dropped {(~mask).sum()} locations because they had missing aef tiles.")
        else:
            raise MissingDataError(
                f"Missing aef data for {len(self.df[mask].name_loc)} locations. \n Please download the missing aef tiles"
            )

        # TODO aef retrieval?
        # TODO: in case of zenodo use may need to be moved to UC dataset subclasses
        # or self.setup_aef_from_pooch() <- per children class implementation

    @final
    def pooch_setup(self) -> None:
        """Initialises pooch connection and loads registry."""
        import pooch

        # Initialise pooch client
        self.pooch_cli = pooch.create(
            path=self.cache_dir,
            base_url="",
            registry=None,
        )

        # Add registry with all datasets, hashes and urls
        self.pooch_cli.load_registry(self.registry_path)

    @final
    def load_npy(self, filepath: str, dtype: np.dtype) -> np.ndarray:
        """Loads numpy array from file as a tensor."""
        arr = np.load(filepath).transpose(2, 0, 1)
        if arr.dtype != np.dtype(dtype):
            arr = arr.astype(dtype=dtype, copy=False)

        return arr

    @final
    def load_tiff(self, tiff_file_path: str, dtype: np.dtype) -> np.ndarray:
        """Load tiff file as np array of a specified dtype."""

        with rasterio.open(tiff_file_path) as f:
            im = f.read()
            assert isinstance(im, np.ndarray)
            if im.dtype != np.dtype(dtype):
                im = im.astype(dtype=dtype, copy=False)
        return im

    @final
    def load_aef(self, filepath: str):
        """Loads AEF data from file as a tensor."""

        # Modality settings
        size = self.modalities["aef"]["size"]
        dtype = self.modalities["aef"].get("dtype")
        format = self.modalities["aef"].get("format", "npy")
        dtype, is_bfloat16 = self.resolve_dtype(dtype)

        if format in "tif":
            im = self.load_tiff(filepath, np.dtype(dtype))
        else:
            im = self.load_npy(filepath, np.dtype(dtype))

        if im.shape[-2:] != (size, size):
            im = center_crop_npy(im, (64, size, size))

        # Scan for inf values and clip them (in memory)
        if self.modalities["aef"].get("enable_nans", False):
            if np.isinf(im).any():
                im[np.isinf(im)] = np.nan
        else:
            np.clip(im, -0.5, 0.5, out=im)
            # TODO any other normalisation needed

        tensor = torch.from_numpy(im)
        if is_bfloat16:
            tensor = tensor.to(torch.bfloat16)
        return tensor

    @final
    def load_tessera(self, filepath: str) -> torch.Tensor:
        """Loads."""
        size = self.modalities["tessera"]["size"]
        dtype = self.modalities["tessera"]["dtype"]
        dtype, is_bfloat16 = self.resolve_dtype(dtype)

        arr = self.load_npy(filepath, np.dtype(dtype))

        if arr.shape[-2:] != (size, size):
            arr = center_crop_npy(arr, (128, size, size))

        if self.modalities["tessera"].get("enable_nans", False):
            # Nans are 0 across all 128 channels
            mask = np.all(arr == 0, axis=0)
            arr[mask] = torch.nan
        # TODO any normalisation needed

        tensor = torch.from_numpy(arr)
        if is_bfloat16:
            tensor = tensor.to(torch.bfloat16)
        return tensor

    @staticmethod
    def resolve_dtype(dtype_str: str):
        """Resolve dtype from string into numpy dtype and return flag for mixed precision dtype in
        tensors."""
        is_bfloat16 = dtype_str == "bfloat16"
        np_dtype = np.float32 if is_bfloat16 else np.dtype(dtype_str)

        return np_dtype, is_bfloat16

    def setup_embeds(self, modality):
        params = self.modalities[modality]
        dtype = getattr(torch, self.modalities[modality].get("dtype"))

        path = params.get("path", KeyError(f"Please specify {modality} path to csv file"))
        assert os.path.exists(path), FileNotFoundError(f"{path} does not exist.")
        df = pd.read_csv(path)

        # Filter out locations without data for embeddings
        common = set(df["name_loc"]) & set(self.df["name_loc"])
        df = df[df["name_loc"].isin(common)]
        df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        self.df = self.df[self.df["name_loc"].isin(common)]

        if modality == "aef_avr":
            emb_cols = [f"emb_{i}" for i in range(64)]
        else:
            emb_cols = [f"emb_{i}" for i in range(128)]

        lookup_values = df[emb_cols].to_numpy()
        lookup = {
            name_loc: torch.tensor(lookup_values[i], dtype=dtype)
            for i, name_loc in enumerate(df["name_loc"])
        }

        if modality == "aef_avr":
            self.aef_avr = lookup
        else:
            self.tessera_avr = lookup
