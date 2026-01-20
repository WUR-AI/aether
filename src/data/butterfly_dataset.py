import os
import re
import shutil
from typing import Any, Dict, override

import numpy as np
import torch

import src.data_preprocessing.data_utils as du
from src.data.base_dataset import BaseDataset
from src.utils.errors import IllegalArgumentCombination


class ButterflyDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: bool = False,
        seed: int = 12345,
        cache_dir: str = None,
    ) -> None:
        """A dataset implementation for the Butterfly diversity use case.

        :param path_csv: path to model ready csv file
        :param modalities: a list of modalities needed as EO data (for EO encoder)
        :param use_target_data: if target values should be returned
        :param use_aux_data: if auxiliary values should be returned
        :param seed: random seed
        :param data_dir: path to data dir
        """

        super().__init__(
            data_dir, modalities, use_target_data, use_aux_data, "s2bms", seed, cache_dir
        )
        self.create_records()
        self.setup()  # needs to be called here in case number of data points changes, so that self._len is correctly set before dataloaders are created.

    def create_records(self, columns: list[str] = None) -> None:
        # Placeholder for filtered columns
        if columns is None:
            columns = ["name_loc"]

            for modality, params in self.modalities.items():
                if modality == "coords":
                    columns.extend(["lat", "lon"])
                else:
                    self.add_modality_paths_to_df(modality, params["format"])
                    columns.append(f"{modality}_path")
                    if modality == "s2":
                        if params["preprocessing"] == "zscored":
                            self.init_s2_norm_stats()
                    elif modality == "aef" or modality == "tessera":
                        pass
                    else:
                        raise ValueError(f"Unsupported modality: {modality}")

            if self.use_target_data:
                self.target_names = [c for c in self.df.columns if "target_" in c]
                columns.extend(self.target_names)
                self.num_classes = len(self.target_names)

            if self.use_aux_data:
                self.aux_names = [c for c in self.df.columns if "aux_" in c]
                columns.extend(self.aux_names)

            self.columns = columns
        self.records = self.df.loc[:, columns].to_dict("records")
        self._len = len(self.records)

    def setup(self):
        """Setups the whole dataset, makes available data of requested modalities."""

        if len(self.modalities.keys()) == 1 and self.modalities.get("coords", None) is not None:
            return

        import pooch

        # Initialise pooch client
        self.pooch_cli = pooch.create(
            path=os.path.join(self.cache_dir, "s2bms"),
            base_url="",
            registry=None,
        )

        # Add registry with all datasets, hashes and urls
        self.pooch_cli.load_registry(os.path.join(self.data_dir, "registry.txt"))

        # Set up each requested modality
        for mod, params in self.modalities.items():
            if mod == "s2":
                self.setup_s2bms()
            elif mod == "tessera":
                self.setup_tessera(year=params["year"], size=params["size"])
            elif mod == "aef":
                self.setup_aef()

    def setup_tessera(self, year: int, size: int) -> None:
        """Prepares (downloads, places)"""
        print("\n\nSetting up Tessera data...\n\n")
        # Check if data is already available
        dst_dir = os.path.join(self.data_dir, "eo/tessera")
        if os.path.exists(dst_dir):
            for p in self.df.tessera_path:
                if not os.path.basename(p) in os.listdir(dst_dir):
                    raise FileNotFoundError(f"Missing S2 data: {p}")
            return
        else:
            os.makedirs(dst_dir, exist_ok=True)

        # TODO: if we compile the dataset and use zenodo (or sth else) then change to pooch downloading/loading
        # Now it downloads from function calls
        from src.data_preprocessing.tessera_embeds import tessera_from_df

        tessera_from_df(
            self.df,
            data_dir=os.path.join(self.data_dir, "eo/tessera"),
            year=year,
            tile_size=size,
            cache_dir=self.cache_dir,
        )

    def setup_aef(self) -> None:
        print("\n\nSetting up AEF data...\n\n")

        dst_dir = os.path.join(self.data_dir, "eo/aef")
        assert os.path.exists(dst_dir), f"AEF data directory {dst_dir} does not exist."

        inds_keep = []
        for i_row, row in self.df.iterrows():
            p = row.aef_path
            if os.path.exists(p):
                inds_keep.append(i_row)
        inds_keep = np.array(inds_keep)
        print(f"Keeping {len(inds_keep)}/{len(self.df)} entries with available AEF data.")
        self.df = self.df.iloc[inds_keep].reset_index(drop=True)
        self.create_records(columns=self.columns)  # recreate records after filtering df

    def setup_s2bms(self) -> None:
        """Prepares (downloads, renames and moves) data from S2BMS study."""
        print("\n\nSetting up S2BMS data...\n\n")
        import pooch

        # Check if data is already available
        dst_dir = os.path.join(self.data_dir, "eo/s2")
        if os.path.exists(dst_dir):
            n_files = len(os.listdir(dst_dir))
            if n_files == 0:
                print("Warning: S2 data directory exists but is empty, re-downloading data.")
            elif n_files < len(self.df):
                print(
                    f"Warning: S2 data directory exists but has only {n_files} files, expected {len(self.df)}. Re-downloading data."
                )
            else:
                for p in self.df.s2_path:
                    if not os.path.basename(p) in os.listdir(dst_dir):
                        raise FileNotFoundError(f"Missing S2 data: {p}")
                return
        else:
            os.makedirs(dst_dir, exist_ok=True)

        # Download data through pooch
        fnames = self.pooch_cli.fetch("S2BMS.zip", processor=pooch.Unzip())

        # Copy ukbms_species-presence
        # df_dir = os.path.dirname([n for n in fnames if 'ukbms_species-presence' in n and 'MACOSX' not in n and '.DS_Store' not in n][0])
        # shutil.move(df_dir, 'source/ukbms_species-presence')

        # Move files to data dir
        LOC_PATTERN = re.compile(r"([A-Z0-9]+_loc-\d+)")
        for fname in fnames:
            if ("__MACOSX" in fname or "DS_Store" in fname) or "tif" not in fname:
                continue

            # Get name_loc from the file name
            m = LOC_PATTERN.search(fname)
            if not m:
                raise ValueError(f"Could not extract name_loc from filename: {fname}")

            # Move
            new_name = "s2_<name_loc>.tif".replace("<name_loc>", m.group(1))
            dst_path = os.path.join(dst_dir, new_name)
            shutil.move(fname, dst_path)

        with open(os.path.join(dst_dir, "meta.tx"), "w") as f:
            f.writelines("Data from S2BMS study\n")
            f.writelines("Containing 4 channel S2 256x256px imagery.\n")
            # TODO: add more

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

    def init_s2_norm_stats(self, means: list[float] = None, stds: list[float] = None):
        if means is None or stds is None:
            print("Using S2BMS default zscore means and stds")
            means = np.array([661.1047, 770.6800, 531.8330, 3228.5588]).astype(
                np.float32
            )  # computed across entire ds
            stds = np.array([640.2482, 571.8545, 597.3570, 1200.7518]).astype(np.float32)
        if self.modalities["s2"]["channels"] == "rgb":
            means = means[:3]
            stds = stds[:3]
        self.s2_norm_means = means[:, None, None]
        self.s2_norm_std = stds[:, None, None]

    def zscore_image(self, im: np.ndarray):
        """Apply preprocessing function to a single image.

        raw_sent2_means = torch.tensor([661.1047,  770.6800,  531.8330, 3228.5588]) raw_sent2_stds
        = torch.tensor([640.2482,  571.8545,  597.3570, 1200.7518])
        """
        im = (im - self.s2_norm_means) / self.s2_norm_std
        return im

    def load_s2(self, filepath: str):
        im = du.load_tiff(filepath, datatype="np")

        if self.modalities["s2"]["channels"] == "4c":
            pass
        elif self.modalities["s2"]["channels"] == "rgb":
            im = im[:3, :, :]
        else:
            raise IllegalArgumentCombination(
                f"Channel specification {self.n_bands} is not implemented."
            )

        if self.modalities["s2"]["preprocessing"] == "zscored":
            im = im.astype(np.int32)
            im = self.zscore_image(im)
        else:
            im = np.clip(im, 0, 2000)
            im = im / 2000.0
        return torch.tensor(im).float()

    def load_npy(self, filepath: str):
        im = np.load(filepath).transpose(2, 0, 1)
        return torch.from_numpy(im).float()

    def load_aef(self, filepath: str):
        im = du.load_tiff(filepath, datatype="np")
        if np.isinf(im).any():
            im = np.clip(im, a_min=-0.5, a_max=0.5)
        return torch.tensor(im).float()

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        assert idx < len(
            self.records
        ), f"Index {idx} out of bounds for dataset of size {len(self.records)} while len ds is {self._len} and {self.__len__()}."
        row = self.records[idx]

        formatted_row = {"eo": {}}

        for modality in self.modalities:
            if modality in ["coords"]:
                formatted_row["eo"][modality] = torch.tensor([row["lat"], row["lon"]])
            elif modality == "s2":
                formatted_row["eo"][modality] = self.load_s2(row["s2_path"])
                # TODO: augmentations
            elif modality == "tessera":
                formatted_row["eo"][modality] = self.load_npy(row["tessera_path"])
                # TODO any normalisation needed
            elif modality == "aef":
                formatted_row["eo"][modality] = self.load_aef(row["aef_path"])

        if self.use_target_data:
            formatted_row["target"] = torch.tensor(
                [row[k] for k in self.target_names], dtype=torch.float32
            )

        if self.use_aux_data:
            formatted_row["aux"] = [row[i] for i in self.aux_names]

        return formatted_row


if __name__ == "__main__":
    _ = ButterflyDataset(None, None, None, None, None, None, None, None)
