import os
from typing import Any, Dict, List, override

import numpy as np
import pooch
import torch

import src.data_preprocessing.data_utils as du
from src.data.base_dataset import BaseDataset
from src.data_preprocessing.renaming_utils import rename_s2bms
from src.utils.errors import IllegalArgumentCombination


class ButterflyDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: Dict[str, List[str] | str] | None = None,
        seed: int = 12345,
        cache_dir: str = None,
        mock: bool = False,
    ) -> None:
        """A dataset implementation for the Butterfly diversity use case.

        :param data_dir: path to data dir
        :param modalities: a dict of modalities needed as EO data (for EO encoder) (e.g.,
            {"coords": None, "s2": {"channels": "rgb", "preprocessing": "zscored"}})
        :param use_target_data: if target values should be returned
        :param use_aux_data: if auxiliary values should be returned
        :param seed: random seed
        :param cache_dir: path to cache dir
        :param mock: whether to mock csv file
        """

        super().__init__(
            data_dir=data_dir,
            modalities=modalities,
            use_target_data=use_target_data,
            use_aux_data=use_aux_data,
            dataset_name="s2bms",
            seed=seed,
            cache_dir=cache_dir,
            implemented_mod={"s2", "tessera", "coords", "aef"},
            mock=mock,
        )

    def setup(self):
        """Setups the whole dataset, makes available data of requested modalities."""

        # Set up each requested modality
        for mod in self.modalities.keys():
            if mod == "coords" and len(self.modalities.keys()) == 1:
                return
            elif mod == "s2":
                self.setup_s2bms()
                if self.modalities["s2"].get("preprocessing", "") == "zscored":
                    self.init_norm_stats()
            elif mod == "tessera":
                self.setup_tessera()
            elif mod == "aef":
                self.setup_aef()

    def setup_s2bms(self) -> None:
        """Prepares (downloads, renames and moves) data from S2BMS study."""
        print("\n\nSetting up S2BMS data...\n\n")

        # Check if data is already available
        dst_dir = os.path.join(self.data_dir, "eo/s2")

        # If data does not exist or is empty → full download
        if not os.path.exists(dst_dir) or len(os.listdir(dst_dir)) == 0:
            if self.pooch_cli is None:
                self.pooch_setup()

            os.makedirs(dst_dir, exist_ok=True)
            fnames = self.pooch_cli.fetch("S2BMS.zip", processor=pooch.Unzip())

            # Copy ukbms_species-presence
            # df_dir = os.path.dirname([n for n in fnames if 'ukbms_species-presence' in n and 'MACOSX' not in n and '.DS_Store' not in n][0])
            # shutil.move(df_dir, 'source/ukbms_species-presence')

            # Move files to data dir
            rename_s2bms(dst_dir, fnames)

            with open(os.path.join(dst_dir, "meta.txt"), "w") as f:
                f.writelines("Data from S2BMS study\n")
                f.writelines("Containing 4 channel S2 256x256px imagery.\n")
                # TODO: add more

        else:
            # Check for missing files
            avail_files = os.listdir(dst_dir)
            for rec in self.records:
                fname = os.path.basename(rec["s2_path"])
                if fname not in avail_files:
                    raise FileNotFoundError(f"Missing S2 data: {fname}")
                # TODO potentially handle single missing files with GEE API?

    def init_norm_stats(self, means: list[float] = None, stds: list[float] = None):
        """Initializes normalization statistics for the original S2BMS dataset."""
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

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
            formatted_row["aux"] = {}
            for aux_cat, vals in self.use_aux_data.items():
                if aux_cat == "aux":
                    formatted_row["aux"][aux_cat] = torch.tensor(
                        [row[v] for v in vals], dtype=torch.float32
                    )
                else:
                    formatted_row["aux"][aux_cat] = [row[v] for v in vals]

        return formatted_row


if __name__ == "__main__":
    _ = ButterflyDataset(None, None, None, None, None, None, None, None)
