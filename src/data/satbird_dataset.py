import os
from typing import override

import torch
from rasterio import open as ropen
from torchvision.transforms import v2

from src.data.base_dataset import BaseDataset
from src.data_preprocessing.satbird import setup_satbird_from_pooch


class SatBirdDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool,
        use_aux_data: bool,
        seed: int,
        study_site: str,
        cache_dir: str = None,
    ):
        """A dataset implementation for the Butterfly diversity use case.

        :param data_dir: path to data dir
        :param modalities: a list of modalities needed as EO data (for EO encoder)
        :param use_target_data: if target values should be returned
        :param use_aux_data: if auxiliary values should be returned
        :param seed: random seed
        :param cache_dir: path to cache dir
        :param study_site: study site name [Kenya, USA_summer, USA_winter]
        """
        # assert study_site in ["Kenya", "USA_summer", "USA_winter"]
        assert study_site in ["Kenya"]

        super().__init__(
            data_dir=data_dir,
            modalities=modalities,
            use_target_data=use_target_data,
            use_aux_data=use_aux_data,
            dataset_name=f"satbird-{study_site}",
            seed=seed,
            cache_dir=cache_dir,
            implemented_mod=["coords", "s2", "s2rgb", "tessera"],
        )

        self.study_site = study_site

    @override
    def setup(self):
        """Setups the whole dataset, makes available data of requested modalities."""

        # Set up each requested modality

        for mod in self.modalities.keys():
            if mod == "coords" and len(self.modalities.keys()) == 1:
                return
            if mod in ["s2", "s2rgb", "environmental"]:
                self.setup_satbird()
            elif mod == "tessera":
                self.setup_tessera()

    def setup_satbird(self):
        """Prepares (downloads, renames and moves) data for each requested modality."""
        print(f"\n\nSetting up SatBird {self.study_site} data...\n\n")

        # Check if data is already available
        dst_dirs = [os.path.join(self.data_dir, "eo", i) for i in ["s2", "s2rgb", "environmental"]]

        # If data does not exist or is empty → full download
        for dst_dir in dst_dirs:
            if not os.path.exists(dst_dir) or len(os.listdir(dst_dir)) == 0:
                setup_satbird_from_pooch(
                    self.data_dir, self.cache_dir, self.study_site, self.registry_path
                )
                return

    def load_s2(self, path: str):
        """Loads S2 data from path."""
        img = ropen(path).read()
        tensor = v2.ToImage()(img).permute(1, 2, 0)
        # TODO normalisations etc
        return tensor

    @override
    def __getitem__(self, idx):
        row = self.records[idx]

        formatted_row = {"eo": {}}

        for modality in self.modalities:
            if modality in ["coords"]:
                formatted_row["eo"][modality] = torch.tensor([row["lat"], row["lon"]])
            elif modality in ["s2", "s2rgb"]:
                formatted_row["eo"][modality] = self.load_s2(row[f"{modality}_path"])
                # TODO: augmentations
            elif modality == "tessera":
                formatted_row["eo"][modality] = self.load_npy(row["tessera_path"])
                # TODO any normalisation needed

        if self.use_target_data:
            formatted_row["target"] = torch.tensor(
                [row[k] for k in self.target_names], dtype=torch.float32
            )

        if self.use_aux_data:
            formatted_row["aux"] = [row[i] for i in self.aux_names]

        return formatted_row


if __name__ == "__main__":
    _ = SatBirdDataset(None, None, None, None, None, None, None)
