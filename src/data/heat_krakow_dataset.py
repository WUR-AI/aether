"""Heat Krakow LST dataset.

Location: src/data/heat_krakow_dataset.py

Based on Heat Guatemala dataset class, follows the same changes described below:
Changes vs original:
  - tabular_dim property added so the datamodule (and model) can read it
    without hardcoding anything.
  - implemented_mod stays {"coords"} because tabular data arrives
    automatically through feat_* CSV columns, not through the modalities dict.
    This is documented explicitly below.
  - Implemented an override for `setup_tessera` in `HeatKrakowDataset`:
    setup_tessera inherited from `BaseDataset` where `self.records.pop(i)`
    was mutating the list mid-iteration, caused it to skip every second
    missing file and eventually trigger a PyTorch DataLoader `IndexError: list index out of range`.
  - Minor: __getitem__ guard tightened (tabular only added when feat_names exist
    and modality logic is cleaner).
"""

import os
from typing import Any, Dict, override

import numpy as np
import torch

from src.data.base_dataset import BaseDataset


class HeatKrakowDataset(BaseDataset):
    """Dataset for the urban heat island use case (Kraków, LST regression).

    CSV layout expected (produced by scripts/make_model_ready_heat_krakow.py):
      - name_loc          : unique location identifier
      - lat, lon          : WGS84 coordinates
      - target_lst        : Land Surface Temperature [°C]
      - feat_*            : tabular features (numeric + one-hot categorical)

    Modality design note
    --------------------
    `implemented_mod = {"coords"}` because in this framework a "modality" refers
    to data loaded from a separate file (e.g. a GeoTIFF or .npy embedding).
    Tabular features live directly in the model-ready CSV and are picked up
    automatically by BaseDataset.get_records() via the `feat_` column prefix.
    They do NOT need to be listed in `modalities`.
    """

    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: Dict[str, Any] | str = "all",
        seed: int = 12345,
        cache_dir: str = None,
        mock: bool = False,
        use_features: bool = True,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            modalities=modalities,
            use_target_data=use_target_data,
            use_aux_data=use_aux_data,
            dataset_name="heat_krakow",
            seed=seed,
            cache_dir=cache_dir,
            implemented_mod={"coords", "tessera"},
            mock=mock,
            use_features=use_features,
        )

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """No files to download / prepare for this dataset."""
        # Set up each requested modality
        for mod in self.modalities.keys():
            if mod == "coords" and len(self.modalities.keys()) == 1:
                return
            elif mod == "tessera":
                self.setup_tessera()
            # elif mod == "aef":
            #     self.setup_aef()
        return

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        sample: Dict[str, Any] = {"eo": {}}

        # --- EO modalities ---
        for modality in self.modalities:
            if modality in ["coords"]:
                sample["eo"][modality] = torch.tensor([row["lat"], row["lon"]])
            elif modality == "tessera":
                sample["eo"][modality] = self.load_tessera(row["tessera_path"])
            elif modality == "aef":
                sample["eo"][modality] = self.load_aef(row["aef_path"])

        # --- Tabular features (always included if present in CSV) ---
        if self.use_features and self.feat_names:
            sample["eo"]["tabular"] = torch.tensor(
                [row[k] for k in self.feat_names], dtype=torch.float32
            )

        # --- Target ---
        if self.use_target_data:
            sample["target"] = torch.tensor(
                [row[k] for k in self.target_names], dtype=torch.float32
            )

        # --- Auxiliary data ---
        if self.use_aux_data:
            sample["aux"] = {}
            for aux_cat, vals in self.use_aux_data.items():
                if aux_cat == "aux":
                    sample["aux"][aux_cat] = torch.tensor(
                        [row[v] for v in vals], dtype=torch.float32
                    )
                else:
                    sample["aux"][aux_cat] = [row[v] for v in vals]

        return sample

    @override
    def setup_tessera(self) -> None:
        """Overridden setup_tessera to fix the list mutation bug."""
        from src.data_preprocessing.tessera_embeds import (
            get_tessera_embeds,
            tessera_from_df,
        )

        print("\n\nSetting up Tessera data (Using Krakow Overridden Method)...\n\n")
        download_missing_tiles = False

        # Check if data is already available
        dst_dir = os.path.join(self.data_dir, "eo/tessera")

        year = self.modalities["tessera"].get(
            "year", KeyError('Missing parameter "year" for Tessera modality')
        )
        size = self.modalities["tessera"].get(
            "size", KeyError('Missing parameter "size" for Tessera modality')
        )

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

        # Download missing rows (if any)
        else:
            from geotessera import GeoTessera

            print("Downloading missing Tessera tiles...")
            print("[Warning]: it may download tessera tiles filled with 0a")

            avail_files = os.listdir(dst_dir)
            gt = None

            # Create a safe list to collect valid records
            valid_records = []

            for rec in self.records:
                fname = os.path.basename(rec["tessera_path"])

                if fname in avail_files:
                    # File exists locally, keep it
                    valid_records.append(rec)
                else:
                    # File is missing
                    if download_missing_tiles:
                        print(f"Retrieving missing Tessera data: {fname}")
                        gt = gt or GeoTessera(cache_dir=self.cache_dir)
                        row = self.df[self.df["name_loc"] == rec["name_loc"]]
                        lon, lat = row.lon.item(), row.lat.item()
                        try:
                            get_tessera_embeds(
                                lon,
                                lat,
                                rec["name_loc"],
                                year=year,
                                save_dir=dst_dir,
                                tile_size=size,
                                tessera_con=gt,
                            )
                            valid_records.append(rec)
                            continue
                        except Exception as e:
                            print(f"Tile for {fname} could not be retrieved. Error: {e}")

                    print(f"No tile found for {fname} thus it will not be used.")

            # Safely swap the filtered list in place
            self.records = valid_records
