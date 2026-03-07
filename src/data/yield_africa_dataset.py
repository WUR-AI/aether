"""Yield Africa dataset.

Location: src/data/yield_africa_dataset.py

Crop yield regression use case for East/Southern Africa.
Tabular features (soil, climate, etc.) live in the model-ready CSV as feat_*
columns and are picked up automatically by BaseDataset.get_records().
They do NOT need to be listed in `modalities`.
"""

import logging
from typing import Any, Dict, List, override

import torch

from src.data.base_dataset import BaseDataset

log = logging.getLogger(__name__)


class YieldAfricaDataset(BaseDataset):
    """Dataset for the crop yield regression use case (East/Southern Africa).

    CSV layout expected:
      - name_loc        : unique location identifier
      - lat, lon        : WGS84 coordinates
      - target_*        : crop yield target(s) [t/ha]
      - feat_*          : tabular features (soil properties, climate indices, etc.)
      - aux_*           : auxiliary data columns (optional)
      - country, year   : metadata columns used for optional filtering

    Modality design note
    --------------------
    `implemented_mod = {"coords"}` because tabular features live directly in
    the model-ready CSV and are picked up via the `feat_` column prefix.
    They do NOT need to be listed in `modalities`.
    """

    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: Dict[str, Any] | str = None,
        seed: int = 12345,
        cache_dir: str = None,
        mock: bool = False,
        use_features: bool = True,
        countries: List[str] | None = None,
        years: List[int] | None = None,
        exclude_countries: List[str] | None = None,
        exclude_years: List[int] | None = None,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            modalities=modalities,
            use_target_data=use_target_data,
            use_aux_data=use_aux_data,
            dataset_name="yield_africa",
            seed=seed,
            cache_dir=cache_dir,
            implemented_mod={"coords"},
            mock=mock,
            use_features=use_features,
        )

        # Apply country/year filters to self.df and rebuild records if needed.
        # BaseDataset.__init__ has already loaded the CSV; filtering here avoids
        # touching BaseDataset and keeps the logic use-case specific.
        n_before = len(self.df)
        if countries is not None and "country" in self.df.columns:
            self.df = self.df[self.df["country"].isin(countries)].reset_index(drop=True)
        if years is not None and "year" in self.df.columns:
            self.df = self.df[self.df["year"].isin(years)].reset_index(drop=True)
        if exclude_countries is not None and "country" in self.df.columns:
            self.df = self.df[~self.df["country"].isin(exclude_countries)].reset_index(drop=True)
        if exclude_years is not None and "year" in self.df.columns:
            self.df = self.df[~self.df["year"].isin(exclude_years)].reset_index(drop=True)

        n_after = len(self.df)
        if n_after != n_before:
            log.info(f"Country/year filter: {n_before} → {n_after} records ({n_before - n_after} excluded)")
            self.records = self.get_records()

    def setup(self) -> None:
        """No files to download or prepare for this dataset."""
        return

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        sample: Dict[str, Any] = {"eo": {}}

        for modality in self.modalities:
            if modality == "coords":
                sample["eo"]["coords"] = torch.tensor(
                    [row["lat"], row["lon"]], dtype=torch.float32
                )

        if self.use_features and self.feat_names:
            sample["eo"]["tabular"] = torch.tensor(
                [row[k] for k in self.feat_names], dtype=torch.float32
            )

        if self.use_target_data:
            sample["target"] = torch.tensor(
                [row[k] for k in self.target_names], dtype=torch.float32
            )

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
