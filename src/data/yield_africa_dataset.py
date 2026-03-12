"""Yield Africa dataset.

Location: src/data/yield_africa_dataset.py

Crop yield regression use case for East/Southern Africa.
Tabular features (soil, climate, etc.) live in the model-ready CSV as feat_*
columns and are picked up automatically by BaseDataset.get_records().
They do NOT need to be listed in `modalities`.
"""

import logging
import os
from typing import Any, Dict, List, override

import pandas as pd
import torch

from src.data.base_dataset import BaseDataset

# Number of channels in a TESSERA embedding tile (fixed by the geotessera model).
_TESSERA_CHANNELS = 128

log = logging.getLogger(__name__)

# Fixed ordered list of all countries in the full dataset.
# Used to produce a consistent one-hot encoding regardless of which
# countries are present after filtering.
_ALL_COUNTRIES = ["BF", "BUR", "ETH", "KEN", "MAL", "RWA", "TAN", "ZAM"]


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

    In addition to the CSV feat_* columns, `year` and one-hot `country`
    encodings are injected as `feat_year` and `feat_country_{CODE}` so that
    the model can condition on inter-annual and cross-country variation.
    The one-hot set always covers `_ALL_COUNTRIES` (8 countries) so that
    `tabular_dim` is stable regardless of the country filter applied.
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
            implemented_mod={"coords", "tessera"},
            mock=mock,
            use_features=use_features,
        )

        # Inject year and country one-hot columns as feat_* so that
        # get_records() picks them up automatically.  Build all new columns in
        # one concat to avoid pandas PerformanceWarning from repeated assignment.
        if use_features and "year" in self.df.columns and "country" in self.df.columns:
            # Normalise feat_year to the same scale as the pre-scaled CSV feat_* columns
            # (roughly zero-mean, unit-std) so it doesn't dominate LayerNorm.
            _YEAR_MEAN = 2018.0
            _YEAR_STD = 2.0
            new_cols: Dict[str, Any] = {
                "feat_year": (self.df["year"].astype(float) - _YEAR_MEAN) / _YEAR_STD
            }
            for code in _ALL_COUNTRIES:
                new_cols[f"feat_country_{code}"] = (self.df["country"] == code).astype(float)
            self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

        # Apply country/year filters to self.df and rebuild records.
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
            log.info(
                f"Country/year filter: {n_before} → {n_after} records ({n_before - n_after} excluded)"
            )

        # get_records() mutates self.use_aux_data in place (replacing pattern
        # dicts with resolved column-name lists), so reset it from the original
        # parameter before calling it a second time.
        if use_aux_data is None or use_aux_data == "all":
            self.use_aux_data = {
                "aux": {"pattern": "^aux_(?!.*top).*"},
                "top": {"pattern": "^aux_.*top.*"},
            }
        elif isinstance(use_aux_data, dict):
            self.use_aux_data = use_aux_data
        else:
            self.use_aux_data = None

        # Always rebuild so feat_year / feat_country_* are reflected in
        # self.feat_names and self.tabular_dim.
        self.records = self.get_records()

    def setup(self) -> None:
        """Check for requested modality data; warn if TESSERA tiles are absent.

        Unlike other datasets, TESSERA tiles for yield_africa vary per record
        year and must be pre-fetched with the preprocessing script:
            python src/data_preprocessing/yield_africa_tessera_preprocess.py

        setup_tessera() is intentionally not called here because it uses a
        single fixed year for bulk download, which is incompatible with the
        multi-year nature of this dataset.
        """
        if "tessera" in self.modalities:
            tessera_dir = os.path.join(self.data_dir, "eo", "tessera")
            if not os.path.exists(tessera_dir) or len(os.listdir(tessera_dir)) == 0:
                log.warning(
                    "TESSERA tiles not found at %s. "
                    "Run src/data_preprocessing/yield_africa_tessera_preprocess.py "
                    "to pre-fetch tiles. Missing tiles will fall back to zero tensors.",
                    tessera_dir,
                )

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        sample: Dict[str, Any] = {"eo": {}}

        for modality in self.modalities:
            if modality == "coords":
                sample["eo"]["coords"] = torch.tensor(
                    [row["lat"], row["lon"]], dtype=torch.float32
                )
            elif modality == "tessera":
                tile_path = row["tessera_path"]
                if os.path.exists(tile_path):
                    sample["eo"]["tessera"] = self.load_npy(tile_path)
                else:
                    size = self.modalities["tessera"].get("size", 9)
                    log.debug("TESSERA tile missing: %s — using zero fallback.", tile_path)
                    sample["eo"]["tessera"] = torch.zeros(
                        _TESSERA_CHANNELS, size, size, dtype=torch.float32
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
