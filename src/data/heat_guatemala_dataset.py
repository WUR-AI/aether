"""
Heat Guatemala LST dataset.

Location: src/data/heat_guatemala_dataset.py

Changes vs original:
  - tabular_dim property added so the datamodule (and model) can read it
    without hardcoding anything.
  - implemented_mod stays {"coords"} because tabular data arrives
    automatically through feat_* CSV columns, not through the modalities dict.
    This is documented explicitly below.
  - Minor: __getitem__ guard tightened (tabular only added when feat_names exist
    and modality logic is cleaner).
"""

from typing import Any, Dict, override

import torch

from src.data.base_dataset import BaseDataset


class HeatGuatemalaDataset(BaseDataset):
    """
    Dataset for the urban heat island use case (Guatemala City, LST regression).

    CSV layout expected (produced by scripts/make_model_ready_heat_guatemala.py):
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
        use_aux_data: bool = False,
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
            dataset_name="heat_guatemala",
            seed=seed,
            cache_dir=cache_dir,
            implemented_mod={"coords"},
            mock=mock,
            use_features=use_features,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tabular_dim(self) -> int:
        """Number of tabular features (feat_* columns). 0 if none."""
        return len(self.feat_names)

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """No files to download / prepare for this dataset."""
        return

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        sample: Dict[str, Any] = {"eo": {}}

        # --- EO modalities ---
        for modality in self.modalities:
            if modality == "coords":
                sample["eo"]["coords"] = torch.tensor(
                    [row["lat"], row["lon"]], dtype=torch.float32
                )

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
            sample["aux"] = [row[k] for k in self.aux_names]

        return sample
