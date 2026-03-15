"""Heat Guatemala LST dataset.

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
    """Dataset for the urban heat island use case (Guatemala City, LST regression).

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
            dataset_name="heat_guatemala",
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
            sample["aux"] = {}
            for aux_cat, vals in self.use_aux_data.items():
                if aux_cat == "aux":
                    sample["aux"][aux_cat] = torch.tensor(
                        [row[v] for v in vals], dtype=torch.float32
                    )
                else:
                    sample["aux"][aux_cat] = [row[v] for v in vals]

        return sample
