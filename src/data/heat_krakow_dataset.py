"""Heat Kraków LST dataset.

Location: src/data/heat_krakow_dataset.py
"""

import logging
import os
from typing import Any, Dict, override

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import src.utils.experiment_tracking as et
from src.data.base_dataset import BaseDataset

log = logging.getLogger(__name__)

# Compatibility Patche for experiment_tracking.py)
_orig_parse_data_name = et.parse_data_name


def _patched_parse_data_name(run=None, cfg=None):
    if cfg is not None and isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return _orig_parse_data_name(run=run, cfg=cfg)


et.parse_data_name = _patched_parse_data_name

_orig_compose_experiment_name = et.compose_experiment_name

# Patch torch.nn.Module to support base_model's loss_fn.setup() call safely
if not hasattr(nn.Module, "setup"):
    nn.Module.setup = lambda self, *args, **kwargs: None


class HeatKrakowDataset(BaseDataset):
    """Dataset for the urban heat island use case (Kraków, LST regression)."""

    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: Any = None,
        use_features: bool = True,
        seed: int = 12345,
        cache_dir: str = None,
        mock: bool = False,
        dtype: str = "float32",
        return_name_loc: bool = False,
        csv_name: str = "model_ready_heat_krakow.csv",
        **kwargs,
    ) -> None:
        if mock:
            csv_name = None
        else:
            csv_name = "model_ready_heat_krakow.csv"

        # Trim directory path if base_data_dir ends with 'heat_krakow'
        base_data_dir = data_dir.rstrip("/\\")
        if base_data_dir.endswith("heat_krakow"):
            base_data_dir = os.path.dirname(base_data_dir)

        super().__init__(
            data_dir=base_data_dir,
            modalities=modalities,
            use_target_data=use_target_data,
            use_aux_data=use_aux_data,
            dataset_name="heat_krakow",
            seed=seed,
            cache_dir=cache_dir,
            implemented_mod={"tessera", "coords"},
            mock=mock,
            dtype=dtype,
            use_features=use_features,
            return_name_loc=return_name_loc,
            csv_name=csv_name,
        )

    def _setup(self):
        """Sets up active modalities on self.df prior to record generation."""
        for mod in self.modalities.keys():
            if mod == "coords" and len(self.modalities.keys()) == 1:
                return
            elif mod == "tessera":
                super().setup_tessera()

    @override
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        formatted_row: Dict[str, Any] = {"eo": {}}

        # Base tensor dtype fallback
        if isinstance(self.dtype, str):
            tensor_dtype = getattr(torch, self.dtype)
        elif isinstance(self.dtype, torch.dtype):
            tensor_dtype = self.dtype
        else:
            tensor_dtype = torch.float32

        # Process requested modalities
        for modality in self.modalities:
            if modality == "coords":
                mod_cfg = self.modalities["coords"]
                if isinstance(mod_cfg, dict) and "dtype" in mod_cfg:
                    dtype_val = mod_cfg["dtype"]
                    mod_dtype = (
                        getattr(torch, dtype_val) if isinstance(dtype_val, str) else dtype_val
                    )
                else:
                    mod_dtype = tensor_dtype

                formatted_row["eo"]["coords"] = torch.tensor(
                    [row["lat"], row["lon"]],
                    dtype=mod_dtype,
                )
            elif modality == "tessera":
                formatted_row["eo"]["tessera"] = self.load_tessera(row["tessera_path"])

        # Target data
        if self.use_target_data:
            target_names = getattr(self, "target_names", ["target_lst"])
            target_vals = [row[k] for k in target_names if k in row]
            if not target_vals and "target_lst" in row:
                target_vals = [row["target_lst"]]
            formatted_row["target"] = torch.tensor(target_vals, dtype=torch.float32)

        # Auxiliary data
        if self.use_aux_data:
            formatted_row["aux"] = {}
            for aux_cat, vals in self.use_aux_data.items():
                if aux_cat == "aux":
                    formatted_row["aux"][aux_cat] = torch.tensor(
                        [row[v] for v in vals], dtype=tensor_dtype
                    )
                else:
                    formatted_row["aux"][aux_cat] = [row[v] for v in vals]

        # Tabular features
        if self.use_features and getattr(self, "feat_names", None):
            raw = torch.tensor([row[k] for k in self.feat_names], dtype=torch.float32)
            if (
                getattr(self, "_feat_mean", None) is not None
                and getattr(self, "_feat_std", None) is not None
            ):
                formatted_row["eo"]["tabular"] = (raw - self._feat_mean) / self._feat_std
            else:
                formatted_row["eo"]["tabular"] = raw

        if self.return_name_loc:
            formatted_row["name"] = row.get("name_loc") or row.get("name", f"record_{idx}")

        return formatted_row
