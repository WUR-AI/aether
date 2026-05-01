import json
import os
import re
from typing import Any, Dict, List, override

import pandas as pd
import torch

from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.base_dataset import BaseDataset


class YieldAfricaCaptionBuilder(BaseCaptionBuilder):
    """Caption builder for the crop yield Africa dataset.

    Resolves aux_*_cl column tokens and group top-N tokens using ordinal rank metadata loaded from
    per-group CSV files. The ordinal_rank() method is public so the explainability framework can
    compare encoded aux values against concept theta_k thresholds without re-loading metadata.
    """

    GROUPS = ["soil", "climate", "terrain", "landcover", "ndvi", "agera5"]

    CSV_FILES = {
        "soil": "soil_classes.csv",
        "climate": "climate_classes.csv",
        "terrain": "terrain_classes.csv",
        "landcover": "landcover_classes.csv",
        "ndvi": "ndvi_classes.csv",
        "agera5": "agera5_classes.csv",
        "derived": "derived_classes.csv",
        "target": "target_classes.csv",
    }

    NOMINAL_COLS = {"aux_aspect_cl", "aux_landform_cl", "aux_glad_cl"}

    def __init__(
        self, templates_fname: str, concepts_fname: str, data_dir: str, seed: int
    ) -> None:
        super().__init__(templates_fname, concepts_fname, data_dir, seed)
        self.group_columns: Dict[str, List[str]] = {g: [] for g in self.GROUPS}

    @override
    def sync_with_dataset(self, dataset: BaseDataset) -> None:
        """Load group metadata CSVs and index all aux columns present in the dataset."""
        all_metadata: Dict[str, Any] = {}
        for group_key, csv_fname in self.CSV_FILES.items():
            for col, meta in self._load_group_from_csv(group_key, csv_fname).items():
                all_metadata[col] = {**meta, "group": group_key}

        self.column_to_metadata_map = {"aux": {}}
        for i, col in enumerate(dataset.use_aux_data.get("aux", [])):
            meta = all_metadata.get(col, {})
            self.column_to_metadata_map["aux"][col] = {
                "id": i,
                "description": meta.get("description", ""),
                "units": meta.get("units", ""),
                "ordinal_map": meta.get("ordinal_map"),
                "nominal_labels": meta.get("nominal_labels", {}),
            }
            group = meta.get("group")
            if group in self.GROUPS:
                self.group_columns[group].append(col)

        self.sync_concepts()

    def _load_group_from_csv(self, group_key: str, csv_fname: str) -> Dict[str, Any]:
        """Load column metadata from a group CSV and return a col → metadata dict."""
        fpath = os.path.join(self.data_dir, csv_fname)
        if not os.path.isfile(fpath):
            return {}
        df = pd.read_csv(fpath)
        result = {}
        for _, row in df.iterrows():
            ordinal_map = None
            raw = row.get("ordinal_map", "")
            if pd.notna(raw) and str(raw).strip():
                ordinal_map = {int(k): v for k, v in json.loads(str(raw)).items()}

            nominal_labels: Dict[int, str] = {}
            raw = row.get("nominal_labels", "")
            if pd.notna(raw) and str(raw).strip():
                nominal_labels = {int(k): v for k, v in json.loads(str(raw)).items()}

            result[row["col"]] = {
                "description": str(row.get("description", "") or ""),
                "units": str(row.get("units", "") or ""),
                "ordinal_map": ordinal_map,
                "nominal_labels": nominal_labels,
            }
        return result

    def ordinal_rank(self, col: str, encoded_value: int) -> int:
        """Return ordinal rank (0–4) for an encoded column value.

        Returns -1 for NoData entries. Raises ValueError for nominal columns (which have no ordinal
        map).
        """
        meta = self.column_to_metadata_map["aux"].get(col)
        if meta is None:
            raise KeyError(f"Column '{col}' not found in metadata map")
        ordinal_map = meta.get("ordinal_map")
        if ordinal_map is None:
            raise ValueError(f"Column '{col}' is nominal and has no ordinal rank")
        return ordinal_map.get(int(encoded_value), -1)

    def _class_label(self, col: str, encoded_value: int) -> str:
        """Return the full pre-encoding label string for an encoded column value."""
        meta = self.column_to_metadata_map["aux"][col]
        return meta["nominal_labels"].get(int(encoded_value), "")

    def _get_top_n_for_group(self, group: str, aux: torch.Tensor, n: int) -> str:
        """Return the class label of the N-th most extreme column in the group.

        Extremity is abs(ordinal_rank - 2). Nominal columns and NoData entries
        (ordinal_rank == -1) are excluded. Returns an empty string when fewer
        than N valid columns are available.
        """
        ranked: List[tuple[int, str]] = []
        for col in self.group_columns.get(group, []):
            meta = self.column_to_metadata_map["aux"][col]
            if meta.get("ordinal_map") is None:
                continue
            encoded_value = int(aux[meta["id"]].item())
            rank = meta["ordinal_map"].get(encoded_value, -1)
            if rank == -1:
                continue
            label = meta["nominal_labels"].get(encoded_value, "")
            ranked.append((abs(rank - 2), label))
        ranked.sort(key=lambda x: x[0], reverse=True)
        if n <= len(ranked):
            return ranked[n - 1][1].lower()
        return ""

    @override
    def _build_from_template(
        self, template_idx: int, aux: torch.Tensor, top: List[str] | None = None
    ) -> str:
        """Build caption text from template and a row of encoded auxiliary values."""
        template = self.templates[template_idx]
        tokens = self.tokens_in_template[template_idx]
        fillers: Dict[str, str] = {}
        for token in tokens:
            top_match = re.match(r"aux_(.+)_top_(\d+)$", token)
            if top_match:
                group = top_match.group(1)
                n = int(top_match.group(2))
                fillers[token] = self._get_top_n_for_group(group, aux, n)
            else:
                meta = self.column_to_metadata_map["aux"].get(token)
                if meta is None:
                    raise KeyError(
                        f"Token '{token}' not found in column_to_metadata_map. "
                        "Check that the column is present in the dataset aux columns."
                    )
                encoded_value = int(aux[meta["id"]].item())
                fillers[token] = self._class_label(token, encoded_value).lower()
        return self._fill(template, fillers)
