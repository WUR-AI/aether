"""Caption builder for the Kraków urban heat island (LST) dataset.

Location: src/data/heat_krakow_caption_builder.py
"""

import logging
import os
from typing import Dict, List, Tuple, Union, override

import torch

from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.base_dataset import BaseDataset

log = logging.getLogger(__name__)


class HeatKrakowCaptionBuilder(BaseCaptionBuilder):
    """Caption builder for Kraków dataset supporting flexible column prefixes (feat_ / aux_)."""

    def __init__(
        self,
        templates_fname: str,
        concepts_fname: str,
        data_dir: str,
        seed: int = 12345,
        n_captions_for_validation: Union[int, str] = "all",
        return_aux_ids: bool = False,
    ) -> None:
        resolved_data_dir = data_dir
        if not os.path.exists(os.path.join(data_dir, "location_caption_templates")):
            hk_dir = os.path.join(data_dir, "heat_krakow")
            if os.path.exists(os.path.join(hk_dir, "location_caption_templates")):
                resolved_data_dir = hk_dir

        super().__init__(
            templates_fname=templates_fname,
            concepts_fname=concepts_fname,
            data_dir=resolved_data_dir,
            seed=seed,
            n_captions_for_validation=n_captions_for_validation,
            return_aux_ids=return_aux_ids,
        )
        self.top_alias_map: Dict[str, int] = {}
        self.aux_alias_map: Dict[str, int] = {}
        self.top_raw_cols: List[str] = []

    @staticmethod
    def _normalize(name: str) -> str:
        """Strips prefixes (feat_, aux_) and suffixes (_label) for universal matching."""
        s = name
        if s.startswith("feat_"):
            s = s[5:]
        elif s.startswith("aux_"):
            s = s[4:]
        if s.endswith("_label"):
            s = s[:-6]
        return s

    @override
    def sync_with_dataset(self, dataset: BaseDataset) -> None:
        """Builds column mapping supporting feat_*, aux_*, and *_label variations."""
        self.column_to_metadata_map = {"aux": {}}

        # 1. Register numeric feature/aux columns
        aux_cols = dataset.use_aux_data.get("aux", []) if dataset.use_aux_data else []
        for i, col in enumerate(aux_cols):
            meta = {"id": i}
            norm_col = self._normalize(col)

            # Map exact name and prefix-stripped name
            self.column_to_metadata_map["aux"][col] = meta
            self.column_to_metadata_map["aux"][norm_col] = meta

            self.aux_alias_map[col] = i
            self.aux_alias_map[norm_col] = i

        # 2. Register text label columns ('top')
        top_cols = dataset.use_aux_data.get("top", []) if dataset.use_aux_data else []
        self.top_raw_cols = top_cols
        for i, col in enumerate(top_cols):
            self.top_alias_map[col] = i
            self.top_alias_map[self._normalize(col)] = i

        # 3. Synchronize concepts with numeric aux IDs
        self.sync_concepts()

    def _resolve_top_token(self, token: str) -> Union[int, None]:
        """Resolves token to index in 'top' label list."""
        if token in self.top_alias_map:
            return self.top_alias_map[token]
        return self.top_alias_map.get(self._normalize(token))

    def _resolve_aux_token(self, token: str) -> Union[int, None]:
        """Resolves token to index in 'aux' numeric tensor."""
        if token in self.aux_alias_map:
            return self.aux_alias_map[token]
        return self.aux_alias_map.get(self._normalize(token))

    @override
    def _build_from_template(
        self, template_idx: int, aux: torch.Tensor, top: List[str] | None = None
    ) -> Union[str, Tuple[str, List[int]]]:
        """Fills template tokens using text labels or formatted numeric values."""
        template = self.templates[template_idx]
        fillers: Dict[str, str] = {}
        used_aux_ids: List[int] = []

        for token in self.tokens_in_template[template_idx]:
            top_idx = self._resolve_top_token(token)
            aux_idx = self._resolve_aux_token(token)

            if top_idx is not None and top is not None:
                fillers[token] = str(top[top_idx])

                # Match label back to its numeric feature ID for evaluation
                raw_label_col = self.top_raw_cols[top_idx]
                norm_name = self._normalize(raw_label_col)
                for aux_col_name, meta in self.column_to_metadata_map["aux"].items():
                    if self._normalize(aux_col_name) == norm_name:
                        used_aux_ids.append(meta["id"])
                        break

            elif aux_idx is not None:
                val = aux[aux_idx].item()
                fillers[token] = f"{val:.2f}"
                used_aux_ids.append(aux_idx)
            else:
                raise KeyError(
                    f"Token '<{token}>' could not be resolved. "
                    f"Available top columns: {self.top_raw_cols}"
                )

        filled_caption = self._fill(template, fillers)

        if self.return_aux_ids:
            return filled_caption, used_aux_ids
        return filled_caption
