"""Caption builder for the Guatemala City urban-heat (LST) use case.

Two aux categories are used (configured in the data yaml under `use_aux_data`):

  * ``aux``  – numeric raw columns (NDVI, NDWI, slope, built-up age, LST, ...).
               These feed the concept retrieval evaluation: each concept's
               ``theta_k`` is compared directly against these raw values.
  * ``top``  – expert-legend *label* columns (e.g. ``aux_ndvi_label`` =
               "high vegetation greenness", ``aux_density_label`` =
               "very dense urban", ``aux_landuse`` = "discontinuous urban").
               These fill the ``<...>`` tokens in the location-caption templates,
               so the training text uses the authoritative expert wording from
               ``Heat_Guatemala.csv``.

This mirrors the continuous (butterfly) paradigm for concepts, while taking the
caption *words* straight from the legend rather than re-deriving them — the two
are produced from the same block by build_aux_from_original.py, so they stay
consistent. The LST label is deliberately NOT used in any template (the target
must not leak into the training captions); LST appears only as a concept.
"""

from typing import Dict, List, override

import torch

from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.base_dataset import BaseDataset


class HeatGuatemalaCaptionBuilder(BaseCaptionBuilder):
    @override
    def sync_with_dataset(self, dataset: BaseDataset) -> None:
        """Index numeric aux columns (for concepts) and label columns (for text)."""
        # numeric aux -> id (position in the 'aux' tensor); used by sync_concepts()
        self.column_to_metadata_map = {"aux": {}}
        for i, col in enumerate(dataset.use_aux_data.get("aux", [])):
            self.column_to_metadata_map["aux"][col] = {"id": i}

        # label aux -> position in the per-row 'top' list; used to fill template tokens
        self.top_index: Dict[str, int] = {
            col: i for i, col in enumerate(dataset.use_aux_data.get("top", []))
        }

        # wires concept["id"] from the numeric aux map (raises if a concept col is missing)
        self.sync_concepts()

    @override
    def _build_from_template(
        self, template_idx: int, aux: torch.Tensor, top: List[str] | None = None
    ) -> str:
        template = self.templates[template_idx]
        fillers: Dict[str, str] = {}
        for token in self.tokens_in_template[template_idx]:
            if token in self.top_index and top is not None:
                fillers[token] = str(top[self.top_index[token]])
            elif token in self.column_to_metadata_map["aux"]:
                # numeric fallback (default templates don't use numeric tokens)
                fillers[token] = f"{aux[self.column_to_metadata_map['aux'][token]['id']].item():.2f}"
            else:
                raise KeyError(
                    f"Token '{token}' is neither a label ('top') nor a numeric ('aux') "
                    "column in the dataset. Check the template and use_aux_data config."
                )
        return self._fill(template, fillers)
