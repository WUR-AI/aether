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

from typing import Dict, List, Tuple, override

import torch

from src.data.base_caption_builder import BaseCaptionBuilder
from src.data.base_dataset import BaseDataset


class HeatGuatemalaCaptionBuilder(BaseCaptionBuilder):
    # Which numeric aux column each expert-legend label describes. Templates are
    # written in terms of the label columns, but the soft contrastive loss scores
    # location similarity on the numeric values, so it needs the numeric id of
    # every variable a caption actually mentions (see return_aux_ids).
    # Labels with no numeric counterpart (built-up height/density and the purely
    # categorical land-use, block-type and interzone fields) are absent on
    # purpose: they contribute no id. aux_lst is likewise absent because the
    # target must never enter a training caption.
    LABEL_TO_NUMERIC_AUX: Dict[str, str] = {
        "aux_ndvi_label": "aux_ndvi_mean",
        "aux_ndwi_label": "aux_ndwi_mean",
        "aux_forest_label": "aux_forest_cover_perc",
        "aux_age_label": "aux_builtup_age_years",
        "aux_slope_label": "aux_slope_perc",
        "aux_socio_label": "aux_socioeconomic",
    }

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

        # label token -> numeric aux id, for return_aux_ids. Built from whichever
        # numeric columns this run actually configured, so dropping a column from
        # use_aux_data.aux just drops it here too.
        self.token_to_aux_id: Dict[str, int] = {}
        for label_col, numeric_col in self.LABEL_TO_NUMERIC_AUX.items():
            if label_col in self.top_index and numeric_col in self.column_to_metadata_map["aux"]:
                self.token_to_aux_id[label_col] = self.column_to_metadata_map["aux"][numeric_col][
                    "id"
                ]

        # wires concept["id"] from the numeric aux map (raises if a concept col is missing)
        self.sync_concepts()

    @override
    def _build_from_template(
        self, template_idx: int, aux: torch.Tensor, top: List[str] | None = None
    ) -> str | Tuple[str, List[int]]:
        template = self.templates[template_idx]
        fillers: Dict[str, str] = {}
        ids: List[int] = []
        for token in self.tokens_in_template[template_idx]:
            if token in self.top_index and top is not None:
                fillers[token] = str(top[self.top_index[token]])
                if token in self.token_to_aux_id:
                    ids.append(self.token_to_aux_id[token])
            elif token in self.column_to_metadata_map["aux"]:
                # numeric fallback (default templates don't use numeric tokens)
                aux_id = self.column_to_metadata_map["aux"][token]["id"]
                fillers[token] = f"{aux[aux_id].item():.2f}"
                ids.append(aux_id)
            else:
                raise KeyError(
                    f"Token '{token}' is neither a label ('top') nor a numeric ('aux') "
                    "column in the dataset. Check the template and use_aux_data config."
                )
        filled_template = self._fill(template, fillers)
        if self.return_aux_ids:
            return filled_template, ids
        return filled_template
