import os
from typing import List, override

import pandas as pd
import torch

from src.data.base_caption_builder import (
    BaseCaptionBuilder,
    sample_adjective_for_percentage,
)
from src.data.base_dataset import BaseDataset


class SatBirdCaptionBuilder(BaseCaptionBuilder):
    def __init__(
        self,
        templates_fname: str,
        concepts_fname: str,
        data_dir: str,
        seed: int,
        n_captions_for_validation: int | str = "all",
        n_captions_for_train: int = 1,
        return_aux_ids: bool = False,
    ) -> None:
        super().__init__(
            templates_fname=templates_fname,
            concepts_fname=concepts_fname,
            data_dir=data_dir,
            seed=seed,
            n_captions_for_train=n_captions_for_train,
            n_captions_for_validation=n_captions_for_validation,
            return_aux_ids=return_aux_ids,
        )

    @override
    def sync_with_dataset(self, dataset: BaseDataset) -> None:
        """Synchronize the dataset with bioclimatic, corine, and human footprint column
        metadata."""
        bioclim_columns = self.get_bioclim_column_keys()
        corine_columns = self.get_dynamic_world_column_keys()
        soil_columns = self.get_soil_grid_keys()
        aux_columns = {**bioclim_columns, **corine_columns, **soil_columns}

        self.column_to_metadata_map = {k: {} for k in dataset.use_aux_data.keys()}

        for aux_cat, cols in dataset.use_aux_data.items():
            for i, c in enumerate(cols):
                if "top" in aux_cat:
                    description, units = None, None
                else:
                    description, units = aux_columns.get(c) or (None, None)

                self.column_to_metadata_map[aux_cat][c] = {
                    "id": i,
                    "description": description,
                    "units": units,
                }
        self.sync_concepts()

    def get_dynamic_world_column_keys(self):
        """Returns metadata for corine columns."""
        assert os.path.exists(
            os.path.join(self.data_dir, "dynamicworld_classes.csv")
        ), FileNotFoundError()
        df = pd.read_csv(os.path.join(self.data_dir, "dynamicworld_classes.csv"))
        return dict(zip(df["code"], zip(df["category"], ["%"] * len(df["category"]))))

    def get_bioclim_column_keys(self):
        """Returns metadata for bioclim columns."""
        assert os.path.exists(
            os.path.join(self.data_dir, "bioclim_classes.csv")
        ), FileNotFoundError()
        df = pd.read_csv(os.path.join(self.data_dir, "bioclim_classes.csv"))
        return dict(zip(df["name"], zip(df["description"], df["units"])))

    def get_soil_grid_keys(self):
        """Returns metadata for soil grid columns."""
        assert os.path.exists(
            os.path.join(self.data_dir, "soilgrid_classes.csv")
        ), FileNotFoundError()

        df = pd.read_csv(os.path.join(self.data_dir, "soilgrid_classes.csv"))
        return dict(zip(df["name"], zip(df["description"], df["units"])))

    def _build_from_template(
        self,
        template_idx: int,
        aux: torch.Tensor,
        top: List[str] | None = None,
        convert_corine_perc: bool = True,
    ) -> str:
        """Create caption from template and row of auxiliary data."""
        template = self.templates[template_idx]
        tokens = self.tokens_in_template[template_idx]
        replacements = {}
        if self.return_aux_ids:
            ids = []

        for token in tokens:
            init_token = token
            if "top" in token:
                idx = self.column_to_metadata_map["top"][token]["id"]
                token = f"aux_{top[idx]}"
            try:
                values_dict = self.column_to_metadata_map["aux"][token]
            except KeyError:
                raise KeyError(
                    f"Token {token} not found in column_to_metadata_map {self.column_to_metadata_map}. Check if the token in the template matches the column names in the dataset."
                )

            idx = values_dict["id"]
            if self.return_aux_ids:
                ids.append(idx)
            value = aux[idx].item()

            formatted_desc = values_dict["description"].lower() or ""
            units = values_dict["units"]
            value = value * 100 if units == "%" else value

            if "corine" in token:
                if convert_corine_perc:
                    adjective = sample_adjective_for_percentage(value)
                    formatted_desc = f"{adjective} {formatted_desc}"
                else:
                    formatted_desc = formatted_desc + f' ({round(value)} {units if units else ""})'
            else:
                formatted_desc = formatted_desc + f' of {round(value)} {units if units else ""}'
            replacements[init_token] = formatted_desc

        filled_template = self._fill(template, replacements)
        if self.return_aux_ids:
            return filled_template, ids
        return filled_template
