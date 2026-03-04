import os
from typing import Any, List, override

import pandas as pd
import torch

from src.data.base_caption_builder import (
    BaseCaptionBuilder,
    get_adjective_for_percentage,
)
from src.data.base_dataset import BaseDataset
from src.data_preprocessing.data_utils import (
    process_bioclim_classes,
    process_corine_classes,
)


class ButterflyCaptionBuilder(BaseCaptionBuilder):
    def __init__(
        self, templates_fname: str, concepts_fname: str, data_dir: str, seed: int
    ) -> None:
        super().__init__(templates_fname, concepts_fname, data_dir, seed)

    @override
    def sync_with_dataset(self, dataset: BaseDataset) -> None:
        """Synchronize the dataset with bioclimatic, corine, and human footprint column
        metadata."""
        bioclim_columns = self.get_bioclim_column_keys()
        corine_columns = self.get_corine_column_keys()
        humanfootprint_columns = self.get_humanfootprint_column_keys()
        aux_columns = {**bioclim_columns, **corine_columns, **humanfootprint_columns}
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

    def get_corine_column_keys(self):
        """Returns metadata for corine columns."""
        if not os.path.isfile(os.path.join(self.data_dir, "corine_classes.csv")):
            process_corine_classes(
                os.path.join(self.data_dir, "source/corine_classes.json"),
                os.path.join(self.data_dir, "corine_classes.csv"),
            )
        df = pd.read_csv(os.path.join(self.data_dir, "corine_classes.csv"))

        legend_lowlevel = dict(
            zip(
                df["code"],
                zip(df["category_level_3"], ["%"] * len(df["category_level_3"])),
            )
        )

        legend_midlevel = dict(
            zip(
                df["code"].apply(lambda x: x[:-1]),
                zip(df["category_level_2"], ["%"] * len(df["category_level_2"])),
            )
        )

        legend_highlevel = dict(
            zip(
                df["code"].apply(lambda x: x[:-2]),
                zip(df["category_level_1"], ["%"] * len(df["category_level_1"])),
            )
        )

        combined_legend = {**legend_lowlevel, **legend_midlevel, **legend_highlevel}
        return combined_legend

    def get_bioclim_column_keys(self):
        """Returns metadata for bioclim columns."""
        if not os.path.isfile(os.path.join(self.data_dir, "bioclim_classes.csv")):
            process_bioclim_classes(
                os.path.join(self.data_dir, "source/bioclim_classes.json"),
                os.path.join(self.data_dir, "bioclim_classes.csv"),
            )

        df = pd.read_csv(os.path.join(self.data_dir, "bioclim_classes.csv"))
        df.sort_values(by=["name"], inplace=True)
        return dict(zip(df["name"], zip(df["description"], df["units"])))

    def get_humanfootprint_column_keys(self):
        """Returns metadata for human footprint columns."""
        dict_hf = {
            "aux_maxdist_road": ("farthest distance to road", "m"),
            "aux_meandist_road": ("mean distance to road", "m"),
            "aux_pop_density": ("population density", "people/km²"),
            "aux_total_population": ("total population", "people"),
        }
        return dict_hf

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
            value = aux[idx].item()

            formatted_desc = values_dict["description"].lower() or ""
            units = values_dict["units"]
            value = value * 100 if units == "%" else value

            if "corine" in token:
                if convert_corine_perc:
                    adjective = get_adjective_for_percentage(value)
                    formatted_desc = f"{adjective} {formatted_desc}"
                else:
                    formatted_desc = formatted_desc + f' ({round(value)} {units if units else ""})'
            else:
                formatted_desc = formatted_desc + f' of {round(value)} {units if units else ""}'
            replacements[init_token] = formatted_desc

        template = self._fill(template, replacements)
        return template


if __name__ == "__main__":
    _ = ButterflyCaptionBuilder(None, None)
