import os
from typing import override, Any, List
import math

import pandas as pd

from src.data.base_dataset import BaseDataset
from src.data.base_caption_builder import BaseCaptionBuilder, get_adjective_for_percentage
from src.data_preprocessing.data_utils import process_corine_classes, process_bioclim_classes


class ButterflyCaptionBuilder(BaseCaptionBuilder):
    def __init__(self, templates_path: str, data_dir: str):
        super().__init__(templates_path, data_dir)

    @override
    def sync_with_dataset(self, dataset: BaseDataset) -> None:

        bioclim_columns = self.get_bioclim_column_keys()
        corine_columns = self.get_corine_column_keys()

        self.column_to_metadata_map = {}

        for id, key in enumerate(dataset.aux_names):
            if key.startswith('aux_corine_frac_top'):  # to avoid assert statement
                description, units = None, None  
            else:
                description, units = bioclim_columns.get(key) or corine_columns.get(key) or (None, None)
                assert description is not None, f'Key {key} not found in bioclim or corine columns'
            self.column_to_metadata_map[key] = {"id": id, "description": description, "units": units}

    def get_corine_column_keys(self):

        if not os.path.isfile(os.path.join(self.data_dir, "caption_templates/corine_classes.csv")):
            process_corine_classes(
                os.path.join(self.data_dir, "source/corine_classes.json"),
                os.path.join(self.data_dir, "caption_templates/corine_classes.csv"),
            )
        df = pd.read_csv(os.path.join(self.data_dir, "caption_templates/corine_classes.csv"))

        return dict(zip(df["code"], zip(df["category_level_3"], ["%"] * len(df["category_level_3"]))))

    def get_bioclim_column_keys(self):
        if not os.path.isfile(os.path.join(self.data_dir, "caption_templates/bioclim_classes.csv")):
            process_bioclim_classes(
                os.path.join(self.data_dir, "source/bioclim_classes.json"),
                os.path.join(self.data_dir, "caption_templates/bioclim_classes.csv"),
            )

        df = pd.read_csv(os.path.join(self.data_dir, "caption_templates/bioclim_classes.csv"))
        df.sort_values(by=["name"], inplace=True)
        return dict(zip(df["name"], zip(df["description"], df["units"])))

    def _build_from_template(self, template_idx: int, row: List[Any],
                             convert_corine_perc: bool = False) -> str:
        '''Create caption from template and row of auxiliary data.'''
        template = self.templates[template_idx]
        tokens = self.template_tokens[template_idx]
        replacements = {}
        for token in tokens:
            if token.startswith('aux_corine_frac_top_'):
                values_dict_top = self.column_to_metadata_map[token]
                idx_top = values_dict_top["id"]
                referral_token = row[idx_top]  # e.g., token 'aux_corine_frac_top_1' might refer to 'corine_frac_211' in this row
                referral_token = 'aux_' + referral_token if 'aux_' not in referral_token else referral_token
                values_dict = self.column_to_metadata_map[referral_token]
            else:
                values_dict = self.column_to_metadata_map[token]
            idx = values_dict["id"]
            value = row[idx]

            formatted_desc = values_dict["description"].lower() or ""
            units = values_dict["units"]
            value = value * 100 if units == "%" else value

            if 'corine' in token:
                if convert_corine_perc:
                    adjective = get_adjective_for_percentage(value)
                    formatted_desc = f'{adjective} {formatted_desc}'
                else:
                    formatted_desc = formatted_desc + f' ({round(value)}{units if units else ""})'
            elif 'bioclim' in token:
                formatted_desc = formatted_desc + f' of {round(value)}{units if units else ""}'
            replacements[token] = formatted_desc

        template = self._fill(template, replacements)
        return template


if __name__ == "__main__":
    _ = ButterflyCaptionBuilder(None, None)
