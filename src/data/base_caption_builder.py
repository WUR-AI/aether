import json
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, final

import torch

from src.data.base_dataset import BaseDataset


class BaseCaptionBuilder(ABC):
    def __init__(self, templates_fname: str, data_dir: str, seed: int) -> None:
        """Interface of caption builder class for converting numerical auxiliary data values into
        textual descriptions from provided caption templates.

        :param templates_fname: path to a json file with caption templates.
        :param data_dir: directory where data is stored.
        :param seed: random seed.
        """

        self.data_dir = data_dir
        templates_path = os.path.join(self.data_dir, "location_caption_templates", templates_fname)
        self.templates = json.load(open(templates_path))
        self.tokens_in_template = [self._extract_tokens(t) for t in self.templates]

        self.column_to_metadata_map: Dict[str] | None = None
        self.seed = seed
        random.seed(self.seed)

    @final
    def __len__(self):
        """Number of caption templates."""
        return len(self.templates)

    @abstractmethod
    def sync_with_dataset(self, dataset: BaseDataset) -> None:
        """Synchronize the dataset with column metadata to obtain column_to_metadata_map."""
        pass

    @staticmethod
    def _extract_tokens(template: str) -> List[str]:
        """Extract tokens in template and return a list of tokens."""
        tokens = re.findall(r"<([^<>]+)>", template)
        # TODO: check if those columns exist in the dataset maps
        return tokens

    @staticmethod
    def _fill(template: str, fillers: Dict[str, str]) -> str:
        """Fill in templates with values from fillers."""
        for t, f in fillers.items():
            template = template.replace(f"<{t}>", f, 1)
        return template

    @abstractmethod
    def _build_from_template(
        self, template_idx: int, aux: torch.Tensor, top: List[str] | None = None
    ) -> str:
        """Build caption text from template and row of auxiliary data."""
        pass

    def random(self, aux_values) -> List[str]:
        """Return a caption from a randomly sampled template for each data point."""
        formatted_rows = []

        batch_size = len(aux_values["aux"])

        template_ids = random.choices(
            range(len(self.templates)),
            k=batch_size,
        )
        for (
            i,
            template_idx,
        ) in enumerate(template_ids):
            row_aux = aux_values["aux"][i]
            row_top = aux_values.get("top")[i] if aux_values.get("top") else None
            formatted_rows.append(
                self._build_from_template(template_idx, aux=row_aux, top=row_top)
            )

        return formatted_rows

    def all(self, aux_values) -> List[str]:
        """Return a list of captions from all available templates."""
        formatted_rows = []
        for i in range(0, len(aux_values["aux"])):
            descriptions = []
            row_aux = aux_values["aux"][i]
            row_top = aux_values.get("top")[i] if aux_values.get("top") else None

            for template_idx in range(0, len(self)):
                descriptions.append(
                    self._build_from_template(template_idx, aux=row_aux, top=row_top)
                )
            formatted_rows.append(descriptions)

        return formatted_rows

    def build_concepts(self, aux_values) -> List[str]:
        pass


class DummyCaptionBuilder(BaseCaptionBuilder):
    """Dummy caption builder for testing purposes."""

    def __init__(self, templates_fname: str, data_dir: str, seed: int) -> None:
        super().__init__(templates_fname, data_dir, seed)

    def sync_with_dataset(self, dataset) -> None:
        pass

    def _build_from_template(
        self, template_idx: int, aux: torch.Tensor, top: List[str] | None = None
    ) -> str:
        first_val = aux[0].item()
        return f"Location with value {first_val}"


def get_adjective_for_percentage(value: float) -> str:
    """Get adjective for percentage value (for land cover etc.)."""
    if value < 10:
        return "little"
    elif value < 20:
        return "some"
    elif value < 30:
        return "quite some"
    elif value < 40:
        return "a lot of"
    elif value < 50:
        return "much"
    elif value < 60:
        return "mostly"
    elif value < 75:
        return "predominantly"
    else:
        return "almost entirely"
