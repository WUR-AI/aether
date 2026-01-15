import json
import random
import re
from abc import ABC, abstractmethod
from typing import Dict, List, final

import torch

from src.data.base_dataset import BaseDataset


class BaseCaptionBuilder(ABC):
    def __init__(self, templates_path: str, data_dir: str, seed: int) -> None:
        """Interface of caption builder class for converting numerical auxiliary data values into
        textual descriptions from provided caption templates.

        :param templates_path: path to a json file with caption templates.
        :param data_dir: directory where data is stored.
        :param seed: random seed.
        """

        self.templates = json.load(open(templates_path))
        self.tokens_in_template = [self._extract_tokens(t) for t in self.templates]

        self.column_to_metadata_map: Dict[str] | None = None
        self.data_dir = data_dir
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
        return re.findall(r"<([^<>]+)>", template)

    @staticmethod
    def _fill(template: str, fillers: Dict[str, str]) -> str:
        """Fill in templates with values from fillers."""
        for t, f in fillers.items():
            template = template.replace(f"<{t}>", f, 1)
        return template

    @abstractmethod
    def _build_from_template(self, template_idx: int, row: torch.Tensor) -> str:
        """Build caption text from template and row of auxiliary data."""
        pass

    def random(self, aux_values: torch.Tensor, n_random=1) -> List[str]:
        """Return a caption from a randomly sampled template for each data point."""
        n_random = min(n_random, len(aux_values))  # TODO unused
        formatted_rows = []
        template_idx = random.choices(
            range(len(self.templates)),
            k=len(aux_values),
        )
        for idx, row in zip(template_idx, aux_values):
            formatted_rows.append(self._build_from_template(idx, row))

        return formatted_rows

    def all(self, aux_values: torch.Tensor) -> List[str]:
        """Return a list of captions from all available templates."""
        formatted_rows = []
        for row in aux_values:
            descriptions = []
            for template_idx in range(0, len(self)):
                descriptions.append(self._build_from_template(template_idx, row))
            formatted_rows.append(descriptions)

        return formatted_rows

class DummyCaptionBuilder(BaseCaptionBuilder):
    '''Dummy caption builder for testing purposes.'''
    def __init__(self, templates_path: str, data_dir: str, seed: int) -> None:
        super().__init__(templates_path, data_dir, seed)

    def sync_with_dataset(self, dataset) -> None:
        self.dataset = dataset

    def _build_from_template(self, template_idx: int, row: torch.Tensor) -> str:
        first_val = row[0].item() if torch.is_tensor(row) else row[0]
        return f"aux-{first_val}"

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
