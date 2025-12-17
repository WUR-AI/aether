import random
from abc import abstractmethod, ABC
import json
import re

import torch
from typing import List, final, Dict

from src.data.base_dataset import BaseDataset


class BaseCaptionBuilder(ABC):
    def __init__(self, templates_path: str, data_dir: str) -> None:

        self.templates = json.load(open(templates_path, "r"))
        self.template_tokens = [self._extract_tokens(t) for t in self.templates]

        self.column_to_metadata_map: Dict[str] | None = None

        self.data_dir = data_dir

    @final
    def __len__(self):
        return len(self.templates)

    @abstractmethod
    def sync_with_dataset(self, dataset: BaseDataset) -> None:
        pass

    @staticmethod
    def _count(template: str, token: str) -> int:
        return template.count(f"<{token}>")

    @staticmethod
    def _extract_tokens(template: str) -> List[str]:
        return re.findall(r"<([^<>]+)>", template)

    @staticmethod
    def _fill(template: str, fillers: Dict[str, str]) -> str:
        for t, f in fillers.items():
            template = template.replace(f"<{t}>", f, 1)
        return template

    @abstractmethod
    def _build_from_template(self, template_idx: int, row: torch.Tensor) -> str:
        pass

    def random(self, aux_values: torch.Tensor) -> List[str]:

        formatted_rows = []
        template_idx = random.choices(range(len(self.templates)), k=len(aux_values))
        for idx, row in zip(template_idx, aux_values):
            formatted_rows.append(self._build_from_template(idx, row))

        return formatted_rows

    def all(self, aux_values: torch.Tensor) -> List[str]:

        formatted_rows = []
        for row in aux_values:
            descriptions = []
            for template_idx in range(0, len(self)):
                descriptions.append(self._build_from_template(template_idx, row))
            formatted_rows.append(descriptions)

        return formatted_rows
