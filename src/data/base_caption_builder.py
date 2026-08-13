import json
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, final

import torch

from src.data.base_dataset import BaseDataset
from src.utils.errors import IllegalArgumentCombination


class BaseCaptionBuilder(ABC):
    def __init__(
        self,
        templates_fname: str,
        concepts_fname: str,
        data_dir: str,
        seed: int,
        n_captions_for_validation: int | str = "all",
        return_aux_ids: bool = False,
    ) -> None:
        """Interface of caption builder class for converting numerical auxiliary data values into
        textual descriptions from provided caption templates.

        :param templates_fname: path to a json file with caption templates.
        :param concepts_fname: path to a json file with concepts.
        :param data_dir: directory where data is stored.
        :param seed: random seed.
        :param n_captions_for_validation: number of captions to randomly sample for validation
        :param return_aux_ids: whether to return auxiliary column ids.
        """

        self.data_dir = data_dir
        templates_path = os.path.join(self.data_dir, "location_caption_templates", templates_fname)
        self.templates = json.load(open(templates_path))
        self.tokens_in_template = [self._extract_tokens(t) for t in self.templates]

        concepts_path = os.path.join(self.data_dir, "concept_captions", concepts_fname)
        assert os.path.exists(concepts_path), f"Concepts file not found at {concepts_path}"
        assert re.match(
            r"v\d+\.json", concepts_fname
        ), f"Concepts file must be in format v<number>.json, got {concepts_fname}"
        self.concepts_path = concepts_path
        self.concepts = json.load(open(concepts_path))

        self.column_to_metadata_map: Dict[str] | None = None
        self.seed = seed
        random.seed(self.seed)

        if n_captions_for_validation == "all":
            self.n = self.__len__
        elif n_captions_for_validation > len(self):
            raise IllegalArgumentCombination(
                f"Requested {n_captions_for_validation} captions exceeds template dictionary size"
            )
        else:
            self.n = n_captions_for_validation

        self.return_aux_ids = return_aux_ids

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

    @final
    def store_concept_thresholds(self, concept_configs: Dict[str, Any], update_self=True) -> None:
        current_version = re.search(r"v(\d+)\.json", os.path.basename(self.concepts_path)).group(1)
        new_version = int(current_version) + 1
        new_concepts_fname = f"v{new_version}.json"
        while os.path.exists(os.path.join(self.data_dir, "concept_captions", new_concepts_fname)):
            new_version += 1
            new_concepts_fname = f"v{new_version}.json"
        new_concepts_path = os.path.join(self.data_dir, "concept_captions", new_concepts_fname)
        json.dump(concept_configs, open(new_concepts_path, "w"), indent=4)
        print(f"Concept thresholds stored in {new_concepts_path}")
        if update_self:
            self.update_concept_thresholds(concept_configs)
            self.concepts_path = new_concepts_path

    @final
    def update_concept_thresholds(self, concept_configs: Dict[str, Any]) -> None:
        self.concepts = concept_configs

    @abstractmethod
    def _build_from_template(
        self, template_idx: int, aux: torch.Tensor, top: List[str] | None = None
    ) -> str:
        """Build caption text from template and row of auxiliary data."""
        pass

    def random(self, aux_values) -> Tuple[List[str], List[int] | None]:
        """Return a caption per location from a randomly sampled template for each data point.

        :param aux_values: a batch of auxiliary values to use for random sampling.
        :return: a batch of text captions and optionally aux col ids used for each of the caption.
        """
        batch_size = len(aux_values["aux"])

        # Location captions holders
        formatted_location_captions = []

        # Ids of used aux col ids per template (location)
        if self.return_aux_ids:
            ids = []

        # Sample templates
        template_ids = random.choices(range(len(self.templates)), k=batch_size)
        for i, template_idx in enumerate(template_ids):
            # Get aux and top values per location
            row_aux = aux_values["aux"][i]
            row_top = aux_values.get("top")[i] if aux_values.get("top") else None

            # Get filled in template for location
            if self.return_aux_ids:
                filled_template, template_ids = self._build_from_template(
                    template_idx, aux=row_aux, top=row_top
                )
                ids.append(template_ids)
            else:
                filled_template = self._build_from_template(template_idx, aux=row_aux, top=row_top)
            formatted_location_captions.append(filled_template)

        if self.return_aux_ids:
            return formatted_location_captions, ids
        return formatted_location_captions

    def sample_multiple_or_all(self, aux_values) -> Tuple[List[str], List[int] | None]:
        """Return self.n captions from randomly sampled templates for each data point.

        :param aux_values: a batch of auxiliary values to use for random sampling.
        :return: a batch of text captions and optionally aux col ids used for each of the caption.
        """
        batch_size = len(aux_values["aux"])

        # Location captions holders
        formatted_location_captions = []

        # Ids of used aux col ids per template (location)
        if self.return_aux_ids:
            ids = []

        for i in range(0, batch_size):
            # Get aux and top values per location
            row_aux = aux_values["aux"][i]
            row_top = aux_values.get("top")[i] if aux_values.get("top") else None

            # Sample templates
            template_ids = random.choices(range(len(self.templates)), k=self.n)

            # Get filled in templates for location
            filled_in_location_templates = []
            ids_per_location = []
            for template_idx in template_ids:
                if self.return_aux_ids:
                    filled_template, template_ids = self._build_from_template(
                        template_idx, aux=row_aux, top=row_top
                    )
                    ids_per_location.extend(filled_template)
                else:
                    filled_template = self._build_from_template(
                        template_idx, aux=row_aux, top=row_top
                    )
                filled_in_location_templates.append(filled_template)

            if self.return_aux_ids:
                ids.append(template_ids)
            formatted_location_captions.append(filled_in_location_templates)
        if self.return_aux_ids:
            return formatted_location_captions, ids
        return formatted_location_captions

    def sync_concepts(self) -> List[str]:
        for concept in self.concepts:
            concept["id"] = self.column_to_metadata_map["aux"][concept["col"]]["id"]


class DummyCaptionBuilder(BaseCaptionBuilder):
    """Dummy caption builder for testing purposes."""

    def __init__(
        self, templates_fname: str, concepts_fname: str, data_dir: str, seed: int
    ) -> None:
        super().__init__(templates_fname, concepts_fname, data_dir, seed)

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


def sample_adjective_for_percentage(percent: float) -> str:
    """Convert a percentage (0-100) to a descriptive adjective, randomly sampled from synonyms."""
    if not 0 <= percent <= 100:
        raise ValueError(f"Percentage must be between 0 and 100, got {percent}")

    synonyms = {
        "none": ["none", "zero", "absent", "nonexistent"],
        "negligible": ["negligible", "trivial", "trace", "barely any", "scarcely any"],
        "minimal": ["minimal", "tiny", "very little", "marginal", "meager"],
        "slight": ["slight", "small", "modest", "limited", "faint"],
        "some": ["some", "a bit of", "a portion of", "partial", "a measure of"],
        "moderate": ["moderate", "fair", "reasonable", "middling", "decent"],
        "considerable": ["considerable", "notable", "meaningful", "appreciable", "marked"],
        "substantial": ["substantial", "solid", "sizable", "hefty", "goodly"],
        "significant": ["significant", "large", "strong", "pronounced", "prominent"],
        "major": ["major", "great", "high", "intense", "serious"],
        "extensive": ["extensive", "vast", "sweeping", "far-reaching", "immense"],
        "complete": ["complete", "total", "full", "entire", "absolute"],
    }

    if percent == 0:
        key = "none"
    elif percent < 10:
        key = "negligible"
    elif percent < 20:
        key = "minimal"
    elif percent < 30:
        key = "slight"
    elif percent < 40:
        key = "some"
    elif percent < 50:
        key = "moderate"
    elif percent < 60:
        key = "considerable"
    elif percent < 70:
        key = "substantial"
    elif percent < 80:
        key = "significant"
    elif percent < 90:
        key = "major"
    elif percent < 100:
        key = "extensive"
    else:
        key = "complete"

    return random.choice(synonyms[key])
