from typing import Any, Dict, List

import torch

from src.data.base_caption_builder import BaseCaptionBuilder


def smart_stack(values):
    first = values[0]

    if isinstance(first, (torch.Tensor, int, float)):
        return torch.stack(values, dim=0)

    return values


def collate_fn(
    batch: List[Any],
    mode: str = "train",
    caption_builder: BaseCaptionBuilder = None,
) -> Dict[str, torch.Tensor]:
    """Collates batch into stacked tensors and label lists."""

    batch_collected = {}

    if "eo" in batch[0]:
        batch_collected["eo"] = {
            k: torch.stack([item["eo"][k] for item in batch]) for k in batch[0]["eo"].keys()
        }

    if batch[0].get("aux") is not None:
        batch_collected["aux"] = {
            k: smart_stack([item["aux"][k] for item in batch]) for k in batch[0]["aux"].keys()
        }

    if batch[0].get("target") is not None:
        batch_collected["target"] = smart_stack([item["target"] for item in batch])

    # convert aux into captions
    if mode == "train":
        batch_collected["text"] = caption_builder.random(batch_collected["aux"])
    elif mode == "val":
        batch_collected["text"] = caption_builder.all(batch_collected["aux"])
    else:
        batch_collected["text"] = caption_builder.all(batch_collected["aux"])
        # batch_collected['concepts'] = caption_builder.build_concepts(batch_collected["aux"])

    return batch_collected
