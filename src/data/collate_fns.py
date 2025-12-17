from typing import Any, Dict, List

import torch

from src.data.base_caption_builder import BaseCaptionBuilder


def collate_fn(
    batch: List[Any], mode: str = 'train', caption_builder: BaseCaptionBuilder = None
) -> Dict[str, torch.Tensor]:
    """Collates batch into stacked tensors and label lists"""

    # map of all keys present in the batch
    keys = batch[0].keys()
    eo_keys = batch[0].get('eo', {}).keys()
    collected = {k: ([] if k != 'eo' else {k_1: [] for k_1 in eo_keys}) for k in keys}

    # fill-in collected items into batch dict
    for item in batch:
        for k in keys:
            if k == 'eo':
                for k_1 in eo_keys:
                    collected[k][k_1].append(item[k][k_1])
            else:
                collected[k].append(item[k])

    # stack tensors
    for k in keys:
        if k == 'eo':
            for k_1 in eo_keys:
                collected[k][k_1] = torch.stack(collected[k][k_1], dim=0)
        elif type(collected[k][0]) == torch.Tensor:
            collected[k] = torch.stack(collected[k], dim=0)

    # convert aux into captions
    if mode == 'train':
        collected['text'] = caption_builder.random(collected['aux'])
    else:
        collected['text'] = caption_builder.all(collected['aux'])

    collected.pop('aux')

    return collected
