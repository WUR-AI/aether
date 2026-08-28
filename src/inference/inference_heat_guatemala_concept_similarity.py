"""Validate the Guatemala UHI alignment against ground-truth LST.

Guatemala counterpart of inference_s2bms_habitat_similarity.py. For every
location in a split it collects three things -- the aligned geo embedding, the
predictive model's LST estimate, and the measured LST -- then asks, for each
concept caption, whether text similarity agrees with reality.

The two branches share an input but not a representation: the prediction head
consumes the pooled 128-d Tessera vector, while the aligned embedding is that
same vector pushed through the alignment projection into the 512-d text space.
Neither can be fed to the other.

Run: python src/inference/inference_heat_guatemala_concept_similarity.py
"""

import logging
import os
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv()

if os.environ.get("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.utils.heat_guatemala_inference import (  # noqa: E402
    concept_vectors,
    load_geo_projector,
    load_prediction_branch,
    load_text_encoder,
    read_concepts,
)
from src.utils import extras  # noqa: E402

log = logging.getLogger(__name__)


def collect_split(datamodule, split):
    """Pooled Tessera vectors, targets and ids for one split."""
    dataset = {"train": datamodule.data_train, "val": datamodule.data_val, "test": datamodule.data_test}[split]
    pooled, targets, names = [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        tile = item["eo"]["tessera"]
        pooled.append(torch.nan_to_num(tile.nanmean(dim=(-2, -1)), nan=0.0))
        targets.append(item["target"])
        names.append(dataset.dataset.records[dataset.indices[i]]["name_loc"])
    return torch.stack(pooled), torch.stack(targets).squeeze(-1), names


@hydra.main(
    version_base="1.3",
    config_path="../../configs/",
    config_name="inference_heat_guatemala_concept_similarity.yaml",
)
def main(cfg: DictConfig) -> Optional[pd.DataFrame]:
    extras(cfg)
    device = torch.device(cfg.device)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    pooled, target_lst, names = collect_split(datamodule, cfg.split)
    pooled = pooled.to(device)
    log.info(f"Collected {len(names)} locations from the {cfg.split} split.")

    projector = load_geo_projector(cfg.alignment_ckpt_path).to(device)
    head, normalizer = load_prediction_branch(cfg.predictive_ckpt_path)
    head = head.to(device)
    if normalizer is not None:
        normalizer = normalizer.to(device)

    with torch.no_grad():
        aligned = F.normalize(projector(pooled), dim=-1)
        feats = normalizer(pooled) if normalizer is not None else pooled
        pred_lst = head(feats).squeeze(-1)

    target_lst = target_lst.numpy()
    pred_lst = pred_lst.cpu().numpy()
    log.info(
        f"Predictive model on this split: r={np.corrcoef(pred_lst, target_lst)[0, 1]:.3f}, "
        f"RMSE={np.sqrt(((pred_lst - target_lst) ** 2).mean()):.3f} degC"
    )

    concepts = read_concepts(cfg.concepts_file)
    text_encoder = load_text_encoder(cfg.alignment_ckpt_path, cfg.paths.huggingface_cache).to(
        device
    )
    vectors = concept_vectors(text_encoder, concepts, device)

    rows, scores = [], {}
    for key, vec in vectors.items():
        with torch.no_grad():
            sim = (aligned @ vec).cpu().numpy()
        scores[key] = sim
        rows.append(
            {
                "concept": key,
                "label": concepts[key].get("label", key),
                "r_vs_measured_lst": np.corrcoef(sim, target_lst)[0, 1],
                "r_vs_predicted_lst": np.corrcoef(sim, pred_lst)[0, 1],
            }
        )

    summary = pd.DataFrame(rows)
    log.info(
        "Concept similarity vs LST (correlation over %d locations):\n%s",
        len(names),
        summary.round(3).to_string(index=False),
    )

    if cfg.get("output_csv"):
        per_location = pd.DataFrame(
            {"name_loc": names, "measured_lst": target_lst, "predicted_lst": pred_lst}
        )
        for key, sim in scores.items():
            per_location[key] = sim
        os.makedirs(os.path.dirname(os.path.abspath(cfg.output_csv)) or ".", exist_ok=True)
        per_location.to_csv(cfg.output_csv, index=False)
        log.info(f"Wrote {cfg.output_csv} {per_location.shape}")

    if cfg.get("output_summary_csv"):
        summary.to_csv(cfg.output_summary_csv, index=False)
        log.info(f"Wrote {cfg.output_summary_csv}")

    return summary


if __name__ == "__main__":
    main()
