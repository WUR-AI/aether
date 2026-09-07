"""Produce text-aligned Tessera embeddings for the Guatemala UHI use case.

Reads a csv of locations, fetches any Tessera tile not already cached through
the GeoTessera API, pools it and applies the alignment model's geo projection.
The result lives in the same space as the CLIP text embeddings, so a caption can
be scored against a location with a dot product.

Run: python src/inference/inference_heat_guatemala_embeddings.py
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

from src.data_preprocessing.tessera_embeds import (  # noqa: E402
    NoTileError,
    PartialTileError,
    get_tessera_embeds,
)
from src.utils.heat_guatemala_inference import TESSERA_N_BANDS, load_geo_projector  # noqa: E402
from src.utils import extras  # noqa: E402

log = logging.getLogger(__name__)


def build_geotessera(cache_dir, version):
    from geotessera import GeoTessera

    cache_dir = os.path.join(cache_dir, "tessera")
    os.makedirs(cache_dir, exist_ok=True)
    variant = {"v1.1": "cambridge", "v1": "vultr", "v1.0": "vultr"}[version]
    return GeoTessera(
        cache_dir=cache_dir,
        embeddings_dir=cache_dir,
        dataset_version=version,
        dataset_variant=variant,
    )


def fetch_pooled(name_loc, lat, lon, tile_dir, year, tile_size, version, gt):
    """Pooled Tessera embedding for one location, fetching the tile if needed.

    Pooling happens per tile rather than after stacking: across 13k locations the
    full tiles are ~1.7 GB while the pooled vectors are ~7 MB.
    """
    path = os.path.join(tile_dir, f"tessera_{name_loc}.npy")
    if not os.path.exists(path):
        get_tessera_embeds(
            lon=lon,
            lat=lat,
            name_loc=name_loc,
            year=year,
            save_dir=tile_dir,
            tile_size=tile_size,
            tessera_con=gt,
            version=version,
        )

    arr = np.load(path).transpose(2, 0, 1).astype(np.float32, copy=False)
    if arr.shape[0] != TESSERA_N_BANDS:
        raise ValueError(f"{path}: expected {TESSERA_N_BANDS} bands, got {arr.shape[0]}")
    # Same operation as AverageEncoder.forward.
    return torch.nan_to_num(torch.from_numpy(arr).nanmean(dim=(-2, -1)), nan=0.0)


def _vector_strings(mat, precision=6):
    return ["[" + ", ".join(f"{v:.{precision}g}" for v in row) + "]" for row in mat]


@hydra.main(
    version_base="1.3",
    config_path="../../configs/",
    config_name="inference_heat_guatemala_embeddings.yaml",
)
def main(cfg: DictConfig) -> Optional[str]:
    extras(cfg)

    points = pd.read_csv(cfg.points_csv, low_memory=False)
    missing = {"lat", "lon"} - set(points.columns)
    if missing:
        raise ValueError(f"{cfg.points_csv} is missing column(s): {sorted(missing)}")
    if "name_loc" not in points.columns:
        points["name_loc"] = [f"pt_{i:06d}" for i in range(len(points))]
    if cfg.get("limit"):
        points = points.head(cfg.limit)
    log.info(f"Embedding {len(points)} location(s).")

    device = torch.device(cfg.device)
    projector = load_geo_projector(cfg.alignment_ckpt_path).to(device)

    gt = None  # opened lazily; nothing to fetch when every tile is cached
    pooled_vecs, kept, skipped = [], [], []
    for row in points.itertuples(index=False):
        try:
            if gt is None and not os.path.exists(
                os.path.join(cfg.tile_dir, f"tessera_{row.name_loc}.npy")
            ):
                gt = build_geotessera(cfg.paths.cache_dir, cfg.tessera_version)
            pooled_vecs.append(
                fetch_pooled(
                    row.name_loc,
                    row.lat,
                    row.lon,
                    cfg.tile_dir,
                    cfg.year,
                    cfg.tile_size,
                    cfg.tessera_version,
                    gt,
                )
            )
            kept.append(row)
        except (NoTileError, PartialTileError, FileNotFoundError, ValueError) as e:
            log.warning(f"Skipping {row.name_loc}: {e}")
            skipped.append(row.name_loc)

    if not pooled_vecs:
        raise RuntimeError("No Tessera tiles could be obtained for any requested location.")

    pooled = torch.stack(pooled_vecs).to(device)
    with torch.no_grad():
        aligned = projector(pooled)
    if cfg.normalize:
        aligned = F.normalize(aligned, dim=-1)
    pooled, aligned = pooled.cpu().numpy(), aligned.cpu().numpy()
    log.info(f"Embedded {len(kept)} location(s) -> {aligned.shape}, skipped {len(skipped)}.")

    out = pd.DataFrame(
        {
            "name_loc": [r.name_loc for r in kept],
            "lat": [r.lat for r in kept],
            "lon": [r.lon for r in kept],
        }
    )
    if cfg.wide_columns:
        out = pd.concat(
            [
                out,
                pd.DataFrame(pooled, columns=[f"tessera_{i:03d}" for i in range(pooled.shape[1])]),
                pd.DataFrame(
                    aligned, columns=[f"aligned_{i:03d}" for i in range(aligned.shape[1])]
                ),
            ],
            axis=1,
        )
    else:
        # Whole vector per cell; json.loads reads it back.
        out["tessera_embedding"] = _vector_strings(pooled)
        out["aligned_embedding"] = _vector_strings(aligned)

    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_csv)) or ".", exist_ok=True)
    out.to_csv(cfg.output_csv, index=False)
    log.info(f"Wrote {cfg.output_csv} {out.shape}")
    return cfg.output_csv


if __name__ == "__main__":
    main()
