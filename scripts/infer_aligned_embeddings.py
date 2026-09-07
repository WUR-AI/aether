"""Produce text-aligned Tessera embeddings for arbitrary coordinates.

Given (lat, lon) points, this script

  1. fetches the Tessera embedding tile for each point through the GeoTessera API
     (or reuses a tile already cached on disk),
  2. reproduces the geo branch of a trained ``TextAlignmentModel``
     (spatial average -> the learned linear projection), and
  3. writes the resulting embeddings, which live in the same space as the CLIP
     text embeddings the model was aligned against.

The geo branch is rebuilt from the checkpoint rather than from a hydra config so
the script stays runnable outside a training context. Only the modules that the
alignment run actually trained are needed: the checkpoint stores
``geo_encoder.extra_projector`` (Linear 128 -> 512) and ``text_encoder.projector``
(the GeoCLIP mlp), which is everything the two branches need.

Examples
--------
Single point, printed to stdout::

    python scripts/infer_aligned_embeddings.py \
        --ckpt data/checkpoints/other_ckpt/epoch_035-v2.ckpt \
        --lat 14.6349 --lon -90.5069

A csv of points (needs lat/lon columns; name_loc optional), written to npz+csv::

    python scripts/infer_aligned_embeddings.py \
        --ckpt data/checkpoints/other_ckpt/epoch_035-v2.ckpt \
        --csv data/heat_guatemala/model_ready_heat_guatemala.csv \
        --limit 50 --out outputs/aligned.npz

Sanity-check the alignment by scoring each point against free-text captions::

    python scripts/infer_aligned_embeddings.py \
        --ckpt data/checkpoints/other_ckpt/epoch_035-v2.ckpt \
        --lat 14.6349 --lon -90.5069 \
        --captions "a dense built-up urban area" "a green vegetated park"
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_preprocessing.tessera_embeds import (  # noqa: E402
    NoTileError,
    PartialTileError,
    get_tessera_embeds,
)

log = logging.getLogger("infer_aligned_embeddings")

# The geo branch is an AverageEncoder over Tessera, whose band count is fixed.
TESSERA_N_BANDS = 128


# ----------------------------------------------------------------------------
# Tessera retrieval
# ----------------------------------------------------------------------------
def build_geotessera(cache_dir: str, version: str):
    """Open a GeoTessera connection, tolerating both the 0.8 and 0.9 APIs.

    geotessera>=0.9 splits the dataset into named variants; 0.8 has no such
    argument and only ever serves the v1 dataset.
    """
    from geotessera import GeoTessera

    cache_dir = os.path.join(cache_dir, "tessera")
    os.makedirs(cache_dir, exist_ok=True)

    variant = {"v1.1": "cambridge", "v1": "vultr", "v1.0": "vultr"}[version]
    try:
        return GeoTessera(
            cache_dir=cache_dir,
            embeddings_dir=cache_dir,
            dataset_version=version,
            dataset_variant=variant,
        )
    except TypeError:
        if version != "v1":
            raise SystemExit(
                f"geotessera installed here does not support dataset variants, so "
                f"--tessera-version {version} cannot be served. Upgrade to "
                f"geotessera>=0.9.0 (see pyproject.toml) or pass --tessera-version v1."
            )
        log.warning("Falling back to the geotessera 0.8 API (v1 dataset only).")
        return GeoTessera(cache_dir=cache_dir, embeddings_dir=cache_dir)


def fetch_pooled(name_loc, lat, lon, tile_dir, year, tile_size, version, gt):
    """Return one spatially-pooled Tessera embedding (bands,), fetching if needed.

    Mirrors ``BaseDataset.load_npy``: tiles are stored (H, W, bands) on disk and
    transposed to channels-first. Pooling happens here rather than after
    stacking, because across 13k locations the full tiles are ~1.7 GB while the
    pooled vectors are ~7 MB.
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

    # Same operation as AverageEncoder.forward: nanmean over the spatial dims,
    # with an all-NaN channel folded back to 0.
    tile = torch.from_numpy(arr)
    return torch.nan_to_num(tile.nanmean(dim=(-2, -1)), nan=0.0)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_geo_projector(state_dict):
    """Rebuild the trained geo-side projection (Linear 128 -> text dim)."""
    w, b = "geo_encoder.extra_projector.weight", "geo_encoder.extra_projector.bias"
    if w not in state_dict:
        raise SystemExit(
            "Checkpoint has no 'geo_encoder.extra_projector' weights. It was probably "
            "trained with match_to_geo=true (text projected onto the geo space), in which "
            "case the geo embedding is the plain 128-d Tessera average and no projection "
            "is needed — rerun with --no-project."
        )
    out_dim, in_dim = state_dict[w].shape
    projector = torch.nn.Linear(in_dim, out_dim)
    projector.load_state_dict({"weight": state_dict[w], "bias": state_dict[b]})
    projector.eval()
    return projector


def load_text_encoder(state_dict, hf_cache_dir):
    """Rebuild the text branch so captions can be scored against the geo embeddings."""
    from src.models.components.text_encoders.clip_text_encoder import ClipTextEncoder

    encoder = ClipTextEncoder(hf_cache_dir=hf_cache_dir, use_geoclip_projector=True)
    encoder.setup()

    text_sd = {
        k[len("text_encoder.") :]: v
        for k, v in state_dict.items()
        if k.startswith("text_encoder.") and not k.endswith("position_ids")
    }
    missing, unexpected = encoder.load_state_dict(text_sd, strict=False)
    # The frozen CLIP tower is not stored in the checkpoint, so it is expected to
    # come up as "missing" here; anything unexpected is a real mismatch.
    if unexpected:
        raise SystemExit(f"Unexpected text-encoder keys in checkpoint: {unexpected}")
    log.info("Text encoder restored (%d frozen tensors kept from CLIP).", len(missing))
    return encoder


def _as_vector_strings(mat, precision=6):
    """Render each row as a bracketed list so a whole vector fits in one cell."""
    return ["[" + ", ".join(f"{v:.{precision}g}" for v in row) + "]" for row in mat]


def embed_geo(pooled_vecs, projector, device, normalize):
    """Apply the trained projection to spatially-pooled Tessera vectors."""
    pooled = torch.stack(pooled_vecs).to(device)

    with torch.no_grad():
        aligned = projector(pooled) if projector is not None else pooled

    if normalize:
        aligned = F.normalize(aligned, dim=-1)
    return pooled.cpu().numpy(), aligned.cpu().numpy()


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def read_points(args):
    """Return a dataframe with name_loc / lat / lon columns."""
    if args.csv:
        df = pd.read_csv(args.csv, low_memory=False)
        missing = {"lat", "lon"} - set(df.columns)
        if missing:
            raise SystemExit(f"{args.csv} is missing required column(s): {sorted(missing)}")
        if "name_loc" not in df.columns:
            df["name_loc"] = [f"pt_{i:06d}" for i in range(len(df))]
        df = df[["name_loc", "lat", "lon"]]
        if args.limit:
            df = df.head(args.limit)
        return df.reset_index(drop=True)

    return pd.DataFrame(
        {"name_loc": [args.name_loc or "query_000000"], "lat": [args.lat], "lon": [args.lon]}
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fetch Tessera embeddings and project them into the aligned text space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="trained TextAlignmentModel checkpoint (.ckpt)")

    src = p.add_argument_group("points (give either --csv or --lat/--lon)")
    src.add_argument("--csv", help="csv with lat/lon (and optionally name_loc) columns")
    src.add_argument("--lat", type=float)
    src.add_argument("--lon", type=float)
    src.add_argument("--name-loc", help="identifier for a single --lat/--lon point")
    src.add_argument("--limit", type=int, help="only process the first N rows of --csv")

    tes = p.add_argument_group("tessera")
    tes.add_argument(
        "--tile-dir",
        default="data/heat_guatemala/eo/tessera",
        help="where tiles are read from and newly fetched tiles are written",
    )
    tes.add_argument("--year", type=int, default=2024)
    tes.add_argument("--tile-size", type=int, default=10, help="tile edge in pixels (10 m each)")
    tes.add_argument("--tessera-version", default="v1.1", choices=["v1", "v1.0", "v1.1"])
    tes.add_argument("--cache-dir", default="data/cache", help="GeoTessera cache directory")

    out = p.add_argument_group("output")
    out.add_argument("--out", help="write embeddings here (.npz, or .csv)")
    out.add_argument(
        "--wide-columns",
        action="store_true",
        help="csv only: one column per dimension (tessera_000.., aligned_000..) "
        "instead of the whole vector in a single cell",
    )
    out.add_argument(
        "--no-normalize",
        action="store_true",
        help="skip L2 normalisation (the contrastive loss normalises, so keep it on "
        "for cosine similarity against text)",
    )
    out.add_argument(
        "--no-project",
        action="store_true",
        help="return the raw 128-d Tessera average instead of the aligned embedding",
    )
    out.add_argument(
        "--captions",
        nargs="+",
        help="score each point against these captions (loads the CLIP text tower)",
    )
    out.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    if not args.csv and (args.lat is None or args.lon is None):
        raise SystemExit("Give either --csv, or both --lat and --lon.")

    points = read_points(args)
    log.info("Embedding %d point(s).", len(points))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    log.info("Loaded checkpoint %s (epoch %s).", args.ckpt, ckpt.get("epoch"))

    device = torch.device(args.device)
    projector = None if args.no_project else load_geo_projector(state_dict).to(device)
    normalize = not args.no_normalize

    gt = None  # opened lazily: nothing to fetch when every tile is already cached
    pooled_vecs, kept, skipped = [], [], []
    for n_done, row in enumerate(points.itertuples(index=False), start=1):
        try:
            if gt is None and not os.path.exists(
                os.path.join(args.tile_dir, f"tessera_{row.name_loc}.npy")
            ):
                gt = build_geotessera(args.cache_dir, args.tessera_version)
            pooled_vecs.append(
                fetch_pooled(
                    row.name_loc,
                    row.lat,
                    row.lon,
                    args.tile_dir,
                    args.year,
                    args.tile_size,
                    args.tessera_version,
                    gt,
                )
            )
            kept.append(row)
        except (NoTileError, PartialTileError, FileNotFoundError, ValueError) as e:
            log.warning("Skipping %s: %s", row.name_loc, e)
            skipped.append(row.name_loc)
        if n_done % 2000 == 0:
            log.info("  %d/%d points read", n_done, len(points))

    if not pooled_vecs:
        raise SystemExit("No Tessera tiles could be obtained for any requested point.")

    pooled, aligned = embed_geo(pooled_vecs, projector, device, normalize)
    names = [r.name_loc for r in kept]
    log.info(
        "Embedded %d point(s) -> %s%s.",
        len(names),
        aligned.shape,
        f" ({len(skipped)} skipped)" if skipped else "",
    )

    if args.captions:
        text_encoder = load_text_encoder(state_dict, os.environ.get("HF_HOME", ".cache")).to(
            device
        )
        text_encoder.eval()
        with torch.no_grad():
            text_embeds = text_encoder({"text": args.captions}, mode="train")
            text_embeds = F.normalize(text_embeds, dim=-1)
        sims = F.normalize(torch.from_numpy(aligned).to(device), dim=-1) @ text_embeds.T
        print("\ncosine similarity to captions")
        print(pd.DataFrame(sims.cpu().numpy(), index=names, columns=args.captions).round(4))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        lats = [r.lat for r in kept]
        lons = [r.lon for r in kept]
        if args.out.endswith(".csv"):
            frame = pd.DataFrame({"name_loc": names, "lat": lats, "lon": lons})
            if args.wide_columns:
                # One column per dimension, for tools that want a flat matrix.
                frame = pd.concat(
                    [
                        frame,
                        pd.DataFrame(
                            pooled, columns=[f"tessera_{i:03d}" for i in range(pooled.shape[1])]
                        ),
                        pd.DataFrame(
                            aligned, columns=[f"aligned_{i:03d}" for i in range(aligned.shape[1])]
                        ),
                    ],
                    axis=1,
                )
            else:
                # Whole vector per cell, as a bracketed list. Read it back with
                # json.loads (or ast.literal_eval) on the column.
                frame["tessera_embedding"] = _as_vector_strings(pooled)
                frame["aligned_embedding"] = _as_vector_strings(aligned)
            frame.to_csv(args.out, index=False)
        else:
            np.savez(
                args.out,
                name_loc=np.array(names),
                lat=np.array(lats),
                lon=np.array(lons),
                aligned=aligned,
                tessera_avg=pooled,
            )
        log.info("Wrote %s", args.out)
    else:
        for name, vec in zip(names, aligned):
            print(f"{name}\t{np.array2string(vec, precision=4, threshold=8)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
