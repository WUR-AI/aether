"""Score text concepts against aligned embeddings and plot one map per concept.

Takes the csv written by ``infer_aligned_embeddings.py`` and a json file of
concepts, encodes each concept's captions with the alignment model's text
branch, and scores every location by cosine similarity. Writes a tidy score
table plus one map per concept.

The aligned embeddings are already L2-normalised, so scoring is a single
matrix multiply — no per-location loop, and no model needed on the geo side.

Concept json layout::

    {"concepts": {"<key>": {"label": "...", "captions": ["...", "..."]}}}

Example::

    python scripts/plot_concept_maps.py \
        --ckpt data/checkpoints/other_ckpt/epoch_035.ckpt \
        --embeddings outputs/heat_guatemala_aligned_embeddings.csv \
        --concepts outputs/demo_concepts.json \
        --out-csv outputs/heat_guatemala_concept_scores.csv \
        --out-dir outputs/maps
"""

import argparse
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display on the training machines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.text_encoders.clip_text_encoder import ClipTextEncoder  # noqa: E402

log = logging.getLogger("plot_concept_maps")


def load_text_encoder(ckpt_path, hf_cache_dir, device):
    """Rebuild the text branch: frozen CLIP tower + the projector training produced."""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    encoder = ClipTextEncoder(hf_cache_dir=hf_cache_dir, use_geoclip_projector=True)
    encoder.setup()
    text_sd = {
        k[len("text_encoder.") :]: v
        for k, v in state_dict.items()
        if k.startswith("text_encoder.") and not k.endswith("position_ids")
    }
    _, unexpected = encoder.load_state_dict(text_sd, strict=False)
    if unexpected:
        raise SystemExit(f"Unexpected text-encoder keys in checkpoint: {unexpected}")
    return encoder.eval().to(device)


def concept_vector(encoder, captions, device):
    """Average a concept's phrasings into one unit vector.

    Averaging several wordings is noticeably less noisy than trusting a single
    caption, which is why the concept file carries a list.
    """
    with torch.no_grad():
        embeds = F.normalize(encoder({"text": list(captions)}, mode="train"), dim=-1)
    return F.normalize(embeds.mean(0), dim=0).cpu().numpy()


def plot_concept(lon, lat, z, title, subtitle, path, cmap, clip):
    fig, ax = plt.subplots(figsize=(9, 8.5))
    sc = ax.scatter(lon, lat, c=z, s=4, cmap=cmap, vmin=-clip, vmax=clip, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    fig.colorbar(sc, ax=ax, label="similarity (z-score)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot one similarity map per text concept.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="trained TextAlignmentModel checkpoint")
    p.add_argument(
        "--embeddings", required=True, help="csv from infer_aligned_embeddings.py (list format)"
    )
    p.add_argument("--concepts", required=True, help="json file of concepts and captions")
    p.add_argument("--out-csv", help="write the per-location score table here")
    p.add_argument("--out-dir", default="outputs/maps", help="write the map pngs here")
    p.add_argument(
        "--clip",
        type=float,
        default=2.5,
        help="colour scale limit in z-score units; values beyond this saturate",
    )
    p.add_argument("--cmap", default="RdYlGn", help="matplotlib colormap")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    d = pd.read_csv(args.embeddings)
    if "aligned_embedding" not in d.columns:
        raise SystemExit(
            f"{args.embeddings} has no 'aligned_embedding' column. Regenerate it without "
            "--wide-columns, which splits the vector across one column per dimension."
        )
    aligned = np.stack(d.aligned_embedding.map(json.loads)).astype("float32")
    log.info("Loaded %d locations, %d-d aligned embeddings.", *aligned.shape)

    concepts = json.load(open(args.concepts))["concepts"]
    encoder = load_text_encoder(args.ckpt, os.environ.get("HF_HOME", ".cache"), args.device)
    log.info("Text encoder ready on %s.", args.device)

    out = d[["name_loc", "lat", "lon"]].copy()
    for key, cfg in concepts.items():
        scores = aligned @ concept_vector(encoder, cfg["captions"], args.device)
        # z-score for display only: raw cosines occupy a narrow band, so a map of
        # them reads as flat. Ranking is identical either way.
        z = (scores - scores.mean()) / scores.std()
        out[key] = scores
        out[f"{key}_z"] = z

        path = os.path.join(args.out_dir, f"{key}.png")
        plot_concept(
            d.lon,
            d.lat,
            z,
            cfg.get("label", key),
            f'"{cfg["captions"][0]}"',
            path,
            args.cmap,
            args.clip,
        )
        log.info("  %-22s raw %.3f..%.3f  -> %s", key, scores.min(), scores.max(), path)

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        log.info("Wrote %s %s", args.out_csv, out.shape)

    # Concepts often overlap heavily; showing two near-identical maps is worse
    # than showing one, so surface the correlations rather than letting them
    # surprise someone mid-demo.
    keys = list(concepts)
    if len(keys) > 1:
        corr = pd.DataFrame(
            np.corrcoef(np.stack([out[k].values for k in keys])), index=keys, columns=keys
        )
        log.info("Correlation between concept scores:\n%s", corr.round(2).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
