"""Shared loaders for the Guatemala UHI inference scripts.

Lives under src/utils rather than next to the scripts because src/inference.py and
src/inference/ collide as import targets, so src.inference is not a usable package.

The alignment and predictive runs each save only the modules they trained, so
both branches are rebuilt here from their checkpoints rather than from a hydra
model config. That keeps inference runnable without reconstructing a trainer.
"""

import json
import logging

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

TESSERA_N_BANDS = 128


def _state_dict(ckpt_path):
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]


def load_geo_projector(ckpt_path):
    """Geo side of the alignment model: Linear(tessera bands -> text dim)."""
    sd = _state_dict(ckpt_path)
    w, b = "geo_encoder.extra_projector.weight", "geo_encoder.extra_projector.bias"
    if w not in sd:
        raise ValueError(
            f"{ckpt_path} has no geo_encoder.extra_projector. It was probably trained with "
            "match_to_geo=true, in which case the geo embedding is the plain pooled Tessera "
            "vector and no projection is needed."
        )
    out_dim, in_dim = sd[w].shape
    projector = nn.Linear(in_dim, out_dim)
    projector.load_state_dict({"weight": sd[w], "bias": sd[b]})
    return projector.eval()


def load_text_encoder(ckpt_path, hf_cache_dir):
    """Text side of the alignment model: frozen CLIP tower + the trained projector."""
    from src.models.components.text_encoders.clip_text_encoder import ClipTextEncoder

    sd = _state_dict(ckpt_path)
    encoder = ClipTextEncoder(hf_cache_dir=hf_cache_dir, use_geoclip_projector=True)
    encoder.setup()

    # position_ids is a legacy non-persistent buffer written by older transformers
    # versions; it carries no learned information.
    text_sd = {
        k[len("text_encoder.") :]: v
        for k, v in sd.items()
        if k.startswith("text_encoder.") and not k.endswith("position_ids")
    }
    _, unexpected = encoder.load_state_dict(text_sd, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected text-encoder keys in {ckpt_path}: {unexpected}")
    return encoder.eval()


def load_prediction_branch(ckpt_path):
    """Predictive model's head, plus the LayerNorm it was trained behind.

    Mirrors PredictiveModel.forward: pooled features -> optional LayerNorm ->
    prediction head. The head consumes the pooled Tessera vector, not the aligned
    embedding, so the two branches share an input but not a representation.
    """
    sd = _state_dict(ckpt_path)
    head_keys = sorted(k for k in sd if k.startswith("prediction_head.net."))
    if not head_keys:
        raise ValueError(f"{ckpt_path} has no prediction_head weights.")

    layers, idx = [], 0
    while f"prediction_head.net.{idx}.weight" in sd:
        w = sd[f"prediction_head.net.{idx}.weight"]
        layer = nn.Linear(w.shape[1], w.shape[0])
        layer.load_state_dict(
            {"weight": w, "bias": sd[f"prediction_head.net.{idx}.bias"]}
        )
        layers.append(layer)
        idx += 1
        if f"prediction_head.net.{idx}.weight" not in sd and idx < len(head_keys):
            idx += 1  # skip the activation, which holds no parameters
    head = nn.Sequential(*[m for layer in layers[:-1] for m in (layer, nn.ReLU())], layers[-1])

    normalizer = None
    if "normalizer.weight" in sd:
        normalizer = nn.LayerNorm(sd["normalizer.weight"].shape[0])
        normalizer.load_state_dict(
            {"weight": sd["normalizer.weight"], "bias": sd["normalizer.bias"]}
        )
        normalizer.eval()
    return head.eval(), normalizer


def concept_vectors(text_encoder, concepts, device):
    """One unit vector per concept, averaged over that concept's phrasings."""
    vectors = {}
    for key, cfg in concepts.items():
        with torch.no_grad():
            embeds = F.normalize(text_encoder({"text": list(cfg["captions"])}, mode="train"), dim=-1)
        vectors[key] = F.normalize(embeds.mean(0), dim=0).to(device)
    return vectors


def read_concepts(path):
    return json.load(open(path))["concepts"]
