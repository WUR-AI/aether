import json
import os
import random

import numpy as np

# -------------------------
# 1. Variable pools
# -------------------------

BIOCLIM = [f"<aux_bio_{i}>" for i in range(1, 20)]

SOIL = [
    "<aux_bdticm>",
    "<aux_bldfie>",
    "<aux_cecsol>",
    "<aux_clyppt>",
    "<aux_orcdrc>",
    "<aux_phihox>",
    "<aux_sltppt>",
    "<aux_sndppt>",
]

DW_CLASSES = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_FRAC = [f"<aux_dw_{cls}>" for cls in DW_CLASSES]

DW_TOP = [f"<aux_dw_top_{i}>" for i in range(1, 4)]

# -------------------------
# 2. Language pools
# -------------------------

ENTITIES = [
    "location",
    "area",
    "region",
    "site",
    "zone",
    "landscape",
    "territory",
    "geographical area",
    "place",
    "environment",
    "setting",
    "locale",
    "patch",
    "habitat",
    "ecosystem",
    "habitat region",
    "satellite image",
    "overhead view",
    "earth observation image",
    "geospatial area",
]

ENTITY_TEMPLATES = [
    "{E}",
    # "This {E}",
    # "A {E}",
    # "The {E}",
]

LC_TEMPLATES = [
    "with {LC}",
    "dominated by {LC}",
    "characterised by {LC}",
    "where {LC} is prevalent",
]

CONTEXT_TEMPLATES_ONE = [
    "with {V1}",
    "under {V1} conditions",
    "influenced by {V1}",
    "showing {V1}",
]

CONTEXT_TEMPLATES_TWO = [
    "with {V1} and {V2}",
    "under {V1} and {V2} conditions",
    "influenced by {V1} and {V2}",
    "showing {V1} and {V2}",
]

CONTEXT_TEMPLATES_THREE = [
    "with {V1}, {V2} and {V3}",
    "under {V1}, {V2} and {V3} conditions",
    "influenced by {V1}, {V2} and {V3}",
    "showing {V1}, {V2} and {V3}",
]


# -------------------------
# 3. Functions
# -------------------------


def pick_entity():
    """Picks a random entity and template."""
    e = random.choice(ENTITIES)
    return random.choice(ENTITY_TEMPLATES).format(E=e)


def pick_landcover():
    """Picks a random landcover description style and variables, based on the dynamic world
    top-1/2/3 land cover classes (there is no low/mid/high hierarchy, unlike CORINE)."""
    style = random.choice(["dominant", "mixed", "transition", "mosaic"])

    if style == "dominant":
        return random.choice(
            [
                f"dominated by {DW_TOP[0]}",
                f"characterised by {DW_TOP[0]}",
                f"of {DW_TOP[0]}",
            ]
        )

    elif style == "mixed":
        return random.choice(
            [
                f"of {DW_TOP[0]}, with {DW_TOP[1]}",
                f"of {DW_TOP[0]}, alongside {DW_TOP[1]}",
                f"of {DW_TOP[0]}, with areas of {DW_TOP[1]}",
            ]
        )

    elif style == "transition":
        return random.choice(
            [
                f"with a transition between {DW_TOP[0]} and {DW_TOP[1]}",
                f"with an interface of {DW_TOP[0]} and {DW_TOP[1]}",
                f"ecotonal between {DW_TOP[0]} and {DW_TOP[1]}",
            ]
        )

    elif style == "mosaic":
        return random.choice(
            [
                f"with a mosaic of {DW_TOP[0]}, {DW_TOP[1]} and {DW_TOP[2]}",
                f"with a patchwork of {DW_TOP[0]}, {DW_TOP[1]} and {DW_TOP[2]}",
                f"with a heterogeneous mix of {DW_TOP[0]}, {DW_TOP[1]} and {DW_TOP[2]}",
            ]
        )


def get_context_template(k):
    """Picks a random context template based on the number of variables k."""
    if k == 1:
        return random.choice(CONTEXT_TEMPLATES_ONE)
    elif k == 2:
        return random.choice(CONTEXT_TEMPLATES_TWO)
    else:
        return random.choice(CONTEXT_TEMPLATES_THREE)


def pick_context(pool=None):
    """Picks a random context description style and variables."""
    if pool is None:
        pool = BIOCLIM + SOIL
    k = min(random.choice([1, 2, 3]), len(pool))
    vars_ = random.sample(pool, k=k)

    tmpl = get_context_template(k)
    if k == 1:
        return tmpl.format(V1=vars_[0])
    elif k == 2:
        return tmpl.format(V1=vars_[0], V2=vars_[1])
    else:
        return tmpl.format(V1=vars_[0], V2=vars_[1], V3=vars_[2])


def generate_captions(n=1000, seed=42, save_path=None, template_type="mixed"):
    """Generates n captions by randomly sampling from the variable and template pools."""
    random.seed(seed)
    captions = set()

    while len(captions) < n:
        if template_type == "parallel":
            style = np.random.choice(["dw", "bioclim", "soil"], 1, p=[0.4, 0.4, 0.2]).item()
        else:
            style = template_type

        if style == "mixed":
            cap = f"{pick_entity()} {pick_landcover()}, {pick_context()}."
        elif style == "dw":
            cap = f"{pick_entity()} {pick_landcover()}."
        elif style == "bioclim":
            cap = f"{pick_entity()} {pick_context(pool=BIOCLIM)}."
        elif style == "soil":
            cap = f"{pick_entity()} {pick_context(pool=SOIL)}."
        else:
            raise ValueError(f"Unknown template_type or style: {style}")

        captions.add(cap)

    if save_path is not None:
        assert os.path.isdir(save_path), f"save_path must be a directory, got {save_path}"
        existing_versions = [
            int(f.split(".")[0].lstrip("v"))
            for f in os.listdir(save_path)
            if f.startswith("v") and f.endswith(".json")
        ]
        version = max(existing_versions + [-1]) + 1
        with open(os.path.join(save_path, f"v{version}.json"), "w") as f:
            json.dump(list(captions), f, indent=4)
        print(f"Saved {len(captions)} captions to {os.path.join(save_path, f'v{version}.json')}")
    return list(captions)


if __name__ == "__main__":
    caps = generate_captions(20)
    for c in caps:
        print(c)
