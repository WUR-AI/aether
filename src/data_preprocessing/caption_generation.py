import itertools
import json
import os
import random

import src.data_preprocessing.data_utils as du

# -------------------------
# 1. Variable pools
# -------------------------

BIOCLIM = [f"<aux_bioclim_{i:02d}>" for i in range(1, 20)]

POP = [
    "<aux_pop_density>",
    "<aux_total_population>",
]

ROADS = [
    "<aux_maxdist_road>",
    "<aux_meandist_road>",
]

CORINE_LOW = [
    "<aux_corine_frac_111>",
    "<aux_corine_frac_112>",
    "<aux_corine_frac_121>",
    "<aux_corine_frac_122>",
    "<aux_corine_frac_123>",
    "<aux_corine_frac_124>",
    "<aux_corine_frac_131>",
    "<aux_corine_frac_132>",
    "<aux_corine_frac_133>",
    "<aux_corine_frac_141>",
    "<aux_corine_frac_142>",
    "<aux_corine_frac_211>",
    "<aux_corine_frac_212>",
    "<aux_corine_frac_213>",
    "<aux_corine_frac_221>",
    "<aux_corine_frac_222>",
    "<aux_corine_frac_223>",
    "<aux_corine_frac_231>",
    "<aux_corine_frac_241>",
    "<aux_corine_frac_242>",
    "<aux_corine_frac_243>",
    "<aux_corine_frac_244>",
    "<aux_corine_frac_311>",
    "<aux_corine_frac_312>",
    "<aux_corine_frac_313>",
    "<aux_corine_frac_321>",
    "<aux_corine_frac_322>",
    "<aux_corine_frac_323>",
    "<aux_corine_frac_324>",
    "<aux_corine_frac_331>",
    "<aux_corine_frac_332>",
    "<aux_corine_frac_333>",
    "<aux_corine_frac_334>",
    "<aux_corine_frac_335>",
    "<aux_corine_frac_411>",
    "<aux_corine_frac_412>",
    "<aux_corine_frac_421>",
    "<aux_corine_frac_422>",
    "<aux_corine_frac_423>",
    "<aux_corine_frac_511>",
    "<aux_corine_frac_512>",
    "<aux_corine_frac_521>",
    "<aux_corine_frac_522>",
    "<aux_corine_frac_523>",
]

CORINE_MID = [
    "<aux_corine_frac_11>",
    "<aux_corine_frac_12>",
    "<aux_corine_frac_13>",
    "<aux_corine_frac_14>",
    "<aux_corine_frac_21>",
    "<aux_corine_frac_22>",
    "<aux_corine_frac_23>",
    "<aux_corine_frac_24>",
    "<aux_corine_frac_31>",
    "<aux_corine_frac_32>",
    "<aux_corine_frac_33>",
    "<aux_corine_frac_41>",
    "<aux_corine_frac_42>",
    "<aux_corine_frac_51>",
    "<aux_corine_frac_52>",
]

CORINE_HIGH = [f"<aux_corine_frac_{i}>" for i in range(1, 6)]

CORINE_HIGH_TOP = [f"<aux_corine_frac_highlevel_top_{i}>" for i in range(1, 4)]

CORINE_MID_TOP = [f"<aux_corine_frac_midlevel_top_{i}>" for i in range(1, 4)]

CORINE_LOW_TOP = [f"<aux_corine_frac_lowlevel_top_{i}>" for i in range(1, 6)]

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
    """Picks a random landcover description style and variables."""
    style = random.choice(["dominant", "mixed", "hierarchical", "transition", "mosaic"])

    if style == "dominant":
        return random.choice(
            [
                f"dominated by {CORINE_LOW_TOP[0]}",
                f"characterised by {random.choice(CORINE_MID_TOP[:2])}",
                f"of {CORINE_HIGH_TOP[0]}",
            ]
        )

    elif style == "mixed":
        return random.choice(
            [
                f"of {CORINE_LOW_TOP[0]}, with {CORINE_LOW_TOP[1]}",
                f"of {CORINE_MID_TOP[0]}, alongside {CORINE_MID_TOP[1]}",
                f"of {CORINE_HIGH_TOP[0]}, with areas of {CORINE_HIGH_TOP[1]}",
            ]
        )

    elif style == "hierarchical":
        return random.choice(
            [
                # broad system → group → specific
                f"of {CORINE_HIGH_TOP[0]} landscapes, especially {CORINE_MID_TOP[0]}",
                f"of {CORINE_MID_TOP[0]} areas containing {CORINE_LOW_TOP[0]}",
                f"of {CORINE_LOW_TOP[0]} within a broader {CORINE_MID_TOP[0]} context",
            ]
        )

    elif style == "transition":
        return random.choice(
            [
                f"with a transition between {CORINE_MID_TOP[0]} and {CORINE_MID_TOP[1]}",
                f"with an interface of {CORINE_HIGH_TOP[0]} and {CORINE_HIGH_TOP[1]}",
                f"ecotonal between {CORINE_LOW_TOP[0]} and {CORINE_LOW_TOP[1]}",
            ]
        )

    elif style == "mosaic":
        return random.choice(
            [
                f"with a mosaic of {CORINE_LOW_TOP[0]}, {CORINE_LOW_TOP[1]} and {CORINE_LOW_TOP[2]}",
                f"with a patchwork of {CORINE_MID_TOP[0]}, {CORINE_MID_TOP[1]} and {CORINE_MID_TOP[2]}",
                f"with a heterogeneous mix of {CORINE_HIGH_TOP[0]}, {CORINE_HIGH_TOP[1]} and {CORINE_HIGH_TOP[2]}",
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


def pick_context():
    """Picks a random context description style and variables."""
    pool = BIOCLIM + POP + ROADS
    k = random.choice([1, 2, 3])
    vars_ = random.sample(pool, k=k)
    tmpl = get_context_template(k)
    # print(get_context_template(k), tmpl)
    if k == 1:
        return tmpl.format(V1=vars_[0])
    elif k == 2:
        return tmpl.format(V1=vars_[0], V2=vars_[1])
    else:
        return tmpl.format(V1=vars_[0], V2=vars_[1], V3=vars_[2])


def generate_captions(n=1000, seed=42, save_path=None):
    """Generates n captions by randomly sampling from the variable and template pools."""
    random.seed(seed)
    captions = set()

    while len(captions) < n:
        cap = f"{pick_entity()} {pick_landcover()}, {pick_context()}."
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
