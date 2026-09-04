import argparse
import itertools
import json
import os
import random

# -------------------------
# 1. Variable pools
# -------------------------

VEGIDX = [
    "<aux_ndvi_20240829_label>",
    "<aux_ndwi_20240829_label>",
    "<aux_ari_20240829_label>",
    "<aux_nbr_20240829_label>",
]

URBAN = [
    "<aux_meanbuildingsheight_label>",
    "<aux_buildingscoverha_label>",
    "<aux_buildingscoverpercent_label>",
    "<aux_copernicusmeanimperviousdensity2021_label>",
    "<aux_res_per_1ha_label>",
]

TERRAIN = [
    "<aux_dem_5m_label>",
    "<aux_slope_5m_label>",
    "<aux_aspect_5m_label>",
    "<aux_tpi_5m_label>",
    "<aux_tri_5m_label>",
    "<aux_annualsolarradiation_label>",
]

VEGETATION_STRUCTURE = [
    "<aux_meancanopyheight_label>",
    "<aux_forestcoverha_label>",
    "<aux_forestcoverpercent_label>",
    "<aux_vegetationvoxelnumber_label>",
    "<aux_vegetationlidarptsnumber_label>",
    "<aux_hvv2bv_label>",
    "<aux_copernicusmeantreecoverdensity2023_label>",
    "<aux_copernicusherbaceousvegpresence2023_label>",
]

LULC = [
    "<aux_gridtype_label>",
    "<aux_griddominantlanduse_label>",
    "<aux_copernicusclcplusbackbone2023class_label>",
]
DISTRICTS = ["<aux_districtname_label>"]

# -------------------------
# 2. Language pools
# -------------------------

ENTITIES = [
    "Location",
    "Area",
    "Region",
    "Site",
    "Landscape",
    "Territory",
    "Geographical area",
    "Place",
    "Environment",
    "Setting",
    "Locale",
    "Patch",
    "Overhead view",
    "Earth observation image",
    "Geospatial area",
]

ENTITY_TEMPLATES = [
    "{E}"
    # "This {E}",
    # "A {E}",
    # "The {E}",
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
    lulc_item = random.choice(LULC)

    return random.choice(
        [
            f"dominated by {lulc_item}",
            f"characterised by {lulc_item}",
            f"where {lulc_item} is prevalent",
            f"with {lulc_item}",
        ]
    )


def pick_district():
    """Picks a random district style."""
    district_item = random.choice(DISTRICTS)

    return random.choice(
        [
            f"situated in {district_item}",
            f"placed in {district_item}",
            f"located in {district_item}",
            f"in {district_item}",
            f"that is a part of {district_item}",
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
    pool = VEGIDX + TERRAIN + URBAN + VEGETATION_STRUCTURE + LULC
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


def generate_captions(n=100, seed=42, save_path=None):
    """Generates n captions by randomly sampling from the variable and template pools."""
    random.seed(seed)
    captions = set()

    while len(captions) < n:
        cap = f"{pick_entity()} {pick_landcover()}, {pick_district()}, {pick_context()}."
        captions.add(cap)

    if save_path is not None:
        assert os.path.isdir(save_path), f"save_path must be a directory, got {save_path}"
        existing_versions = [
            int(f.split(".")[0].lstrip("v"))
            for f in os.listdir(save_path)
            if f.startswith("v") and f.endswith(".json") and f.split(".")[0].lstrip("v").isdigit()
        ]
        version = max(existing_versions + [-1]) + 1
        with open(os.path.join(save_path, f"v{version}.json"), "w") as f:
            json.dump(list(captions), f, indent=4)
        print(f"Saved {len(captions)} captions to {os.path.join(save_path, f'v{version}.json')}")
    return list(captions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions.")
    parser.add_argument("--n", type=int, default=20, help="Number of captions to generate")
    parser.add_argument(
        "--save_path", type=str, default=None, help="Directory to save the JSON file"
    )

    args = parser.parse_args()

    caps = generate_captions(n=args.n, save_path=args.save_path)

    # Only print if we aren't saving to a file
    if args.save_path is None:
        for c in caps:
            print(c)
