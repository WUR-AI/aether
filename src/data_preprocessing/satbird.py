# https://drive.google.com/drive/folders/1eaL2T7U9Imq_CTDSSillETSDJ1vxi5Wq
import os
import shutil

import pandas as pd
import torch

from src.data_preprocessing.pooch_helpers import drive_downloader

cache_dir = "data/cache"
data_dir = "data/"


def setup_satbird_from_pooch(
    data_dir: str, cache_dir: str, study_site: str = "Kenya", registry_file=None
) -> None:
    """Gets satbird data from the source Google Drive using pooch, structurises this data for this
    project.

    :param data_dir: data directory for the specific study site (e.g. data/satbird_Kenya)
    :param cache_dir: cache directory for pooch
    :param study_site: name of satbird sub dataset (Kenya, USA_summer, USA_winter)
    :param registry_file: path to registry file for pooch
    :return: None
    """

    # Model ready csv path
    model_ready_csv_path = os.path.join(data_dir, f"model_ready_satbird-{study_site}.csv")

    # Flag for missing data (model ready csv or some mod folder empty)
    download_flag = not os.path.exists(model_ready_csv_path)
    for mod in ["s2", "s2rgb", "environmental"]:
        if (
            os.path.join(data_dir, "eo", mod) is not None
            or len(os.path.join(data_dir, "eo", mod)) == 0
        ):
            download_flag = True
            break

    # Get data from pooch
    if download_flag:
        pooch_satbird_downloader(data_dir, cache_dir, study_site, registry_file)


def pooch_satbird_downloader(
    data_dir: str, cache_dir: str, study_site: str, registry_file: str
) -> None:
    """Gets satbird data from the source Google Drive using pooch, structures this data for this
    project.

    :param data_dir: data directory for the specific study site (e.g. data/satbird_Kenya)
    :param cache_dir: cache directory for pooch (optional, will default to base_data_dir/cache)
    :param study_site: name of satbird sub dataset (Kenya, USA_summer, USA_winter)
    :param registry_file: path to registry file for pooch
    :return: None
    """
    import pooch

    # Initialise pooch
    pooch_cli = pooch.create(
        path=cache_dir,
        base_url="",
        registry=None,
    )
    # Load and clean registry
    if not os.path.exists(registry_file):
        raise FileNotFoundError(f"Could not find {registry_file}")

    pooch_cli.load_registry(registry_file)
    for k, v in pooch_cli.registry.items():
        if v in ["None", "none"]:
            pooch_cli.registry[k] = None

    conf = {
        "Kenya": ("Kenya.zip", pooch.Unzip),
        "USA-summer": ("USA_summer.tar.gz", pooch.Untar),
        "USA-winter": ("USA_winter.tar.gz", pooch.Untar),
    }

    fnames = pooch_cli.fetch(
        fname=conf.get(study_site, ValueError)[0],
        downloader=drive_downloader,
        processor=conf.get(study_site, ValueError)[1](),
    )

    extract_satbird_data(data_dir, fnames, study_site)

    # Delete the unzipped dir at the end
    if True:
        unzip_dir = os.path.join(cache_dir, f"{study_site}.zip.unzip")
        for name in os.listdir(unzip_dir):
            path = os.path.join(unzip_dir, name)

            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        shutil.rmtree(unzip_dir)
        print(f"{unzip_dir} emptied")


def extract_satbird_data(data_dir: str, fnames: list[str], study_site: str) -> None:
    """Moves, renames satbird data into desired structure.

    :param data_dir: data directory (e.g. data/satbird_kenya)
    :param fnames: file names of data made available by pooch
    :param study_site: name of satbird sub dataset (Kenya, USA_summer, USA_winter)
    :rtype: None
    """

    # Make eo modality dirs
    env_dir = os.path.join(data_dir, "eo", "environmental")
    s2rgb_dir = os.path.join(data_dir, "eo", "s2rgb")
    s2_dir = os.path.join(data_dir, "eo", "s2")
    for d in [env_dir, s2rgb_dir, s2_dir]:
        os.makedirs(d, exist_ok=True)

    # Initialise model_ready csv extraction if it does not exist yet
    model_ready_csv_path = os.path.join(data_dir, f"model_ready_satbird-{study_site}.csv")
    df = None if os.path.exists(model_ready_csv_path) else pd.DataFrame()
    splits_file = []

    # Iterate through all file names from pooch

    for fname in fnames:
        # get the base name
        base = os.path.basename(fname)
        dst = None

        # json targets to df
        if "target" in fname and df is not None:
            row_df = pd.read_json(fname, orient="index").T
            df = pd.concat([df, row_df], ignore_index=True)
            continue
        # Move modality data
        elif "environmental" in fname:
            dst = os.path.join(env_dir, f"environmental_{base}")
        elif "images_visual" in fname:
            base = base.replace("_visual", "")
            dst = os.path.join(s2rgb_dir, f"s2rgb_{base}")
        elif "images" in fname:
            dst = os.path.join(s2_dir, f"s2_{base}")
        elif "splits_final.csv" in fname:
            splits_file.append(fname)

        if dst is not None and not os.path.exists(dst):
            shutil.move(fname, dst)
            print(f"Moving {base} to {dst}")

    # Compile model ready csv and split file
    if df is not None:
        assert len(splits_file) == 1
        split_df = pd.read_csv(splits_file[0])
        make_model_ready_csv(df, split_df, model_ready_csv_path, study_site)

    split_name = os.path.join(data_dir, "splits", f"split_indices_satbird-{study_site}.pth")
    if not os.path.exists(split_name):
        df = pd.read_csv(model_ready_csv_path)

        # Save split file based on split col
        split_indices = {
            "train_indices": df[df.split == "train"].name_loc,
            "val_indices": df[df.split == "valid"].name_loc,
            "test_indices": df[df.split == "test"].name_loc,
        }
        os.makedirs(os.path.dirname(split_name), exist_ok=True)
        torch.save(split_indices, split_name)
        print(f"Saved split indices to {split_name}")


def make_model_ready_csv(
    df: pd.DataFrame, split_df: pd.DataFrame, model_ready_csv_path: str, study_site: str
):
    """Compiles model ready csv file from retrieved target df and split dataframe.

    :param df: dataframe of target values compiled from target json files
    :param split_df: split dataframe
    :param model_ready_csv_path: name of model ready csv file
    :param study_site: study site name (Kenya, USA_summer, USA_winter)
    """

    if study_site != "Kenya":
        raise NotImplementedError(f"Dataset not implemented for {study_site}")
    # Check for duplicates
    assert not df["hotspot_id"].duplicated().any()

    # Expand target probs into sep columns
    probs_df = pd.DataFrame(df["probs"].to_list(), index=df.index)
    probs_df.columns = [f"target_{i+1}" for i in range(probs_df.shape[1])]
    df = pd.concat([df.drop(columns="probs"), probs_df], axis=1)

    # Clean split_df
    keep_col = [
        "hotspot_id",
        "lon",
        "lat",
        "num_different_species",
        "bio_1",
        "bio_2",
        "bio_3",
        "bio_4",
        "bio_5",
        "bio_6",
        "bio_7",
        "bio_8",
        "bio_9",
        "bio_10",
        "bio_11",
        "bio_12",
        "bio_13",
        "bio_14",
        "bio_15",
        "bio_16",
        "bio_17",
        "bio_18",
        "bio_19",
        "split",
    ]  # TODO USA

    split_df = split_df[keep_col]
    split_df_indexed = split_df.set_index("hotspot_id")

    # Join with env var data and splits
    df_joined = df.join(split_df_indexed, on="hotspot_id", how="left")

    # Standardise names TODO USA
    rename_col = {bio: f"aux_{bio}" for bio in keep_col if "bio" in bio}
    rename_col["hotspot_id"] = "name_loc"

    # Save model ready csv
    df_joined.rename(columns=rename_col, inplace=True)
    df_joined.to_csv(model_ready_csv_path, index=False)
    print(f"Model ready csv saved {model_ready_csv_path}")


if __name__ == "__main__":
    print(os.getcwd())
    study_site = "USA-winter"

    setup_satbird_from_pooch(
        f"data/satbird-{study_site}/",
        cache_dir="data/cache",
        study_site=study_site,
        registry_file="data/registry.txt",
    )
