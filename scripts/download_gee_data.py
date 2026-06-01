import argparse
import os
import sys

import pandas as pd

from src.data_preprocessing.create_aux_data import get_aux_data_from_coords_list


def main(start=0, stop=2000):
    path_csv = os.path.join(os.environ["DATA_DIR"], "s2bms/source/", "unlabelled_samples_10k.csv")
    assert os.path.exists(path_csv), f"CSV file with locations does not exist: {path_csv}"
    df_samples = pd.read_csv(path_csv)
    assert (
        start >= 0 and stop > start
    ), f"Invalid start ({start}) and stop ({stop}) values. Ensure that 0 <= start < stop."
    if start >= len(df_samples):
        print(
            f"Start index ({start}) exceeds number of available samples ({len(df_samples)}). No data to process."
        )
        return
    if stop > len(df_samples):
        print(
            f"Warning: stop index ({stop}) exceeds number of available samples ({len(df_samples)}). Adjusting stop to {len(df_samples)}."
        )
        stop = len(df_samples)
    coords_list = [(float(row.lon), float(row.lat)) for _, row in df_samples.iterrows()]
    coords_list = coords_list[start:stop]
    name_list = df_samples.name.values[start:stop]

    _, __ = get_aux_data_from_coords_list(
        coords_list=coords_list,
        name_list=name_list,
        save_file=True,
        save_filename=f"aux_data_unlabelled_samples_10k_{start}_{stop}.csv",
        patch_size=2560,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    print(f"Starting download of GEE data for locations from index {args.start} to {args.stop}...")
    main(start=args.start, stop=args.stop)
