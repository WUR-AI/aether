import glob

import numpy as np
import pandas as pd
import rasterio
import torch

from src.utils.data_utils import center_crop_npy


def _load_tiff(tiff_file_path: str, dtype: np.dtype) -> np.ndarray:
    """Load tiff files as desired dtype np arrays."""
    with rasterio.open(tiff_file_path) as f:
        im = f.read()
        assert isinstance(im, np.ndarray)
        if im.dtype != np.dtype(dtype):
            im = im.astype(dtype=dtype, copy=False)
    return im


def _load_npy(filepath: str, dtype: np.dtype) -> np.ndarray:
    """Load npy files as desired dtype np arrays."""
    arr = np.load(filepath).transpose(2, 0, 1)
    if arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype=dtype, copy=False)
    return arr


def main():
    """Compute average embeddings into csv file."""
    size = 128
    mod = "aef"
    format = "tif"

    # save aef averages
    paths = glob.glob(f"data/s2bms/eo/{mod}/*.{format}")

    rows = []
    for p in paths:
        if format == "npy":
            im = _load_npy(p, np.dtype("float32"))
        else:
            im = _load_tiff(p, np.dtype("float32"))
        im = center_crop_npy(im, (64 if mod == "aef" else 128, size, size))
        tile = torch.from_numpy(im)
        tile = tile.nanmean(dim=(-2, -1))  # shape [64]
        row = {f"avr_{i}": v.item() for i, v in enumerate(tile)}
        row["name_loc"] = f"UKBMS_loc-{p.split('-')[-1].split('.')[0]}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"data/s2bms/eo/{mod}_avr_{size}.csv", index=False)
    print(f"Saved to data/s2bms/eo/{mod}_avr_{size}.csv")


if __name__ == "__main__":
    import os

    os.chdir("../..")
    print(os.getcwd())
    main()
