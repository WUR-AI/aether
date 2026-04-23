import glob
import os

import numpy as np
from torchvision.transforms import v2

import src.data_preprocessing.data_utils as du


def main(paths):
    """Check for missing data in available aef tiles.

    :param paths: paths with all tessera tiles
    :return: None
    """
    sizes = [128, 256]
    for p in paths:
        p_id = os.path.basename(p).split(".")[0].split("-")[-1]
        img = du.load_tiff(p)
        for s in sizes:
            im = v2.CenterCrop(size=s)(img)

            if im.shape[1:] != (s, s):
                with open(f"logs/aef_size_mismatch_{s}.txt", "a") as f:
                    print(f"{p_id} has shape {im.shape[1:]}")
                    f.write(f"{p_id}\n")

            nans = np.sum(np.any(np.isinf(im), axis=0))

            if nans > (s * s) * 0.5:
                with open(f"logs/aef_50per_empty_{s}.txt", "a") as f:
                    print(f"50% of {p_id} is 0")
                    f.write(f"{p_id}\n")

            if nans > (s * s) * 0.25:
                with open(f"logs/aef_25per_empty_{s}.txt", "a") as f:
                    print(f"25% of {p_id} is 0")
                    f.write(f"{p_id}\n")

            # if np.isinf(im).any():
            #     with open(f'logs/aef_nans_{s}.txt', 'a') as f:
            #         f.write(f"{p_id}\n")


if __name__ == "__main__":
    os.chdir("../..")
    print(os.getcwd())

    paths = glob.glob("data/s2bms/eo/aef/*.tif")
    # paths = glob.glob("/lustre/backup/SHARED/AIN/aether/data/s2bms/eo/aef/*.tif")
    paths.sort()

    main(paths)
