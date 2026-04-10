import glob
import os

import numpy as np


def center_crop(arr, target_shape):
    slices = []
    for dim, target in zip(arr.shape, target_shape):
        start = (dim - target) // 2
        end = start + target
        slices.append(slice(start, end))
    return arr[tuple(slices)]


def main(paths):

    sizes = [256, 128]

    for p in paths:
        img = np.load(p)
        for s in sizes:
            p_id = os.path.basename(p).split(".")[0].split("-")[-1]
            crop = img
            if s != img.shape[0]:
                crop = center_crop(img, (s, s, 128))

            if crop.shape[0:2] != (s, s):
                with open(f"logs/tessera_size_mismatch_{s}.txt", "a") as f:
                    print(f"{p_id} has shape {crop.shape[0:2]}")
                    f.write(f"{p_id}\n")

            if np.isinf(crop.any()):
                with open(f"logs/tessera_nans_{s}.txt", "a") as f:
                    f.write(f"{p_id}\n")

            nulls = np.sum(crop == 0)
            if nulls > (s * s * 128) * 0.5:
                with open(f"logs/tessera_50per_empty_{s}.txt", "a") as f:
                    print(f"50% of {p_id} is 0")
                    f.write(f"{p_id}\n")

            if nulls > (s * s * 128) * 0.25:
                with open(f"logs/tessera_25per_empty_{s}.txt", "a") as f:
                    print(f"25% of {p_id} is 0")
                    f.write(f"{p_id}\n")


if __name__ == "__main__":
    os.chdir("../..")
    print(os.getcwd())

    paths = glob.glob("/lustre/backup/SHARED/AIN/aether/data/s2bms/eo/tessera/*.npy")
    paths = glob.glob("data/s2bms/eo/tessera/*.npy")
    paths.sort()

    main(paths)
