import os
import re
import shutil


def rename_s2bms(src_dir: str, dst_dir: str, pattern: str):
    """Move and rename S2BMS files using a name_loc extracted from filenames.

    pattern example: "s2_<name_loc>.tif"
    """

    os.makedirs(dst_dir, exist_ok=True)

    LOC_PATTERN = re.compile(r"([A-Z0-9]+_loc-\d+)")

    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)

        # skip directories
        if not os.path.isfile(src_path):
            continue

        m = LOC_PATTERN.search(fname)
        if not m:
            raise ValueError(f"Could not extract name_loc from filename: {fname}")

        name_loc = m.group(1)

        new_name = pattern.replace("<name_loc>", name_loc)
        dst_path = os.path.join(dst_dir, new_name)

        shutil.move(src_path, dst_path)
        print(f"{fname} → {new_name}")


if __name__ == "__main__":
    os.chdir("../..")
    print(os.getcwd())

    src_dir = "../data/s2bms/s2/"
    dst_dir = "../data/s2bms/eo/s2/"
    pattern = "s2_<name_loc>.tif"

    rename_s2bms(src_dir, dst_dir, pattern)
