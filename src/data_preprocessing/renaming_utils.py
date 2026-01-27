import os
import re
import shutil


def rename_s2bms(dst_dir: str, fnames: list | None = None, src_dir: str | None = None):
    """Move and rename S2BMS files using a name_loc extracted from filenames."""

    assert fnames is not None or src_dir is not None
    fnames = fnames or os.listdir(src_dir)

    pattern = "s2_<name_loc>.tif"
    LOC_PATTERN = re.compile(r"([A-Z0-9]+_loc-\d+)")

    os.makedirs(dst_dir, exist_ok=True)

    for fname in fnames:
        if ("__MACOSX" in fname or "DS_Store" in fname) or "tif" not in fname:
            continue

        # Get name_loc from the file name
        m = LOC_PATTERN.search(fname)
        if not m:
            raise ValueError(f"Could not extract name_loc from filename: {fname}")

        # Move
        new_name = "s2_<name_loc>.tif".replace("<name_loc>", m.group(1))
        dst_path = os.path.join(dst_dir, new_name)
        shutil.move(fname, dst_path)
