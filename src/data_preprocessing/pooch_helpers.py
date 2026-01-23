import os

import gdown


def drive_downloader(url, output_file, pooch_obj):
    if os.path.exists(output_file):
        print(f"{output_file} already exists, skipping.")
        return
    gdown.download(url, str(output_file), quiet=False)
