import sys

import gdown


def drive_downloader(url, output_file, pooch_obj):
    """Downloader callback for pooch that uses gdown to fetch files from Google Drive.

    Uses fuzzy=True to handle Google Drive's virus scanning page and use_cookies=True to handle
    access restrictions.
    """
    gdown.download(
        url,
        str(output_file),
        quiet=False,
        fuzzy=True,
        use_cookies=True,
    )
