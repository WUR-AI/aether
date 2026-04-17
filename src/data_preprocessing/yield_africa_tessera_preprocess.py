"""Fetch and cache TESSERA embedding tiles for the yield_africa dataset.

Location: src/data_preprocessing/yield_africa_tessera_preprocess.py

Tiles are saved as NumPy arrays to:
    {data_dir}/yield_africa/eo/tessera/tessera_{name_loc}.npy

This matches the path built by BaseDataset.add_modality_paths_to_df() and
loaded by BaseDataset.setup_tessera() at training time.

Unlike tessera_from_df() (which takes a single fixed year), this script
uses each record's own `year` column so that per-record inter-annual
phenology is captured — the key signal missing from the static tabular
features.

The script is resumable: get_tessera_embeds() skips files that already
exist, so interrupted runs can be continued safely.

Usage
-----
    # All records
    python src/data_preprocessing/yield_africa_tessera_preprocess.py \\
        --data_dir data/

    # Single country, useful for incremental fetching
    python src/data_preprocessing/yield_africa_tessera_preprocess.py \\
        --data_dir data/ --countries KEN RWA

    # Smaller tile size (faster, less context)
    python src/data_preprocessing/yield_africa_tessera_preprocess.py \\
        --data_dir data/ --tile_size 5
"""

import argparse
import logging
import os
import socket
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from geotessera import GeoTessera

from src.data_preprocessing.tessera_embeds import get_tessera_embeds

log = logging.getLogger(__name__)

DATASET_NAME = "yield_africa"
MODEL_READY_CSV = f"model_ready_{DATASET_NAME}.csv"

# Tile size in pixels.  A small tile (e.g. 9) captures local context around
# each plot point without pulling in large surrounding areas.  Consistent
# with the typical smallholder farm size in the region.
DEFAULT_TILE_SIZE = 9


def fetch_tessera_tiles(
    data_dir: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    countries: list[str] | None = None,
    years: list[int] | None = None,
    cache_dir: str | None = None,
    workers: int = 2,
) -> None:
    """Fetch TESSERA tiles for every record in the yield_africa CSV.

    :param data_dir: root data directory (same as ``paths.data_dir`` in configs)
    :param tile_size: spatial extent of each tile in pixels
    :param countries: optional list of country codes to restrict fetching
    :param years: optional list of years to restrict fetching
    :param cache_dir: base directory for all TESSERA cache files.  GeoTessera's
        internal registry is stored here; the large raw downloaded source tiles
        (``global_0.1_degree_representation/`` etc.) are kept in the ``raw/``
        subfolder.  Defaults to the ``TESSERA_EMBEDDINGS_DIR`` env var when set,
        otherwise ``{data_dir}/cache/tessera``.  Point this at an external drive
        when disk space is limited.
    :param workers: number of parallel download threads.  Each thread keeps its
        own GeoTessera instance to avoid shared state.  Default: 2 (external
        drive I/O is usually the bottleneck; more workers add contention).
    """
    dataset_dir = Path(data_dir) / DATASET_NAME
    csv_path = dataset_dir / MODEL_READY_CSV
    save_dir = dataset_dir / "eo" / "tessera"

    if not csv_path.exists():
        raise FileNotFoundError(f"Model-ready CSV not found: {csv_path}")

    save_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = os.environ.get("TESSERA_EMBEDDINGS_DIR") or str(
            Path(data_dir) / "cache" / "tessera"
        )

    embeddings_dir = str(Path(cache_dir) / "raw")

    df = pd.read_csv(csv_path)

    # Optional filters (consistent with YieldAfricaDataset filter params)
    if countries is not None:
        df = df[df["country"].isin(countries)]
        log.info(f"Filtered to countries {countries}: {len(df)} records")
    if years is not None:
        df = df[df["year"].isin(years)]
        log.info(f"Filtered to years {years}: {len(df)} records")

    n_total = len(df)
    n_existing = sum(
        1 for _, row in df.iterrows() if (save_dir / f"tessera_{row.name_loc}.npy").exists()
    )
    n_to_fetch = n_total - n_existing

    print(
        f"Records: {n_total} total, {n_existing} already cached, "
        f"{n_to_fetch} to fetch  (tile_size={tile_size}, workers={workers})\n"
        f"  cache_dir    : {cache_dir}\n"
        f"  embeddings_dir: {embeddings_dir}"
    )

    # Build GeoTessera constructor kwargs shared across all threads.
    # Each thread creates its own instance (thread-local) to avoid sharing
    # internal state such as open file handles and rasterio MemoryFiles.
    _default_registry_dir = Path.home() / ".cache" / "geotessera"
    _use_local_registry = (_default_registry_dir / "registry.parquet").exists()
    _gt_kwargs: dict = {
        # Skip SHA-256 hash verification after each tile download.  Verification
        # reads the entire (potentially large) file again after download, adding
        # noticeable CPU time per tile and making progress look stalled.
        "verify_hashes": False,
    }
    _gt_kwargs["embeddings_dir"] = embeddings_dir

    _thread_local = threading.local()

    def _get_gt() -> GeoTessera:
        """Return a thread-local GeoTessera instance, creating it on first use."""
        if not hasattr(_thread_local, "gt"):
            if _use_local_registry:
                _thread_local.gt = GeoTessera(registry_dir=_default_registry_dir, **_gt_kwargs)
            else:
                _thread_local.gt = GeoTessera(cache_dir=cache_dir, **_gt_kwargs)
        return _thread_local.gt

    def _fetch_one(row) -> str:
        get_tessera_embeds(
            lon=row.lon,
            lat=row.lat,
            name_loc=row.name_loc,
            year=int(row.year),
            save_dir=str(save_dir),
            tile_size=tile_size,
            tessera_con=_get_gt(),
        )
        return row.name_loc

    # Bound all socket operations (urllib HTTP requests inside geotessera).
    # Without this, a stalled connection blocks the thread until the OS TCP
    # keepalive fires, which can take many minutes.
    SOCKET_TIMEOUT = 60  # seconds per socket operation
    HEARTBEAT = 30  # print a heartbeat when no future completes this fast
    TILE_TIMEOUT = 600  # give up warning after 10 min of complete silence
    socket.setdefaulttimeout(SOCKET_TIMEOUT)

    rows = [row for _, row in df.iterrows()]
    done = 0
    pending: set = set()
    silent_seconds = 0

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Submit all jobs up-front; the pool queues them internally.
            futures = {pool.submit(_fetch_one, row): row.name_loc for row in rows}
            pending = set(futures)

            while pending:
                finished, pending = wait(pending, timeout=HEARTBEAT, return_when=FIRST_COMPLETED)

                if not finished:
                    silent_seconds += HEARTBEAT
                    print(
                        f"  ... still working — {done}/{n_total} done, "
                        f"{len(pending)} pending, {silent_seconds}s since last completion"
                    )
                    if silent_seconds >= TILE_TIMEOUT:
                        print(
                            f"  WARNING: no progress in {TILE_TIMEOUT}s, something may be stuck."
                        )
                    continue

                silent_seconds = 0
                for fut in finished:
                    done += 1
                    if done % 100 == 0 or done == n_total:
                        print(f"  {done}/{n_total}")
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"  ERROR fetching {futures[fut]}: {exc}")

    except KeyboardInterrupt:
        print("\nInterrupted — cancelling queued futures (in-flight downloads will finish).")
        for fut in pending:
            fut.cancel()

    print(f"Done. Tiles saved to: {save_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Fetch TESSERA embedding tiles for the yield_africa dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Root data directory (same as paths.data_dir in configs). Default: data/",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size in pixels. Default: {DEFAULT_TILE_SIZE}",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        metavar="CODE",
        help="Country codes to restrict fetching (e.g. KEN RWA). Default: all",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help="Years to restrict fetching (e.g. 2019 2020). Default: all",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Base directory for all TESSERA cache files. "
            "GeoTessera's registry is stored here; large raw source tiles go in "
            "the raw/ subfolder. "
            "Falls back to the TESSERA_EMBEDDINGS_DIR env var, then "
            "{data_dir}/cache/tessera. Set TESSERA_EMBEDDINGS_DIR in .env to "
            "avoid passing this flag every run."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Number of parallel download threads. Default: 2. "
            "When writing to an external drive too many workers can cause I/O "
            "bottlenecks. Reduce the number of workers to improve throughput."
        ),
    )
    args = parser.parse_args()

    print(
        f"Fetching TESSERA tiles  data_dir={args.data_dir}  "
        f"tile_size={args.tile_size}  countries={args.countries or 'all'}  "
        f"years={args.years or 'all'}"
    )
    fetch_tessera_tiles(
        data_dir=args.data_dir,
        tile_size=args.tile_size,
        countries=args.countries,
        years=args.years,
        cache_dir=args.cache_dir,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
