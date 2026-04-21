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
import multiprocessing
import os
import socket
import time
import sys
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_preprocessing.tessera_embeds import NoTileError, PartialTileError, get_tessera_embeds

log = logging.getLogger(__name__)

DATASET_NAME = "yield_africa"
MODEL_READY_CSV = f"model_ready_{DATASET_NAME}.csv"

# Tile size in pixels.  A small tile (e.g. 9) captures local context around
# each plot point without pulling in large surrounding areas.  Consistent
# with the typical smallholder farm size in the region.
DEFAULT_TILE_SIZE = 9


# Per-process GeoTessera instance, created once by _init_worker.
_process_gt = None


def _init_worker(cache_dir: str, use_local_registry: bool, registry_dir: str) -> None:
    """Pool initializer: runs once per worker process to set up GeoTessera."""
    from geotessera import GeoTessera
    global _process_gt
    socket.setdefaulttimeout(60)
    embeddings_dir = str(Path(cache_dir) / "raw")
    gt_kwargs = {"verify_hashes": False, "embeddings_dir": embeddings_dir}
    if use_local_registry:
        _process_gt = GeoTessera(registry_dir=Path(registry_dir), **gt_kwargs)
    else:
        _process_gt = GeoTessera(cache_dir=cache_dir, **gt_kwargs)


def _worker_fetch(args: tuple) -> str:
    """Multiprocessing worker — reuses the per-process GeoTessera instance.

    Must be a top-level function so it is picklable across processes.
    Returning normally means success; raising means error (logged by caller).
    """
    lon, lat, name_loc, year, save_dir, tile_size = args
    get_tessera_embeds(
        lon=lon, lat=lat, name_loc=name_loc, year=year,
        save_dir=save_dir, tile_size=tile_size, tessera_con=_process_gt,
    )
    return name_loc


def fetch_tessera_tiles(
    data_dir: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    countries: list[str] | None = None,
    years: list[int] | None = None,
    cache_dir: str | None = None,
    workers: int = 1,
    retry_stuck: bool = False,
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
    :param workers: number of parallel worker processes.  Each process owns its
        own GeoTessera instance and can be killed on timeout.  Default: 1.
    :param retry_stuck: if True, clear stuck.txt and retry previously-stuck
        records instead of skipping them.
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

    _default_registry_dir = Path.home() / ".cache" / "geotessera"
    _use_local_registry = (_default_registry_dir / "registry.parquet").exists()

    # Per-task timeout: when a worker process exceeds this, it is killed and
    # the record added to stuck.txt.  Multiprocessing (unlike threading) allows
    # true process termination, so stuck downloads cannot block forward progress.
    HEARTBEAT = 15      # seconds between "still fetching" log lines
    TILE_TIMEOUT = 60   # seconds per record before the worker process is killed
    # Note: stuck workers spin at 100% CPU and leak ~55 MB/s of rasterio
    # MemoryFile objects inside GeoTessera's fetch loop.  60s caps peak memory
    # at ~3 GB per stuck record and recovers 3x faster than the old 180s limit.

    # Records that caused a stall in a previous run are skipped unless
    # --retry-stuck is passed, which clears stuck.txt before the run.
    stuck_file = save_dir / "stuck.txt"
    stuck_records: set[str] = set()
    if stuck_file.exists():
        if retry_stuck:
            stuck_file.unlink()
            print("  Cleared stuck.txt — previously-stuck records will be retried.")
        else:
            stuck_records = set(stuck_file.read_text().splitlines())
            if stuck_records:
                print(f"  Skipping {len(stuck_records)} previously-stuck record(s): {sorted(stuck_records)}")

    rows = [row for _, row in df.iterrows() if row.name_loc not in stuck_records]
    done = 0

    _pool_initargs = (cache_dir, _use_local_registry, str(_default_registry_dir))
    pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=_pool_initargs)
    try:
        for row in rows:
            args = (row.lon, row.lat, row.name_loc, int(row.year), str(save_dir), tile_size)
            result = pool.apply_async(_worker_fetch, (args,))
            start = time.monotonic()
            timed_out = False
            while True:
                try:
                    result.get(timeout=HEARTBEAT)
                    break  # completed successfully
                except multiprocessing.TimeoutError:
                    elapsed = int(time.monotonic() - start)
                    if elapsed >= TILE_TIMEOUT:
                        timed_out = True
                        break
                    print(f"  ... fetching {row.name_loc} ({elapsed}s)")
                except NoTileError:
                    print(f"  Skipped {row.name_loc}: no TESSERA data for this location/year")
                    break
                except PartialTileError:
                    print(f"  Skipped {row.name_loc}: tile too close to mosaic edge, not enough context")
                    break
                except Exception as exc:
                    print(f"  ERROR fetching {row.name_loc}: {exc}")
                    break

            if timed_out:
                pool.terminate()
                pool.join()
                pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=_pool_initargs)
                with open(stuck_file, "a") as fh:
                    fh.write(row.name_loc + "\n")
                print(
                    f"  Stuck: {row.name_loc}  "
                    f"lon={row.lon:.4f} lat={row.lat:.4f} year={int(row.year)}"
                )

            done += 1
            if done % 100 == 0 or done == len(rows):
                print(f"  {done}/{n_total}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        pool.terminate()
        pool.join()
        return

    pool.close()
    pool.join()

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
        default=1,
        help=(
            "Number of parallel download threads. Default: 1. "
            "When writing to an external drive too many workers can cause I/O "
            "bottlenecks. Increase with caution."
        ),
    )
    parser.add_argument(
        "--retry-stuck",
        action="store_true",
        default=False,
        help="Clear stuck.txt and retry previously-stuck records instead of skipping them.",
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
        retry_stuck=args.retry_stuck,
    )


if __name__ == "__main__":
    main()
