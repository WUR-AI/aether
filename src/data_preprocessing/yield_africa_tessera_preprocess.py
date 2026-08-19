"""Fetch and cache TESSERA embedding tiles for the yield_africa dataset.

Location: src/data_preprocessing/yield_africa_tessera_preprocess.py

Tiles are saved as NumPy arrays to:
    {data_dir}/yield_africa/eo/tessera/tessera_{name_loc}_{year}.npy

The year suffix makes each tile's harvest year unambiguous and enables
dual-year fusion (year Y and year Y-1) without a separate directory.
YieldAfricaDataset overrides the default path built by
BaseDataset.add_modality_paths_to_df() to match this convention.

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

    # Also fetch year-1 tiles for dual-year tessera fusion
    python src/data_preprocessing/yield_africa_tessera_preprocess.py \\
        --data_dir data/ --include-prev-year
"""

import argparse
import logging
import multiprocessing
import os
import socket
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_preprocessing.tessera_embeds import (
    NoTileError,
    PartialTileError,
    get_tessera_embeds,
)

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

    Must be a top-level function so it is picklable across processes. Returning normally means
    success; raising means error (logged by caller).
    """
    lon, lat, name_loc, year, save_dir, tile_size = args
    get_tessera_embeds(
        lon=lon,
        lat=lat,
        name_loc=name_loc,
        year=year,
        save_dir=save_dir,
        tile_size=tile_size,
        tessera_con=_process_gt,
    )
    return name_loc


def _run_fetch_loop(
    items: list[tuple[str, tuple]],
    pool_initargs: tuple,
    stuck_file: Path,
    workers: int,
) -> None:
    """Execute one fetch pass through the multiprocessing pool.

    :param items: list of ``(display_name, worker_args)`` pairs.  ``display_name``
        is used in progress messages and appended to ``stuck_file`` on timeout.
        ``worker_args`` is the tuple accepted by ``_worker_fetch``:
        ``(lon, lat, name_loc_stem, year, save_dir_str, tile_size)``.
    :param pool_initargs: forwarded verbatim to ``_init_worker`` on pool start.
    :param stuck_file: path to the stuck log; timed-out display names are appended.
    :param workers: number of pool worker processes.
    :raises KeyboardInterrupt: re-raised after pool cleanup so the caller can exit.
    """
    HEARTBEAT = 15  # seconds between "still fetching" log lines
    TILE_TIMEOUT = 60  # seconds per record before the worker process is killed

    n = len(items)
    pool = multiprocessing.Pool(
        processes=workers, initializer=_init_worker, initargs=pool_initargs
    )
    try:
        for done, (display_name, args) in enumerate(items, 1):
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
                    print(f"  ... fetching {display_name} ({elapsed}s)")
                except NoTileError:
                    print(f"  Skipped {display_name}: no TESSERA data for this location/year")
                    break
                except PartialTileError:
                    print(
                        f"  Skipped {display_name}: tile too close to mosaic edge, not enough context"
                    )
                    break
                except Exception as exc:
                    print(f"  ERROR fetching {display_name}: {exc}")
                    break

            if timed_out:
                pool.terminate()
                pool.join()
                pool = multiprocessing.Pool(
                    processes=workers, initializer=_init_worker, initargs=pool_initargs
                )
                with open(stuck_file, "a") as fh:
                    fh.write(display_name + "\n")
                lon, lat, _, year = args[0], args[1], args[2], args[3]
                print(f"  Stuck: {display_name}  lon={lon:.4f} lat={lat:.4f} year={year}")

            if done % 100 == 0 or done == n:
                print(f"  {done}/{n}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        pool.terminate()
        pool.join()
        raise

    pool.close()
    pool.join()


def fetch_tessera_tiles(
    data_dir: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    countries: list[str] | None = None,
    years: list[int] | None = None,
    cache_dir: str | None = None,
    workers: int = 1,
    retry_stuck: bool = False,
    include_prev_year: bool = False,
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
    :param include_prev_year: if True, after the main year-Y fetch, also fetch
        year-Y−1 tiles needed for dual-year tessera fusion.  Tiles that already
        exist on disk are skipped.  For locations without a CSV row at year-1, a
        synthetic stem ``{name_loc}_prev_{year-1}`` is used and a warning is logged.
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

    # Keep the full unfiltered df for the prev-year lat/lon index.
    df = pd.read_csv(csv_path)
    df_full = df

    # Optional filters (consistent with YieldAfricaDataset filter params)
    if countries is not None:
        df = df[df["country"].isin(countries)]
        log.info(f"Filtered to countries {countries}: {len(df)} records")
    if years is not None:
        df = df[df["year"].isin(years)]
        log.info(f"Filtered to years {years}: {len(df)} records")

    n_total = len(df)
    n_existing = sum(
        1
        for _, row in df.iterrows()
        if (save_dir / f"tessera_{row.name_loc}_{int(row.year)}.npy").exists()
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

    stuck_file = save_dir / "stuck.txt"
    stuck_records: set[str] = set()
    if stuck_file.exists():
        if retry_stuck:
            stuck_file.unlink()
            print("  Cleared stuck.txt — previously-stuck records will be retried.")
        else:
            stuck_records = set(stuck_file.read_text().splitlines())
            if stuck_records:
                print(
                    f"  Skipping {len(stuck_records)} previously-stuck record(s): {sorted(stuck_records)}"
                )

    # Build main fetch items; display_name == row.name_loc for backward compat
    # with existing stuck.txt entries (which store bare name_loc values).
    main_items: list[tuple[str, tuple]] = [
        (
            row.name_loc,
            (
                row.lon,
                row.lat,
                f"{row.name_loc}_{int(row.year)}",
                int(row.year),
                str(save_dir),
                tile_size,
            ),
        )
        for _, row in df.iterrows()
        if row.name_loc not in stuck_records
    ]

    _pool_initargs = (cache_dir, _use_local_registry, str(_default_registry_dir))

    try:
        _run_fetch_loop(main_items, _pool_initargs, stuck_file, workers)
    except KeyboardInterrupt:
        return

    print(f"Done. Tiles saved to: {save_dir}")

    if not include_prev_year:
        return

    # ------------------------------------------------------------------ #
    # Prev-year fetch: tiles for year Y-1 needed for dual-year fusion.    #
    # ------------------------------------------------------------------ #
    print("\nBuilding prev-year tile list...")

    # Refresh stuck records — main loop may have appended new entries.
    stuck_records_prev: set[str] = set()
    if stuck_file.exists():
        stuck_records_prev = set(stuck_file.read_text().splitlines())

    # Build (lat, lon, year) → name_loc index from the full unfiltered CSV so
    # that rows excluded by a countries/years filter can still supply their
    # tile paths as prev-year sources.
    loc_year_index: dict[tuple[float, float, int], str] = {
        (round(float(r.lat), 6), round(float(r.lon), 6), int(r.year)): r.name_loc
        for _, r in df_full.iterrows()
    }

    prev_items: list[tuple[str, tuple]] = []
    n_prev_already_present = 0
    n_synthetic = 0

    for _, row in df.iterrows():
        year_prev = int(row.year) - 1
        key = (round(float(row.lat), 6), round(float(row.lon), 6), year_prev)

        prev_name_loc = loc_year_index.get(key)
        if prev_name_loc is not None:
            # A CSV row exists for this location at year-1; reuse its name_loc.
            stem = f"{prev_name_loc}_{year_prev}"
        else:
            # No CSV row at year-1 (first-year location or gap year).
            # Use a synthetic stem so the tile can be audited.
            stem = f"{row.name_loc}_prev_{year_prev}"
            n_synthetic += 1

        tile_path = save_dir / f"tessera_{stem}.npy"
        if tile_path.exists():
            n_prev_already_present += 1
            continue

        if stem in stuck_records_prev:
            continue

        args = (float(row.lon), float(row.lat), stem, year_prev, str(save_dir), tile_size)
        prev_items.append((stem, args))

    if n_synthetic:
        log.warning(
            "%d prev-year record(s) have no matching CSV row at year-1 "
            "(synthetic stems used; audit tiles named *_prev_* in %s).",
            n_synthetic,
            save_dir,
        )

    print(
        f"Prev-year tiles: {n_prev_already_present} already present, "
        f"{len(prev_items)} to fetch, {n_synthetic} with synthetic stems."
    )

    if prev_items:
        try:
            _run_fetch_loop(prev_items, _pool_initargs, stuck_file, workers)
        except KeyboardInterrupt:
            return

    print("Prev-year fetch done.")


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
    parser.add_argument(
        "--include-prev-year",
        action="store_true",
        default=False,
        help=(
            "After the main year-Y fetch, also fetch year-Y-1 tiles for dual-year "
            "tessera fusion. Tiles already on disk are skipped. "
            "Use with configs/data/yield_africa_tessera_dual.yaml."
        ),
    )
    args = parser.parse_args()

    print(
        f"Fetching TESSERA tiles  data_dir={args.data_dir}  "
        f"tile_size={args.tile_size}  countries={args.countries or 'all'}  "
        f"years={args.years or 'all'}  include_prev_year={args.include_prev_year}"
    )
    fetch_tessera_tiles(
        data_dir=args.data_dir,
        tile_size=args.tile_size,
        countries=args.countries,
        years=args.years,
        cache_dir=args.cache_dir,
        workers=args.workers,
        retry_stuck=args.retry_stuck,
        include_prev_year=args.include_prev_year,
    )


if __name__ == "__main__":
    main()
