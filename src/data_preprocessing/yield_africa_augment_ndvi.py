"""Augment the crop yield Africa model-ready dataset with year-specific NDVI features.

For each row in the model-ready CSV this script locates the pre-computed monthly mean
NDVI file that matches the row's year, finds the nearest (Lat, Lon) point in that file
using a KD-tree, and attaches the 12 monthly NDVI values plus five seasonal aggregates.

NDVI files are expected to live in a single directory and be named:
    NDVI_100m_{year}.csv

They must contain columns ``Lat``, ``Lon``, and twelve monthly mean columns following
the pattern ``{i}_month{m}_meanNDVI`` (i=0..11, m=1..12).  Duplicate (Lat, Lon) pairs
are averaged before the spatial lookup.

--- MODIS download and gap-filling (optional) ---

When ``--download`` is passed, MODIS MOD13Q1 v061 data is used in two ways:

1. **Missing years** — years present in the input CSV but with no pre-computed NDVI CSV
   are fetched from NASA EarthData and a new NDVI_100m_{year}.csv is generated.

2. **Incomplete years** — years that do have a CSV but contain NaN values in one or more
   monthly NDVI columns are also fetched from MODIS.  The MODIS monthly means are used
   to fill only the missing (NaN) months at each grid point; existing values are kept.

Downloaded HDF files are cached under ``--modis_cache_dir`` (default:
``<ndvi_dir>/modis_hdf``).  Re-running without deleting the cache will skip
already-downloaded granules.

The MODIS download requires:
  - ``earthaccess`` Python package  (``uv sync --extra create-data``)
  - NASA EarthData credentials set as environment variables:
      EARTHDATA_USERNAME  — your Earthdata Login username
      EARTHDATA_PASSWORD  — your Earthdata Login password
    Register at https://urs.earthdata.nasa.gov/

MODIS MOD13Q1 provides 16-day composite NDVI at 250 m resolution.  For each year the
script downloads all granules that cover the bounding box of the input CSV, then
aggregates them into monthly means, producing the same NDVI_100m_{year}.csv layout
that the lookup step expects.

New columns added (feat_ndvi_ prefix):
    feat_ndvi_month_{1..12}   — monthly mean NDVI for the matched location/year
    feat_ndvi_mean_{season}   where season ∈ {mam, jja, son, djf, grow}

All new columns are written as raw NDVI values (range roughly −1 to 1).
Normalisation and imputation are handled at training time by the model's TabularEncoder.
Rows whose year has no NDVI file are left as NaN.

Usage (existing CSVs only):
    python src/data_preprocessing/yield_africa_augment_ndvi.py \\
        --input_csv  data/yield_africa/model_ready_yield_africa.csv \\
        --output_csv data/yield_africa/model_ready_yield_africa_ndvi.csv \\
        --ndvi_dir   /Volumes/data_and_models_2/aether/data/cache/ndvi

Usage (with MODIS download for missing/incomplete years):
    python src/data_preprocessing/yield_africa_augment_ndvi.py \\
        --input_csv  data/yield_africa/model_ready_yield_africa.csv \\
        --output_csv data/yield_africa/model_ready_yield_africa_ndvi.csv \\
        --ndvi_dir   data/cache/ndvi \\
        --download
"""

import argparse
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASONS: Dict[str, List[int]] = {
    "djf": [12, 1, 2],
    "mam": [3, 4, 5],
    "jja": [6, 7, 8],
    "son": [9, 10, 11],
    "grow": list(range(3, 11)),  # March–October
}

# Column name template used in the NDVI files.
# Pattern: "{i}_month{month}_meanNDVI" where i = month - 1.
NDVI_COL_PATTERN = re.compile(r"^\d+_month(\d+)_meanNDVI$")

# Warn when the nearest-neighbour distance exceeds this threshold (degrees).
DISTANCE_WARN_DEG = 0.5

# MODIS product configurations.
# MOD13Q1: 16-day composite, 250 m resolution — high detail, ~200 MB/granule.
# MOD13A3: monthly composite, 1 km resolution — lower detail, ~12 MB/granule.
#           MOD13A3 gives exactly one granule per tile per month, so for 9 tiles
#           a full year is only 108 granules (~1.3 GB) vs MOD13Q1's 216 (~44 GB).
MODIS_PRODUCTS = {
    "MOD13Q1": {
        "short_name": "MOD13Q1",
        "version": "061",
        "ndvi_sds_hint": "ndvi",        # substring match against SDS names
        "scale": 0.0001,
        "fill": -3000,
        "is_monthly": False,            # 16-day composite; month derived from DoY
    },
    "MOD13A3": {
        "short_name": "MOD13A3",
        "version": "061",
        "ndvi_sds_hint": "ndvi",
        "scale": 0.0001,
        "fill": -3000,
        "is_monthly": True,             # monthly composite; month derived from DoY
    },
}
# Default product.
MODIS_SHORT_NAME = "MOD13Q1"
MODIS_VERSION = "061"
# Subdataset name inside the HDF4 file.
MODIS_NDVI_SUBDATASET = "250m 16 days NDVI"
# Raw int16 → float NDVI scale factor.
MODIS_NDVI_SCALE = 0.0001
# Fill / no-data value in raw int16.
MODIS_NDVI_FILL = -3000
# MODIS sinusoidal grid parameters (used to map lat/lon → tile h/v).
MODIS_TILE_SIZE_DEG = 10.0      # each tile spans ~10° in both h and v
MODIS_N_HORIZONTAL = 36         # tiles across full longitude range
MODIS_N_VERTICAL   = 18         # tiles across full latitude range


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class _Heartbeat:
    """Background thread that logs a status message every *interval* seconds.

    Usage::

        with _Heartbeat(interval=30, fn=lambda: log.info("still working…")):
            do_slow_blocking_work()

    The callback *fn* is called from the heartbeat thread immediately on entry
    and then again every *interval* seconds until the context exits.  It is
    *not* called on exit (the final state should be logged by the caller).
    """

    def __init__(self, interval: float, fn) -> None:
        self._interval = interval
        self._fn = fn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self._fn()
        while not self._stop.wait(self._interval):
            self._fn()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()


# ---------------------------------------------------------------------------
# NDVI file discovery
# ---------------------------------------------------------------------------


def discover_ndvi_files(ndvi_dir: Path) -> Dict[int, Path]:
    """Return a mapping {year: file_path} for all NDVI files found in *ndvi_dir*."""
    files: Dict[int, Path] = {}
    for path in sorted(ndvi_dir.glob("NDVI_100m_*.csv")):
        m = re.search(r"NDVI_100m_(\d{4})\.csv$", path.name)
        if m:
            files[int(m.group(1))] = path
    return files


# ---------------------------------------------------------------------------
# Load & index a single NDVI file
# ---------------------------------------------------------------------------


def load_ndvi_file(path: Path) -> Tuple[KDTree, pd.DataFrame]:
    """Load one NDVI CSV, aggregate duplicate (Lat, Lon) pairs, and build a KDTree.

    Returns:
        tree   — KDTree over (Lat, Lon) coordinates.
        points — DataFrame with columns Lat, Lon, feat_ndvi_month_1 .. 12 + seasonal.
    """
    log.info(f"  Loading {path.name} ...")
    raw = pd.read_csv(path, low_memory=False)

    # Identify the 12 monthly NDVI columns.
    month_cols: Dict[int, str] = {}
    for col in raw.columns:
        m = NDVI_COL_PATTERN.match(col)
        if m:
            month_cols[int(m.group(1))] = col

    if len(month_cols) != 12:
        log.warning(
            f"  {path.name}: expected 12 monthly NDVI columns, found {len(month_cols)}. "
            f"Months present: {sorted(month_cols)}"
        )

    # Cast monthly columns to float (may arrive as object due to empty cells).
    for col in month_cols.values():
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Aggregate duplicate (Lat, Lon) pairs by mean.
    agg_cols = {col: "mean" for col in month_cols.values()}
    grouped = raw.groupby(["Lat", "Lon"], as_index=False).agg(agg_cols)
    log.info(
        f"  {path.name}: {len(raw)} raw rows → {len(grouped)} unique (Lat, Lon) points"
    )

    # Rename to feat_ndvi_month_{1..12}.
    rename = {col: f"feat_ndvi_month_{month}" for month, col in month_cols.items()}
    grouped = grouped.rename(columns=rename)

    # Compute seasonal aggregates.
    for season, months in SEASONS.items():
        available = [f"feat_ndvi_month_{m}" for m in months if f"feat_ndvi_month_{m}" in grouped.columns]
        col_name = f"feat_ndvi_mean_{season}"
        if available:
            grouped[col_name] = grouped[available].mean(axis=1)
        else:
            grouped[col_name] = np.nan

    # Drop grid points where every monthly NDVI value is NaN.
    all_feat_cols = [f"feat_ndvi_month_{m}" for m in range(1, 13) if f"feat_ndvi_month_{m}" in grouped.columns]
    n_before = len(grouped)
    grouped = grouped[grouped[all_feat_cols].notna().any(axis=1)].reset_index(drop=True)
    n_dropped = n_before - len(grouped)
    if n_dropped:
        log.warning(
            f"  {path.name}: dropped {n_dropped:,} all-NaN grid points before KDTree build; "
            f"{len(grouped):,} points remain"
        )

    if grouped.empty:
        raise ValueError(f"{path.name}: no valid NDVI grid points remain after filtering NaN rows")

    tree = KDTree(grouped[["Lat", "Lon"]].values)
    return tree, grouped


def _monthly_feat_cols(points: pd.DataFrame) -> List[str]:
    """Return the feat_ndvi_month_* columns present in *points*, sorted by month."""
    return sorted(
        [c for c in points.columns if re.match(r"^feat_ndvi_month_(\d+)$", c)],
        key=lambda c: int(c.split("_")[-1]),
    )


def ndvi_has_gaps(points: pd.DataFrame) -> bool:
    """Return True if any grid point in *points* is missing at least one monthly value."""
    month_cols = _monthly_feat_cols(points)
    return bool(points[month_cols].isna().any().any())


def fill_ndvi_gaps(points: pd.DataFrame, modis_df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN monthly NDVI values in *points* using nearest-neighbour lookup in *modis_df*.

    *modis_df* is a DataFrame in the raw NDVI CSV layout (``{i}_month{m}_meanNDVI``
    columns) as returned by ``_hdf_to_monthly_ndvi``.  Only NaN cells are overwritten;
    existing values in *points* are preserved.

    Returns a new DataFrame with the same structure as *points* but with gaps filled
    where MODIS data is available.  Seasonal aggregate columns are recomputed afterward.
    """
    month_cols = _monthly_feat_cols(points)
    months_with_gaps = [
        int(c.split("_")[-1]) for c in month_cols if points[c].isna().any()
    ]
    if not months_with_gaps:
        return points

    log.info(
        f"    Gap-filling {len(months_with_gaps)} month(s) from MODIS: {months_with_gaps}"
    )

    # Build a KDTree on the MODIS grid for fast lookups.
    modis_tree = KDTree(modis_df[["Lat", "Lon"]].values)

    filled = points.copy()
    for month in months_with_gaps:
        feat_col = f"feat_ndvi_month_{month}"
        raw_col = f"{month - 1}_month{month}_meanNDVI"
        if raw_col not in modis_df.columns:
            log.warning(f"    MODIS DataFrame has no column '{raw_col}', skipping month {month}.")
            continue

        nan_mask = filled[feat_col].isna()
        if not nan_mask.any():
            continue

        nan_coords = filled.loc[nan_mask, ["Lat", "Lon"]].values
        _, indices = modis_tree.query(nan_coords)
        modis_values = modis_df[raw_col].iloc[indices].values

        filled.loc[nan_mask, feat_col] = modis_values
        n_filled = int(nan_mask.sum())
        n_still_nan = int(filled[feat_col].isna().sum())
        log.info(
            f"    Month {month:2d}: filled {n_filled} gap(s)"
            + (f", {n_still_nan} still NaN (MODIS also missing)" if n_still_nan else "")
        )

    # Recompute seasonal aggregates with the now-filled monthly values.
    for season, months in SEASONS.items():
        avail = [f"feat_ndvi_month_{m}" for m in months if f"feat_ndvi_month_{m}" in filled.columns]
        col_name = f"feat_ndvi_mean_{season}"
        if avail:
            filled[col_name] = filled[avail].mean(axis=1)
        else:
            filled[col_name] = np.nan

    return filled


# ---------------------------------------------------------------------------
# Nearest-neighbour lookup
# ---------------------------------------------------------------------------


def lookup_ndvi(
    lat: float,
    lon: float,
    tree: KDTree,
    points: pd.DataFrame,
) -> Tuple[pd.Series, float]:
    """Return the NDVI feature row nearest to (lat, lon) and the distance in degrees."""
    dist, idx = tree.query([lat, lon])
    return points.iloc[idx], dist


# ---------------------------------------------------------------------------
# MODIS download helpers
# ---------------------------------------------------------------------------


def _earthaccess_login() -> None:
    """Authenticate with NASA EarthData using environment variables.

    Reads EARTHDATA_USERNAME and EARTHDATA_PASSWORD from the environment.
    Raises RuntimeError if credentials are missing or authentication fails.
    """
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError(
            "The 'earthaccess' package is required for MODIS download. "
            "Install it with: uv sync --extra create-data"
        ) from exc

    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "NASA EarthData credentials not found. "
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables. "
            "Register at https://urs.earthdata.nasa.gov/"
        )

    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError(
            "NASA EarthData authentication failed. "
            "Check EARTHDATA_USERNAME and EARTHDATA_PASSWORD."
        )
    log.info("NASA EarthData authentication successful.")


def _required_modis_tiles(lats: np.ndarray, lons: np.ndarray) -> set:
    """Return the set of MODIS h/v tile IDs that cover the given coordinates.

    Uses the MODIS sinusoidal grid definition: the globe is divided into a
    36×18 grid of ~10°×10° tiles.  Tile (h, v) covers:
        lon ∈ [h*10 - 180,  (h+1)*10 - 180]
        lat ∈ [90 - (v+1)*10, 90 - v*10]

    Returns a set of strings of the form "hHHvVV" (e.g. "h20v09").
    """
    tiles = set()
    for lat, lon in zip(lats, lons):
        h = int((lon + 180.0) / MODIS_TILE_SIZE_DEG)
        v = int((90.0 - lat) / MODIS_TILE_SIZE_DEG)
        # Clamp to valid range
        h = max(0, min(h, MODIS_N_HORIZONTAL - 1))
        v = max(0, min(v, MODIS_N_VERTICAL - 1))
        tiles.add(f"h{h:02d}v{v:02d}")
    return tiles


def _download_modis_year(
    year: int,
    bbox: Tuple[float, float, float, float],
    modis_cache_dir: Path,
    required_tiles: Optional[set] = None,
    product: str = "MOD13Q1",
) -> List[Path]:
    """Download MODIS HDF granules for *year* covering *bbox*.

    *product* selects the MODIS product to download; one of the keys in
    ``MODIS_PRODUCTS`` (``"MOD13Q1"`` or ``"MOD13A3"``).
    *bbox* is (min_lon, min_lat, max_lon, max_lat).

    Already-downloaded files (present in ``modis_cache_dir / str(year)``) are
    skipped so the function is safe to call on repeated runs.

    Returns the list of local HDF file paths (both newly downloaded and cached).
    """
    import earthaccess

    prod_cfg = MODIS_PRODUCTS.get(product, MODIS_PRODUCTS["MOD13Q1"])
    short_name = prod_cfg["short_name"]
    version = prod_cfg["version"]

    year_dir = modis_cache_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    existing = list(year_dir.glob("*.hdf"))
    if existing:
        if required_tiles:
            tile_pattern = re.compile(r"\.(h\d{2}v\d{2})\.")
            filtered_existing = [
                f for f in existing
                if (m := tile_pattern.search(f.name)) and m.group(1) in required_tiles
            ]
            n_skipped = len(existing) - len(filtered_existing)
            log.info(
                f"  {short_name} {year}: {len(existing)} HDF file(s) already cached; "
                f"keeping {len(filtered_existing)} matching required tiles "
                f"({n_skipped} skipped). Required: {sorted(required_tiles)}"
            )
            return filtered_existing
        log.info(f"  {short_name} {year}: {len(existing)} HDF file(s) already cached, skipping download.")
        return existing

    log.info(f"  {short_name} {year}: searching for granules (bbox={bbox}) ...")
    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        bounding_box=bbox,
        temporal=(f"{year}-01-01", f"{year}-12-31"),
        count=-1,
    )

    if not results:
        log.warning(f"  {short_name} {year}: no granules found for bbox={bbox}.")
        return []

    log.info(f"  Granules found: {len(results)}")

    # Filter to only the tiles that actually contain plot locations.
    if required_tiles:
        tile_pattern = re.compile(r"\.(h\d{2}v\d{2})\.")
        filtered = []
        for g in results:
            # earthaccess granule objects expose the filename via producer_granule_id
            # or via the native-id string; fall back to str(g) if neither works.
            name = (
                getattr(g, "producer_granule_id", None)
                or getattr(g, "granule_ur", None)
                or str(g)
            )
            m = tile_pattern.search(str(name))
            if m and m.group(1) in required_tiles:
                filtered.append(g)
        n_skipped = len(results) - len(filtered)
        if n_skipped:
            log.info(
                f"  Tile filter: keeping {len(filtered)} / {len(results)} granules "
                f"({n_skipped} skipped — no plot points in those tiles). "
                f"Required tiles: {sorted(required_tiles)}"
            )
        results = filtered

    if not results:
        log.warning(f"  {short_name} {year}: no granules remain after tile filtering.")
        return []

    log.info(f"  {short_name} {year}: found {len(results)} granule(s), downloading ...")
    files = earthaccess.download(results, str(year_dir))
    downloaded = [Path(f) for f in files if Path(f).exists()]
    log.info(f"  {short_name} {year}: {len(downloaded)} file(s) downloaded to {year_dir}")
    return downloaded


def _make_latlon_grid(
    ul_lon: float,
    ul_lat: float,
    pixel_size_lat: float,
    pixel_size_lon: float,
    nrows: int,
    ncols: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 2-D lat/lon grids for a MODIS tile given its upper-left corner and pixel sizes.

    Returns (lat_2d, lon_2d) arrays shaped (nrows, ncols), where lat decreases
    downward and lon increases rightward.
    """
    row_idx, col_idx = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    lat_2d = ul_lat - (row_idx + 0.5) * pixel_size_lat
    lon_2d = ul_lon + (col_idx + 0.5) * pixel_size_lon
    return lat_2d, lon_2d


def _parse_modis_struct_metadata(hdf_path: Path) -> Optional[dict]:
    """Parse StructMetadata.0 from a MOD13Q1 HDF4 file to extract spatial info.

    Returns a dict with keys: ul_lon, ul_lat, lr_lon, lr_lat, nrows, ncols.
    Returns None if the metadata cannot be parsed.
    """
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise ImportError(
            "pyhdf is required for MODIS HDF4 reading. "
            "Install it with: uv sync --extra create-data"
        ) from exc

    try:
        hdf = SD(str(hdf_path), SDC.READ)
        attrs = hdf.attributes()
        meta_str = attrs.get("StructMetadata.0", "")
        hdf.end()
    except Exception as exc:
        log.debug(f"    {hdf_path.name}: cannot read StructMetadata: {exc}")
        return None

    # Extract UpperLeftPointMtrs and LowerRightMtrs (sinusoidal projection, metres)
    # and XDim / YDim (number of pixels).
    ul_m = re.search(r"UpperLeftPointMtrs=\(([^,]+),([^)]+)\)", meta_str)
    lr_m = re.search(r"LowerRightMtrs=\(([^,]+),([^)]+)\)", meta_str)
    xdim_m = re.search(r"XDim=(\d+)", meta_str)
    ydim_m = re.search(r"YDim=(\d+)", meta_str)

    if not (ul_m and lr_m and xdim_m and ydim_m):
        log.debug(f"    {hdf_path.name}: incomplete StructMetadata, cannot derive grid.")
        return None

    # MODIS sinusoidal projection: x is easting (m), y is northing (m).
    # Earth radius used by MODIS sinusoidal: R = 6371007.181 m.
    R = 6_371_007.181
    ul_x, ul_y = float(ul_m.group(1)), float(ul_m.group(2))
    lr_x, lr_y = float(lr_m.group(1)), float(lr_m.group(2))
    ncols = int(xdim_m.group(1))
    nrows = int(ydim_m.group(1))

    # Convert sinusoidal metres to geographic degrees.
    # Sinusoidal: lat = y / R, lon = x / (R * cos(lat))
    ul_lat = np.degrees(ul_y / R)
    ul_lon = np.degrees(ul_x / (R * np.cos(np.radians(ul_lat)))) if abs(ul_lat) < 90 else 0.0
    lr_lat = np.degrees(lr_y / R)
    lr_lon = np.degrees(lr_x / (R * np.cos(np.radians(lr_lat)))) if abs(lr_lat) < 90 else 0.0

    return {
        "ul_lon": ul_lon, "ul_lat": ul_lat,
        "lr_lon": lr_lon, "lr_lat": lr_lat,
        "nrows": nrows,   "ncols": ncols,
    }


def _read_one_hdf(
    hdf_path: Path,
    year: int,
    product: str = "MOD13Q1",
    plot_lats: Optional[np.ndarray] = None,
    plot_lons: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Read one MODIS HDF4 file and return NDVI values.

    When *plot_lats* / *plot_lons* are provided (the preferred, memory-efficient
    path), only the pixels nearest to those plot locations are sampled.  The
    return value is ``(month, (plot_indices, ndvi_1d, None))`` where
    *plot_indices* are the integer positions into the original plot arrays of
    the points that fall within this tile, and *ndvi_1d* are their sampled
    NDVI values (NaN where the pixel is fill/invalid).  Points outside this
    tile are omitted.

    When no plot coordinates are given (legacy path), the full 2-D tile arrays
    are returned as ``(month, (lat_2d, lon_2d, ndvi_2d))``.

    Uses pyhdf.SD directly because GDAL 3.10+ is commonly compiled without the
    HDF4 driver, making rioxarray/xarray unable to open .hdf files.
    """
    from datetime import datetime

    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise ImportError(
            "pyhdf is required for MODIS HDF4 reading. "
            "Install it with: uv sync --extra create-data"
        ) from exc

    doy_pattern = re.compile(r"\.A(\d{4})(\d{3})\.")
    m = doy_pattern.search(hdf_path.name)
    if not m:
        log.warning(f"    Cannot parse date from filename: {hdf_path.name}, skipping.")
        return None

    file_year = int(m.group(1))
    doy = int(m.group(2))

    if file_year != year:
        log.debug(f"    {hdf_path.name}: year {file_year} != expected {year}, skipping.")
        return None

    month = (datetime(file_year, 1, 1) + pd.Timedelta(days=doy - 1)).month

    # Parse spatial metadata for grid → lat/lon conversion.
    geo = _parse_modis_struct_metadata(hdf_path)
    if geo is None:
        log.warning(f"    {hdf_path.name}: cannot parse spatial metadata, skipping.")
        return None

    nrows, ncols = geo["nrows"], geo["ncols"]
    pixel_size_lat = (geo["ul_lat"] - geo["lr_lat"]) / nrows
    pixel_size_lon = (geo["lr_lon"] - geo["ul_lon"]) / ncols

    # When plot coordinates are supplied, determine which points fall inside
    # this tile and compute their pixel row/col via the affine transform.
    # Points outside the tile bounding box are skipped — no raster needed.
    if plot_lats is not None and plot_lons is not None:
        in_tile = (
            (plot_lats <= geo["ul_lat"]) & (plot_lats >= geo["lr_lat"]) &
            (plot_lons >= geo["ul_lon"]) & (plot_lons <= geo["lr_lon"])
        )
        if not in_tile.any():
            return None  # this tile contains no plot points

        # Keep the original integer indices so the caller can scatter directly
        # into a (n_plots,) accumulator without any coordinate re-mapping.
        pt_indices = np.where(in_tile)[0]
        pt_lats = plot_lats[pt_indices]
        pt_lons = plot_lons[pt_indices]

        # Affine: row = (ul_lat - lat) / pixel_size_lat,  col = (lon - ul_lon) / pixel_size_lon
        rows_f = (geo["ul_lat"] - pt_lats) / pixel_size_lat - 0.5
        cols_f = (pt_lons - geo["ul_lon"]) / pixel_size_lon - 0.5
        row_idx = np.clip(np.round(rows_f).astype(np.int32), 0, nrows - 1)
        col_idx = np.clip(np.round(cols_f).astype(np.int32), 0, ncols - 1)

        try:
            hdf = SD(str(hdf_path), SDC.READ)
            datasets = hdf.datasets()
            ndvi_key = next((n for n in datasets if "ndvi" in n.lower()), None)
            if ndvi_key is None:
                hdf.end()
                log.warning(f"    {hdf_path.name}: NDVI SDS not found.")
                return None
            sds = hdf.select(ndvi_key)
            raw = sds.get().astype(np.float32)
            sds.endaccess()
            hdf.end()
        except Exception as exc:
            log.warning(f"    {hdf_path.name}: error reading NDVI SDS: {exc}")
            try:
                hdf.end()
            except Exception:
                pass
            return None

        actual_rows = min(raw.shape[0], nrows)
        actual_cols = min(raw.shape[1], ncols)
        row_idx = np.clip(row_idx, 0, actual_rows - 1)
        col_idx = np.clip(col_idx, 0, actual_cols - 1)

        prod_cfg = MODIS_PRODUCTS.get(product, MODIS_PRODUCTS["MOD13Q1"])
        raw_pts = raw[row_idx, col_idx]
        ndvi_pts = np.where(raw_pts == prod_cfg["fill"], np.nan, raw_pts * prod_cfg["scale"])
        ndvi_pts = np.where((ndvi_pts < -1.0) | (ndvi_pts > 1.0), np.nan, ndvi_pts)

        # Return (pt_indices, ndvi_pts, None) — caller uses pt_indices to scatter
        # into the (n_plots,) accumulator; the None slot keeps the tuple shape uniform.
        return month, (pt_indices, ndvi_pts, None)

    # --- Legacy path: return full 2-D tile arrays (used when no plot coords given) ---
    try:
        hdf = SD(str(hdf_path), SDC.READ)
    except Exception as exc:
        log.warning(f"    Failed to open {hdf_path.name}: {exc}")
        return None

    try:
        datasets = hdf.datasets()
        ndvi_key = None
        for name in datasets:
            if "ndvi" in name.lower():
                ndvi_key = name
                break
        if ndvi_key is None:
            log.warning(
                f"    {hdf_path.name}: NDVI SDS not found. "
                f"Available: {list(datasets.keys())}"
            )
            hdf.end()
            return None

        sds = hdf.select(ndvi_key)
        raw = sds.get().astype(np.float32)
        sds.endaccess()
        hdf.end()
    except Exception as exc:
        log.warning(f"    {hdf_path.name}: error reading NDVI SDS: {exc}")
        try:
            hdf.end()
        except Exception:
            pass
        return None

    prod_cfg = MODIS_PRODUCTS.get(product, MODIS_PRODUCTS["MOD13Q1"])
    ndvi = np.where(raw == prod_cfg["fill"], np.nan, raw * prod_cfg["scale"])
    ndvi = np.where((ndvi < -1.0) | (ndvi > 1.0), np.nan, ndvi)

    if raw.shape != (nrows, ncols):
        nrows = min(raw.shape[0], nrows)
        ncols = min(raw.shape[1], ncols)
        ndvi = ndvi[:nrows, :ncols]

    lat_2d, lon_2d = _make_latlon_grid(
        ul_lon=geo["ul_lon"], ul_lat=geo["ul_lat"],
        pixel_size_lat=pixel_size_lat, pixel_size_lon=pixel_size_lon,
        nrows=nrows, ncols=ncols,
    )

    return month, (lat_2d, lon_2d, ndvi)


def _hdf_to_monthly_ndvi(
    hdf_files: List[Path],
    year: int,
    workers: int = 4,
    product: str = "MOD13Q1",
    plot_lats: Optional[np.ndarray] = None,
    plot_lons: Optional[np.ndarray] = None,
) -> Optional[pd.DataFrame]:
    """Convert MODIS HDF files for one year into a monthly-mean NDVI DataFrame.

    When *plot_lats* / *plot_lons* are supplied (the fast path), only the pixels
    nearest to those locations are sampled from each tile — the output DataFrame
    has at most ``len(plot_lats)`` rows and is built in seconds regardless of
    tile resolution.  Memory use is O(n_plots), not O(n_pixels).

    Without plot coordinates (legacy path), the full raster grid is returned.

    Output columns: Lat, Lon, 0_month1_meanNDVI, ..., 11_month12_meanNDVI.
    Returns None if no valid data could be extracted.
    """
    n_files = len(hdf_files)
    log.info(
        f"  {product} {year}: sampling {n_files} HDF file(s) "
        f"at {len(plot_lats) if plot_lats is not None else 'all'} location(s) "
        f"with {workers} thread(s) ..."
    )

    # Per-plot, per-month accumulators (sum and count for running mean).
    # When plot coords are given these are shape (n_plots,); otherwise we
    # fall back to collecting full tile arrays as before.
    use_plot_sampling = plot_lats is not None and plot_lons is not None
    n_plots = len(plot_lats) if use_plot_sampling else 0

    if use_plot_sampling:
        month_sum: Dict[int, np.ndarray] = {m: np.zeros(n_plots, dtype=np.float64) for m in range(1, 13)}
        month_cnt: Dict[int, np.ndarray] = {m: np.zeros(n_plots, dtype=np.int32)   for m in range(1, 13)}
    else:
        month_arrays: Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {m: [] for m in range(1, 13)}

    done_count = 0
    hdf_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _read_one_hdf, p, year, product,
                plot_lats if use_plot_sampling else None,
                plot_lons if use_plot_sampling else None,
            ): p
            for p in hdf_files
        }
        for fut in as_completed(futures):
            done_count += 1
            hdf_path = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                log.warning(f"    {hdf_path.name}: unexpected error: {exc}")
                result = None

            if result is not None:
                month, (a, b, ndvi) = result
                if use_plot_sampling:
                    # a = pt_indices (int array into plot_lats/plot_lons)
                    # b = ndvi_pts (1-D float array, may contain NaN)
                    # (the third element is None — unused in this path)
                    pt_indices = a
                    ndvi_vals  = b
                    valid = ~np.isnan(ndvi_vals)
                    if valid.any():
                        np.add.at(month_sum[month], pt_indices[valid], ndvi_vals[valid])
                        np.add.at(month_cnt[month], pt_indices[valid], 1)
                else:
                    month_arrays[month].append((a, b, ndvi))

            elapsed = time.time() - hdf_start
            pct = 100 * done_count / n_files
            filled = int(20 * done_count / n_files)
            bar = "█" * filled + "░" * (20 - filled)
            rate = done_count / elapsed if elapsed > 0 else 0
            eta_str = _fmt_duration((n_files - done_count) / rate) if rate > 0 and done_count < n_files else "—"
            log.info(
                f"    [{bar}] {done_count:3d}/{n_files}  {pct:5.1f}%  "
                f"elapsed={_fmt_duration(elapsed)}  ETA={eta_str}"
            )

    if use_plot_sampling:
        # Build output DataFrame directly from per-plot accumulators.
        rows: Dict[str, np.ndarray] = {
            "Lat": plot_lats.astype(np.float32),
            "Lon": plot_lons.astype(np.float32),
        }
        months_covered = 0
        for month in range(1, 13):
            col = f"{month - 1}_month{month}_meanNDVI"
            cnt = month_cnt[month]
            ndvi_col = np.where(cnt > 0, month_sum[month] / cnt, np.nan).astype(np.float32)
            rows[col] = ndvi_col
            if (cnt > 0).any():
                months_covered += 1

        df = pd.DataFrame(rows)
        monthly_cols = [f"{m - 1}_month{m}_meanNDVI" for m in range(1, 13)]
        df = df[df[monthly_cols].notna().any(axis=1)].reset_index(drop=True)
        log.info(
            f"  {product} {year}: built plot-sampled NDVI table: {len(df)} locations, "
            f"{months_covered} months covered."
        )
        return df

    # --- Legacy full-raster path (no plot coords) ---
    SCALE = 10_000
    month_keys: Dict[int, Optional[np.ndarray]] = {m: None for m in range(1, 13)}
    month_sums: Dict[int, Optional[np.ndarray]] = {m: None for m in range(1, 13)}
    month_cnts: Dict[int, Optional[np.ndarray]] = {m: None for m in range(1, 13)}

    has_any = False
    agg_start = time.time()
    months_total = sum(1 for v in month_arrays.values() if v)
    months_done = [0]
    _cur_month  = [0]
    _cur_npix   = [0]

    def _agg_heartbeat() -> None:
        elapsed = time.time() - agg_start
        log.info(
            f"  {product} {year}: reducing month {_cur_month[0]:2d}/12  "
            f"({_cur_npix[0]:,} px)  done {months_done[0]}/{months_total}  "
            f"elapsed={_fmt_duration(elapsed)}"
        )

    for month, tile_list in month_arrays.items():
        if not tile_list:
            continue
        has_any = True
        _cur_month[0] = month

        lat_parts:  List[np.ndarray] = []
        lon_parts:  List[np.ndarray] = []
        ndvi_parts: List[np.ndarray] = []
        for lat_2d, lon_2d, ndvi_2d in tile_list:
            ndvi_flat = ndvi_2d.ravel()
            valid = ~np.isnan(ndvi_flat)
            if not valid.any():
                continue
            lat_parts.append(lat_2d.ravel()[valid])
            lon_parts.append(lon_2d.ravel()[valid])
            ndvi_parts.append(ndvi_flat[valid])
        month_arrays[month] = []

        if not lat_parts:
            months_done[0] += 1
            continue

        lats_all  = np.concatenate(lat_parts);  lat_parts.clear()
        lons_all  = np.concatenate(lon_parts);  lon_parts.clear()
        ndvis_all = np.concatenate(ndvi_parts).astype(np.float64);  ndvi_parts.clear()
        _cur_npix[0] = len(lats_all)

        with _Heartbeat(interval=30, fn=_agg_heartbeat):
            lat_i = np.round(lats_all * SCALE).astype(np.int64)
            lon_i = np.round(lons_all * SCALE).astype(np.int64)
            keys  = lat_i * 4_000_000 + lon_i
            del lats_all, lons_all, lat_i, lon_i
            unique_keys, inverse = np.unique(keys, return_inverse=True)
            del keys
            ndvi_sum = np.bincount(inverse, weights=ndvis_all, minlength=len(unique_keys))
            ndvi_cnt = np.bincount(inverse,                    minlength=len(unique_keys)).astype(np.int64)
            del ndvis_all, inverse

        month_keys[month] = unique_keys
        month_sums[month] = ndvi_sum
        month_cnts[month] = ndvi_cnt
        months_done[0] += 1
        elapsed_agg = time.time() - agg_start
        log.info(
            f"  {product} {year}: reduced month {month:2d}/12 "
            f"→ {len(unique_keys):,} grid points  elapsed={_fmt_duration(elapsed_agg)}"
        )

    if not has_any:
        log.warning(f"  {product} {year}: no valid monthly means could be computed.")
        return None

    all_key_arrays = [month_keys[m] for m in range(1, 13) if month_keys[m] is not None]
    if not all_key_arrays:
        log.warning(f"  {product} {year}: no valid NDVI pixels after masking fill values.")
        return None

    unique_grid_keys = np.unique(np.concatenate(all_key_arrays))
    n_pts = len(unique_grid_keys)
    log.info(f"  {product} {year}: building output DataFrame ({n_pts:,} unique grid points) ...")

    lats_out = (unique_grid_keys // 4_000_000).astype(np.float32) / SCALE
    lons_out = (unique_grid_keys  % 4_000_000).astype(np.float32) / SCALE
    lons_out = np.where(lons_out > 180.0, lons_out - 400.0, lons_out)

    rows_out: Dict[str, np.ndarray] = {"Lat": lats_out, "Lon": lons_out}
    months_covered = 0
    for month in range(1, 13):
        col = f"{month - 1}_month{month}_meanNDVI"
        if month_keys[month] is None:
            rows_out[col] = np.full(n_pts, np.nan, dtype=np.float32)
            continue
        idx = np.searchsorted(unique_grid_keys, month_keys[month])
        ndvi_col = np.full(n_pts, np.nan, dtype=np.float32)
        ndvi_col[idx] = (month_sums[month] / month_cnts[month]).astype(np.float32)
        rows_out[col] = ndvi_col
        months_covered += 1

    df = pd.DataFrame(rows_out)
    monthly_cols = [f"{m - 1}_month{m}_meanNDVI" for m in range(1, 13)]
    df = df[df[monthly_cols].notna().any(axis=1)].reset_index(drop=True)
    log.info(
        f"  {product} {year}: built monthly NDVI grid with {len(df):,} valid points "
        f"({months_covered} months covered)."
    )
    return df


def build_ndvi_from_modis(
    years: List[int],
    bbox: Tuple[float, float, float, float],
    ndvi_dir: Path,
    modis_cache_dir: Path,
    write_csv: bool = True,
    workers: int = 4,
    required_tiles: Optional[set] = None,
    product: str = "MOD13Q1",
    plot_lats: Optional[np.ndarray] = None,
    plot_lons: Optional[np.ndarray] = None,
) -> Dict[int, pd.DataFrame]:
    """Download MODIS and build monthly NDVI DataFrames for each year in *years*.

    When *plot_lats* / *plot_lons* are supplied, only pixels at those locations
    are sampled from the HDF tiles (fast, memory-efficient path).  Otherwise the
    full raster grid is returned (legacy path).

    Downloads are always sequential (earthaccess is not thread-safe across years).
    HDF processing within each year is parallelised with ``workers`` threads.

    When *write_csv* is True (the default), each result is also saved as
    ``NDVI_100m_{year}.csv`` in *ndvi_dir* so it can be used as a pre-computed file
    on future runs.

    Returns a mapping {year: modis_df} where each DataFrame has the raw
    ``{i}_month{m}_meanNDVI`` column layout produced by ``_hdf_to_monthly_ndvi``.
    """
    import threading
    results: Dict[int, pd.DataFrame] = {}
    results_lock = threading.Lock()

    sorted_years = sorted(years)
    n_years = len(sorted_years)
    year_start = time.time()

    # ------------------------------------------------------------------ download pass (sequential)
    # earthaccess.download is not safe to call concurrently; download all years first.
    hdf_by_year: Dict[int, List[Path]] = {}
    for i, year in enumerate(sorted_years, 1):
        elapsed = time.time() - year_start
        log.info(
            f"  [{i}/{n_years}] Year {year}: downloading granules ...  "
            f"(elapsed={_fmt_duration(elapsed)})"
        )
        hdf_files = _download_modis_year(year, bbox, modis_cache_dir, required_tiles=required_tiles, product=product)
        if not hdf_files:
            log.warning(f"  Year {year}: no HDF files found, skipping.")
        else:
            hdf_by_year[year] = hdf_files

    if not hdf_by_year:
        return results

    # ------------------------------------------------------------------ processing pass (parallel across years)
    # Each year's HDF list is processed independently; run years concurrently.
    # _hdf_to_monthly_ndvi itself spawns ``workers`` threads for per-file I/O.
    # To avoid oversubscription, limit year-level concurrency to min(years, workers).
    year_workers = min(len(hdf_by_year), max(1, workers // 2))
    log.info(
        f"  Processing {len(hdf_by_year)} year(s) with "
        f"year_workers={year_workers}, hdf_workers={workers} ..."
    )
    proc_start = time.time()
    done_years = 0

    def _process_year(year: int) -> Tuple[int, Optional[pd.DataFrame]]:
        df = _hdf_to_monthly_ndvi(
            hdf_by_year[year], year, workers=workers, product=product,
            plot_lats=plot_lats, plot_lons=plot_lons,
        )
        return year, df

    with ThreadPoolExecutor(max_workers=year_workers) as pool:
        year_futures = {pool.submit(_process_year, y): y for y in hdf_by_year}
        for fut in as_completed(year_futures):
            year = year_futures[fut]
            try:
                year, df = fut.result()
            except Exception as exc:
                log.warning(f"  Year {year}: processing error: {exc}")
                df = None

            done_years += 1
            elapsed = time.time() - proc_start
            pct = 100 * done_years / len(hdf_by_year)
            log.info(
                f"  Year {year} processed  [{done_years}/{len(hdf_by_year)}  {pct:.0f}%  "
                f"elapsed={_fmt_duration(elapsed)}]"
            )

            if df is None or df.empty:
                log.warning(f"  MODIS processing yielded no data for year {year}.")
                continue

            if write_csv:
                out_csv = ndvi_dir / f"NDVI_100m_{year}.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_csv, index=False, float_format="%.6f")
                log.info(f"  Saved {out_csv.name}  ({len(df):,} rows)")

            with results_lock:
                results[year] = df

    return results


# ---------------------------------------------------------------------------
# Main augmentation pipeline
# ---------------------------------------------------------------------------


def main(
    input_csv: str,
    output_csv: str,
    ndvi_dir: str,
    download: bool = False,
    modis_cache_dir: Optional[str] = None,
    workers: int = 4,
    modis_product: str = "MOD13A3",
) -> pd.DataFrame:
    t0 = time.time()
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    ndvi_path = Path(ndvi_dir)

    if modis_product not in MODIS_PRODUCTS:
        raise ValueError(
            f"Unknown MODIS product '{modis_product}'. "
            f"Choose from: {list(MODIS_PRODUCTS.keys())}"
        )

    log.info("=" * 60)
    log.info("Augmenting yield Africa dataset with NDVI features")
    log.info(f"  input_csv : {input_path}")
    log.info(f"  output_csv: {output_path}")
    log.info(f"  ndvi_dir  : {ndvi_path}")
    if download:
        cache_path = Path(modis_cache_dir) if modis_cache_dir else ndvi_path / "modis_hdf"
        log.info(f"  modis_cache_dir: {cache_path}  [MODIS download enabled]")
        log.info(f"  modis_product : {modis_product}")
        log.info(f"  workers   : {workers}")
    log.info("=" * 60)

    # ------------------------------------------------------------------ load CSV
    log.info(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df)} rows, {df.columns.size} columns")

    df["year"] = df["year"].astype(int)

    # ------------------------------------------------------------------ optional MODIS download
    # modis_data holds in-memory MODIS DataFrames keyed by year for gap-filling.
    modis_data: Dict[int, pd.DataFrame] = {}

    if download:
        _earthaccess_login()

        cache_path = Path(modis_cache_dir) if modis_cache_dir else ndvi_path / "modis_hdf"
        ndvi_path.mkdir(parents=True, exist_ok=True)

        # Compute bounding box from the CSV (add a small buffer).
        buf = 0.5
        bbox = (
            float(df["lon"].min() - buf),
            float(df["lat"].min() - buf),
            float(df["lon"].max() + buf),
            float(df["lat"].max() + buf),
        )
        log.info(f"Dataset bounding box (with {buf}° buffer): {bbox}")

        # Derive the set of MODIS tiles that actually contain plot locations so
        # the download is restricted to those tiles only.  Africa spans ~44 MODIS
        # tiles but many cover ocean or desert areas with no plots.
        required_tiles = _required_modis_tiles(df["lat"].values, df["lon"].values)
        log.info(
            f"Required MODIS tiles ({len(required_tiles)}): {sorted(required_tiles)}"
        )

        # Unique plot locations — passed to the HDF sampler so only these pixels
        # are extracted from each tile instead of building the full raster grid.
        unique_locs = df[["lat", "lon"]].drop_duplicates()
        plot_lats_arr = unique_locs["lat"].values.astype(np.float32)
        plot_lons_arr = unique_locs["lon"].values.astype(np.float32)
        log.info(f"Unique plot locations for MODIS sampling: {len(plot_lats_arr):,}")

        existing_files = discover_ndvi_files(ndvi_path)
        years_in_csv = sorted(df["year"].unique())

        # Years with no CSV at all — generate from MODIS and write to disk.
        years_no_csv = [y for y in years_in_csv if y not in existing_files]
        if years_no_csv:
            log.info(f"Fetching MODIS for {len(years_no_csv)} year(s) with no NDVI CSV: {years_no_csv}")
            new_data = build_ndvi_from_modis(
                years_no_csv, bbox, ndvi_path, cache_path,
                write_csv=True, workers=workers, required_tiles=required_tiles,
                product=modis_product,
                plot_lats=plot_lats_arr, plot_lons=plot_lons_arr,
            )
            modis_data.update(new_data)
        else:
            log.info("All years already have NDVI CSV files.")

        # Years with a CSV that contains NaN months — fetch MODIS for gap-filling.
        # We probe each existing file quickly without building a KDTree yet.
        years_with_gaps: List[int] = []
        for year, path in existing_files.items():
            if year not in years_in_csv:
                continue
            try:
                probe = pd.read_csv(path, low_memory=False)
                month_raw_cols = [c for c in probe.columns if NDVI_COL_PATTERN.match(c)]
                for col in month_raw_cols:
                    probe[col] = pd.to_numeric(probe[col], errors="coerce")
                if probe[month_raw_cols].isna().any().any():
                    years_with_gaps.append(year)
            except Exception as exc:
                log.warning(f"  Could not probe {path.name} for gaps: {exc}")

        years_need_modis = [y for y in years_with_gaps if y not in modis_data]
        if years_need_modis:
            log.info(
                f"Fetching MODIS for {len(years_need_modis)} year(s) with incomplete NDVI CSV: "
                f"{years_need_modis}"
            )
            # Don't write new CSVs — originals are kept; gaps filled in memory.
            gap_data = build_ndvi_from_modis(
                years_need_modis, bbox, ndvi_path, cache_path,
                write_csv=False, workers=workers, required_tiles=required_tiles,
                product=modis_product,
                plot_lats=plot_lats_arr, plot_lons=plot_lons_arr,
            )
            modis_data.update(gap_data)
        elif years_with_gaps:
            log.info(f"MODIS data already available for gapped year(s): {years_with_gaps}")
        else:
            log.info("No incomplete NDVI CSV files found; no gap-filling needed.")

    # ------------------------------------------------------------------ discover files
    ndvi_files = discover_ndvi_files(ndvi_path)
    if not ndvi_files:
        raise FileNotFoundError(f"No NDVI_100m_*.csv files found in {ndvi_path}")

    years_in_csv = sorted(df["year"].unique())
    years_with_ndvi = sorted(ndvi_files)
    years_missing = [y for y in years_in_csv if y not in ndvi_files]

    log.info(f"Years in CSV       : {years_in_csv}")
    log.info(f"Years with NDVI    : {years_with_ndvi}")
    if years_missing:
        log.warning(f"Years WITHOUT NDVI : {years_missing}  → rows for these years will have NaN features")

    # ------------------------------------------------------------------ load NDVI per year
    ndvi_index: Dict[int, Tuple[KDTree, pd.DataFrame]] = {}
    for year, path in ndvi_files.items():
        if year not in years_in_csv:
            log.info(f"  Skipping {path.name} (year {year} not in CSV)")
            continue

        _, points = load_ndvi_file(path)

        # Gap-fill from MODIS if data is available and the file has missing months.
        if year in modis_data and ndvi_has_gaps(points):
            log.info(f"  Filling gaps in year {year} NDVI from MODIS ...")
            points = fill_ndvi_gaps(points, modis_data[year])

        tree = KDTree(points[["Lat", "Lon"]].values)
        ndvi_index[year] = (tree, points)

    # Determine new feature columns from the first loaded file.
    sample_points = next(iter(ndvi_index.values()))[1]
    new_feat_cols = [c for c in sample_points.columns if c.startswith("feat_ndvi_")]
    log.info(f"New feature columns ({len(new_feat_cols)}): {new_feat_cols}")

    # ------------------------------------------------------------------ row-wise lookup
    n_total = len(df)
    log.info(f"Looking up NDVI for {n_total} rows ...")
    feat_rows: List[Dict] = []
    n_matched = 0
    n_missing_year = 0
    n_warn_dist = 0

    PROGRESS_INTERVAL = 5  # seconds between progress lines
    BAR_WIDTH = 20
    lookup_start = time.time()
    last_progress_time = lookup_start

    def _ndvi_progress(done: int, force: bool = False) -> None:
        nonlocal last_progress_time
        now = time.time()
        if not force and (now - last_progress_time) < PROGRESS_INTERVAL:
            return
        last_progress_time = now
        elapsed = now - lookup_start
        pct = 100 * done / n_total if n_total else 100
        filled = int(BAR_WIDTH * done / n_total) if n_total else BAR_WIDTH
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        rate = done / elapsed if elapsed > 0 else 0
        eta_str = _fmt_duration((n_total - done) / rate) if rate > 0 and done < n_total else "—"
        log.info(
            f"  [{bar}] {done:5d}/{n_total}  {pct:5.1f}%  "
            f"matched={n_matched}  no_ndvi={n_missing_year}  warn_dist={n_warn_dist}  "
            f"elapsed={_fmt_duration(elapsed)}  ETA={eta_str}"
        )

    for done, (i, row) in enumerate(df.iterrows(), start=1):
        year = int(row["year"])

        if year not in ndvi_index:
            feat_rows.append({c: np.nan for c in new_feat_cols})
            n_missing_year += 1
            _ndvi_progress(done)
            continue

        tree, points = ndvi_index[year]
        ndvi_row, dist = lookup_ndvi(row["lat"], row["lon"], tree, points)

        if dist > DISTANCE_WARN_DEG:
            log.warning(
                f"  Row {i}: large nearest-neighbour distance {dist:.4f}° "
                f"for ({row['lat']:.4f}, {row['lon']:.4f}) year={year}"
            )
            n_warn_dist += 1

        feat_rows.append({c: ndvi_row.get(c, np.nan) for c in new_feat_cols})
        n_matched += 1
        _ndvi_progress(done)

    _ndvi_progress(n_total, force=True)

    feat_df = pd.DataFrame(feat_rows, index=df.index)

    # ------------------------------------------------------------------ merge
    merged = pd.concat([df, feat_df], axis=1)
    n_rows_any_nan = merged[new_feat_cols].isna().any(axis=1).sum()
    available = [c for c in new_feat_cols if c in merged.columns]

    # Feature normalisation is handled at training time by TabularEncoder.
    # Any NaN (rows with no matching NDVI year) are left as-is.

    # ------------------------------------------------------------------ save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, float_format="%.7f")

    elapsed = time.time() - t0

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input CSV              : {input_path}  ({len(df)} rows)")
    print(f"  Output CSV             : {output_path}")
    print(f"  NDVI directory         : {ndvi_path}")
    print(f"  Years in CSV           : {years_in_csv}")
    print(f"  Years with NDVI data   : {years_with_ndvi}")
    print(f"  Years missing NDVI     : {years_missing if years_missing else 'none'}")
    years_gap_filled = sorted(y for y in modis_data if y in ndvi_index)
    print(f"  Years gap-filled (MODIS): {years_gap_filled if years_gap_filled else 'none'}")
    print(f"  Rows matched           : {n_matched}/{len(df)}")
    print(f"  Rows missing (no file) : {n_missing_year}")
    print(f"  Rows with large dist   : {n_warn_dist}  (threshold: {DISTANCE_WARN_DEG}°)")
    print(f"  Rows with any NaN feat : {n_rows_any_nan}")
    print(f"  Original columns       : {df.columns.size}")
    print(f"  New NDVI columns       : {len(available)}")
    print(f"  Total columns          : {merged.columns.size}")
    print(f"  Elapsed time           : {elapsed:.1f}s")
    print(f"  New feature names      :")
    for col in sorted(available):
        print(f"    {col}")
    print("=" * 60)
    print(f"Done. Augmented CSV saved to: {output_path}")

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description=(
            "Augment the yield Africa model-ready dataset with year-specific "
            "NDVI features derived from pre-computed monthly mean NDVI files, "
            "optionally downloading missing years from MODIS MOD13Q1 via NASA EarthData."
        )
    )
    ap.add_argument(
        "--input_csv",
        required=True,
        help="Path to model_ready_yield_africa.csv",
    )
    ap.add_argument(
        "--output_csv",
        required=True,
        help="Output path for the augmented CSV",
    )
    ap.add_argument(
        "--ndvi_dir",
        default="/Volumes/data_and_models_2/aether/data/cache/ndvi",
        help="Directory containing (or to receive) NDVI_100m_{year}.csv files",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help=(
            "Download missing MODIS MOD13Q1 granules from NASA EarthData for years "
            "that lack a pre-computed NDVI CSV. Requires EARTHDATA_USERNAME and "
            "EARTHDATA_PASSWORD environment variables."
        ),
    )
    ap.add_argument(
        "--modis_cache_dir",
        default=None,
        help=(
            "Directory for caching raw MODIS HDF files "
            "(default: <ndvi_dir>/modis_hdf). "
            "Re-running will skip already-downloaded granules."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of threads for parallel HDF file reading (default: 4)",
    )
    ap.add_argument(
        "--modis_product",
        choices=list(MODIS_PRODUCTS.keys()),
        default="MOD13A3",
        help=(
            "MODIS product to download when --download is set. "
            "'MOD13Q1' is a 16-day composite at 250 m (~44 GB/year for 9 tiles); "
            "'MOD13A3' is a monthly composite at 1 km (~1.3 GB/year for 9 tiles, ~34× smaller). "
            "Default: MOD13A3 (recommended to minimise download size)."
        ),
    )
    args = ap.parse_args()

    main(
        args.input_csv,
        args.output_csv,
        args.ndvi_dir,
        download=args.download,
        modis_cache_dir=args.modis_cache_dir,
        workers=args.workers,
        modis_product=args.modis_product,
    )
