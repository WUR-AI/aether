"""Augment the crop yield Africa model-ready dataset with year-specific AgERA5/CHIRPS
climate features.

For each unique (lat, lon, year) in the model-ready CSV this script fetches daily
climate data from the AgERA5 weather API (https://agera5.containers.wurnet.nl) and
computes seasonal aggregates for the four standard meteorological seasons plus a
dedicated growing-season window (March–October).

These in-season features capture inter-annual climate variability that the existing
static long-term averages (MAP, MAT, CMD, etc.) cannot represent — a key gap
identified in the AI readiness analysis.

New columns added (feat_agera5_ prefix):
    feat_agera5_{var}_{season}   where
        var    ∈ {tmax, tmin, tavg, vp, ws, prec, rad, snow}
        season ∈ {mam, jja, son, djf, grow}   (grow = March–October)
    Aggregation: sum for prec/rad; mean for temperature, vp, ws, snow.
    Additional:
        feat_agera5_gdd10_grow   — growing-degree-days (base 10°C), March–October
        feat_agera5_wetdays_grow — days with precipitation > 1 mm, March–October

Note on DJF: this script uses months 12, 1, 2 of the *same* calendar year (i.e. the
December from that harvest year rather than the prior year) to avoid the need for an
extra API call. This differs from the standard meteorological convention but is
consistent with how the existing static seasonal features were derived.

Rows for which the API fetch fails or has not yet been cached are written with NaN in
all AgERA5 feature columns.  A sentinel column ``agera5_fetched`` (1 = real data,
0 = NaN / not yet fetched) is added so downstream tools (--complete_only in the
comparison script) can identify incomplete rows without ambiguity.  Imputation of NaN
values is intentionally deferred to training time via TabularEncoder, which fits on
training rows only and avoids leaking val/test statistics into the feature values.

Usage:
    python src/data_preprocessing/yield_africa_augment_agera5.py \\
        --input_csv  data/yield_africa/model_ready_yield_africa.csv \\
        --output_csv data/yield_africa/model_ready_yield_africa_agera5.csv \\
        --cache_dir  data/cache/agera5 \\
        --workers    4

Resume / incremental runs: raw API responses are cached as JSON files under
``cache_dir``. Re-running the script will skip already-cached triplets and only
fetch missing ones.
"""

import argparse
import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_ENDPOINT_CHIRPS = "/api/v1/get_agera5_chirps"
API_ENDPOINT_AGERA5 = "/api/v1/get_agera5"

# CHIRPS precipitation is more accurate over tropical Africa; use it from 1981 onward.
CHIRPS_START_YEAR = 1981

# Months belonging to each season / window.
# DJF uses month 12 of the same calendar year (see module docstring).
SEASONS: Dict[str, List[int]] = {
    "djf": [12, 1, 2],
    "mam": [3, 4, 5],
    "jja": [6, 7, 8],
    "son": [9, 10, 11],
    "grow": list(range(3, 11)),  # March–October
}

# Mapping from AgERA5 API variable names to short names used in column names.
VARIABLES: Dict[str, str] = {
    "temperature_max": "tmax",
    "temperature_min": "tmin",
    "temperature_avg": "tavg",
    "vapourpressure": "vp",
    "windspeed": "ws",
    "precipitation": "prec",
    "radiation": "rad",
    "snowdepth": "snow",
}

# Variables aggregated as seasonal *sum*; all others use seasonal *mean*.
SUM_VARIABLES = {"prec", "rad"}

# ---------------------------------------------------------------------------
# HTTP session with automatic retry
# ---------------------------------------------------------------------------


def _make_session(retries: int = 5, backoff: float = 1.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_file(cache_dir: Path, lat: float, lon: float, year: int) -> Path:
    """Return path for a cached JSON response for the given (lat, lon, year)."""
    lat_str = f"{lat:.4f}".replace("-", "m")
    lon_str = f"{lon:.4f}".replace("-", "m")
    return cache_dir / f"agera5_{lat_str}_{lon_str}_{year}.json"


# ---------------------------------------------------------------------------
# Fetch & parse
# ---------------------------------------------------------------------------


def fetch_year(
    session: requests.Session,
    api_base: str,
    lat: float,
    lon: float,
    year: int,
    cache_dir: Path,
    timeout: int = 120,
    cache_only: bool = False,
) -> Optional[pd.DataFrame]:
    """Fetch daily climate data for a full calendar year and return a DataFrame.

    Returns ``None`` if the request fails or the response cannot be parsed.
    Columns present: ``_month`` plus one column per recognised API variable.

    Strategy:
    - For years ≥ 1981 try the CHIRPS endpoint first (better precipitation over
      tropical Africa); fall back to plain AgERA5 on any HTTP error.
    - Responses are cached as JSON so subsequent runs skip already-fetched triplets.
    - If ``cache_only`` is True, returns ``None`` for any triplet not already cached
      (no API calls are made).
    """
    cached = _cache_file(cache_dir, lat, lon, year)

    if cached.exists():
        log.debug(f"Cache hit : {cached.name}")
        with open(cached) as fh:
            raw = json.load(fh)
    elif cache_only:
        return None
    else:
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "startdate": f"{year}0101",
            "enddate": f"{year}1231",
        }
        raw = None

        # Try CHIRPS first (1981+), fall back to plain AgERA5 on any HTTP error.
        endpoints = (
            [API_ENDPOINT_CHIRPS, API_ENDPOINT_AGERA5]
            if year >= CHIRPS_START_YEAR
            else [API_ENDPOINT_AGERA5]
        )
        for endpoint in endpoints:
            url = f"{api_base.rstrip('/')}{endpoint}"
            try:
                resp = session.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                raw = resp.json()
                break
            except requests.HTTPError as exc:
                log.warning(
                    f"HTTP {exc.response.status_code} from {endpoint} "
                    f"({lat:.4f}, {lon:.4f}, {year})"
                    + (f" — retrying with {API_ENDPOINT_AGERA5}" if endpoint == API_ENDPOINT_CHIRPS else "")
                )
            except Exception as exc:
                log.warning(f"Fetch failed ({lat:.4f}, {lon:.4f}, {year}): {exc}")
                break

        if raw is None:
            return None

        with open(cached, "w") as fh:
            json.dump(raw, fh)
        log.info(f"Downloaded: {cached.name}")

    # Parse: the API returns {"location_info": ..., "weather_variables": [...], "info": ...}
    # where weather_variables is a list of daily records.  Also handle a bare list or
    # other dict-with-data-key shapes for robustness.
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = None
        for key in ("weather_variables", "data", "results", "records", "values"):
            if key in raw and isinstance(raw[key], list):
                records = raw[key]
                break
        if records is None:
            log.warning(f"Unexpected response structure ({lat:.4f}, {lon:.4f}, {year}): {list(raw)[:5]}")
            return None
    else:
        log.warning(f"Unrecognised response type ({lat:.4f}, {lon:.4f}, {year}): {type(raw)}")
        return None

    if not records:
        return None

    df = pd.DataFrame(records)

    # Find the date column — the API uses "day" with ISO datetime strings
    # ("2020-01-01T00:00:00"), but accept other common names too.
    date_col = next(
        (c for c in df.columns if c.lower() in {"day", "date", "time", "datetime"}),
        None,
    )
    if date_col is None:
        log.warning(f"No date column found ({lat:.4f}, {lon:.4f}, {year}); columns: {list(df.columns)}")
        return None

    # pd.to_datetime handles both "20200101" and "2020-01-01T00:00:00" automatically.
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_month"] = df["_date"].dt.month

    # Normalise variable column names (case/hyphen/underscore variations).
    rename: Dict[str, str] = {}
    for api_var, short_var in VARIABLES.items():
        for col in df.columns:
            if col.lower().replace("-", "_") == api_var.lower():
                rename[col] = short_var
                break
    df = df.rename(columns=rename)

    keep = ["_month"] + [v for v in VARIABLES.values() if v in df.columns]
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Feature computation from daily data
# ---------------------------------------------------------------------------


def compute_features(daily: pd.DataFrame) -> Dict[str, float]:
    """Compute seasonal aggregate features from a daily-data DataFrame."""
    feats: Dict[str, float] = {}

    for season, months in SEASONS.items():
        sub = daily[daily["_month"].isin(months)]
        for short_var in VARIABLES.values():
            if short_var not in sub.columns:
                continue
            col = f"feat_agera5_{short_var}_{season}"
            vals = sub[short_var].dropna()
            if vals.empty:
                feats[col] = np.nan
            elif short_var in SUM_VARIABLES:
                feats[col] = float(vals.sum())
            else:
                feats[col] = float(vals.mean())

    # Growing-degree-days above 10 °C (March–October).
    grow_mask = daily["_month"].isin(SEASONS["grow"])
    if "tavg" in daily.columns:
        gdd_vals = daily.loc[grow_mask, "tavg"].dropna()
        gdd = np.maximum(0.0, gdd_vals - 10.0)
        feats["feat_agera5_gdd10_grow"] = float(gdd.sum()) if not gdd.empty else np.nan
    else:
        feats["feat_agera5_gdd10_grow"] = np.nan

    # Wet days in growing season (precipitation > 1 mm/day).
    if "prec" in daily.columns:
        prec_grow = daily.loc[grow_mask, "prec"].dropna()
        feats["feat_agera5_wetdays_grow"] = float((prec_grow > 1.0).sum())
    else:
        feats["feat_agera5_wetdays_grow"] = np.nan

    return feats


# ---------------------------------------------------------------------------
# Parallel fetch driver
# ---------------------------------------------------------------------------


def build_agera5_feature_table(
    df: pd.DataFrame,
    api_base: str,
    cache_dir: Path,
    workers: int,
    timeout: int,
    cache_only: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of AgERA5 features indexed by (lat, lon, year)."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    triplets = list(
        df[["lat", "lon", "year"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    n_total = len(triplets)
    n_cached = sum(1 for lat, lon, year in triplets if _cache_file(cache_dir, lat, lon, year).exists())
    n_to_fetch = 0 if cache_only else (n_total - n_cached)
    print(
        f"Triplets: {n_total} total, {n_cached} already cached, "
        f"{n_to_fetch} to fetch  (workers={workers}, timeout={timeout}s)"
        + ("  [cache_only — API calls disabled]" if cache_only else "") + "\n"
        f"  api_base  : {api_base}\n"
        f"  cache_dir : {cache_dir}"
    )

    results: Dict[Tuple[float, float, int], Dict[str, float]] = {}

    _thread_local = threading.local()

    def _get_session() -> requests.Session:
        """Return a thread-local session, creating it on first use per thread."""
        if not hasattr(_thread_local, "session"):
            _thread_local.session = _make_session()
        return _thread_local.session

    def _worker(triplet: Tuple[float, float, int]):
        lat, lon, year = triplet
        daily = fetch_year(_get_session(), api_base, lat, lon, int(year), cache_dir, timeout, cache_only=cache_only)
        if daily is None or daily.empty:
            return triplet, {}
        return triplet, compute_features(daily)

    # Progress tracking
    HEARTBEAT = 30        # seconds: emit a line even when no future completes
    PROGRESS_INTERVAL = 5 # seconds: minimum gap between progress lines
    done = 0
    done_cached = 0       # triplets served from cache (no API call)
    done_fetched = 0      # triplets actually downloaded
    errors = 0
    last_progress_time = 0.0
    pending: set = set()
    start_time = time.time()

    def _progress_line(elapsed: float, force: bool = False) -> None:
        """Emit a single-line progress update, rate-limited to PROGRESS_INTERVAL."""
        nonlocal last_progress_time
        now = time.time()
        if not force and (now - last_progress_time) < PROGRESS_INTERVAL:
            return
        last_progress_time = now
        pct = 100 * done / n_total if n_total else 100
        bar_width = 20
        filled = int(bar_width * done / n_total) if n_total else bar_width
        bar = "█" * filled + "░" * (bar_width - filled)
        rate = done / elapsed if elapsed > 0 else 0
        eta_str = _fmt_duration((n_total - done) / rate) if rate > 0 and done < n_total else "—"
        log.info(
            f"  [{bar}] {done:4d}/{n_total}  {pct:5.1f}%  "
            f"fetched={done_fetched}  cached={done_cached}  errors={errors}  "
            f"elapsed={_fmt_duration(elapsed)}  ETA={eta_str}"
        )

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Record cache status at submission time — after completion the file
            # exists regardless of whether it was a hit or a fresh download.
            futures = {
                pool.submit(_worker, t): (t, _cache_file(cache_dir, *t).exists())
                for t in triplets
            }
            pending = set(futures)

            while pending:
                finished, pending = wait(pending, timeout=HEARTBEAT, return_when=FIRST_COMPLETED)
                elapsed = time.time() - start_time

                if not finished:
                    # Heartbeat: no future completed within HEARTBEAT seconds
                    _progress_line(elapsed, force=True)
                    continue

                for fut in finished:
                    triplet, was_cached = futures[fut]
                    try:
                        (lat, lon, year), feats = fut.result()
                        results[(lat, lon, year)] = feats
                        if was_cached:
                            done_cached += 1
                        else:
                            done_fetched += 1
                    except Exception as exc:
                        log.error(f"  ERROR fetching {triplet}: {exc}")
                        errors += 1
                    done += 1

                elapsed = time.time() - start_time
                _progress_line(elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted — cancelling queued futures (in-flight requests will finish).")
        for fut in pending:
            fut.cancel()

    # Always emit a final progress line so the terminal shows the end state.
    _progress_line(time.time() - start_time, force=True)

    rows = []
    for (lat, lon, year), feats in results.items():
        rows.append({"lat": lat, "lon": lon, "year": int(year), **feats})

    feat_df = pd.DataFrame(rows)
    print(f"Feature table built: {len(feat_df)} rows, {feat_df.columns.size - 3} new feature columns")
    return feat_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(
    input_csv: str,
    output_csv: str,
    cache_dir: str,
    api_base: str,
    workers: int,
    timeout: int,
    cache_only: bool = False,
) -> pd.DataFrame:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    cache_path = Path(cache_dir)

    print(
        f"Augmenting yield Africa dataset with AgERA5/CHIRPS features\n"
        f"  input_csv  : {input_path}\n"
        f"  output_csv : {output_path}\n"
        f"  cache_dir  : {cache_path}\n"
        f"  workers    : {workers}  timeout: {timeout}s"
        + ("  cache_only: True (no API calls)" if cache_only else "")
    )

    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {df.columns.size} columns")

    # Ensure year is integer for join
    df["year"] = df["year"].astype(int)

    # ------------------------------------------------------------------ fetch
    feat_df = build_agera5_feature_table(df, api_base, cache_path, workers, timeout, cache_only=cache_only)
    feat_df["year"] = feat_df["year"].astype(int)

    meta_cols = {"lat", "lon", "year"}
    new_feat_cols = [c for c in feat_df.columns if c not in meta_cols]

    # --------------------------------------------------------------- merge
    merged = df.merge(feat_df, on=["lat", "lon", "year"], how="left")
    n_rows_missing = merged[new_feat_cols].isna().any(axis=1).sum()
    if n_rows_missing:
        print(
            f"  WARNING: {n_rows_missing} rows have at least one missing AgERA5 feature "
            "(API fetch failed or out of coverage — NaN left in CSV, TabularEncoder will impute)"
        )

    available = [c for c in new_feat_cols if c in merged.columns]

    # Record which rows have real fetched data so the comparison script can
    # identify unfetched rows with --complete_only, and so the TabularEncoder
    # can optionally weight or filter them at training time.
    # 1 = all AgERA5 features were fetched from the API; 0 = at least one is NaN.
    # NaNs are intentionally left in the CSV: imputation is done at training time
    # by TabularEncoder (fitted on training rows only), avoiding data leakage from
    # computing statistics over the full dataset including val/test rows.
    merged["agera5_fetched"] = (~merged[available].isna().any(axis=1)).astype(int)

    # ------------------------------------------------------------------ save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, float_format="%.7f")

    n_fetched = int(merged["agera5_fetched"].sum())
    print(
        f"\n=== Summary ===\n"
        f"  Original columns   : {df.columns.size}\n"
        f"  New AgERA5 columns : {len(available)}\n"
        f"  Total columns      : {merged.columns.size}\n"
        f"  Rows fetched (real): {n_fetched} / {len(merged)} "
        f"({100 * n_fetched / len(merged):.1f}%)  [agera5_fetched=1]\n"
        f"  Rows NaN (no fetch): {n_rows_missing}  [agera5_fetched=0]\n"
        f"  New feature names  :"
    )
    for col in sorted(available):
        print(f"    {col}")
    print(f"\nDone. Augmented CSV saved to: {output_path}")

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description=(
            "Augment the yield Africa model-ready dataset with year-specific "
            "AgERA5/CHIRPS climate features."
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
        "--cache_dir",
        default="data/cache/agera5",
        help="Directory for caching raw API responses (default: data/cache/agera5)",
    )
    ap.add_argument(
        "--api_url",
        default="https://agera5.containers.wurnet.nl",
        help="Base URL of the AgERA5 API (default: https://agera5.containers.wurnet.nl)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel HTTP workers (default: 4)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    ap.add_argument(
        "--cache_only",
        action="store_true",
        help="Only use cached responses — skip API calls for missing triplets (leaves them as NaN)",
    )
    args = ap.parse_args()

    main(
        args.input_csv,
        args.output_csv,
        args.cache_dir,
        args.api_url,
        args.workers,
        args.timeout,
        cache_only=args.cache_only,
    )
