import concurrent.futures
import math
import os
import threading

import numpy as np
from affine import Affine

# Serialises concurrent reads/writes to the per-directory meta.csv log file.
_meta_csv_lock = threading.Lock()
import pandas as pd
import rasterio
from geotessera import GeoTessera
from rasterio import MemoryFile
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.data_preprocessing.crs_utils import (
    crs_to_pixel_coords,
    get_point_utm_crs,
    point_reprojection,
)


class PartialTileError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NoTileError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def reproject_dataset(src_raster: MemoryFile, dst_crs: str) -> MemoryFile:
    """Reprojects Memory file if it's not in dst_crs.

    :param src_raster: Raster file to reproject.
    :param dst_crs: CRS to reproject.
    """
    dst_crs = CRS.from_user_input(dst_crs)
    if src_raster.crs == dst_crs:
        return src_raster, None

    # Reprojection dim
    transform, width, height = calculate_default_transform(
        src_raster.crs, dst_crs, src_raster.width, src_raster.height, *src_raster.bounds
    )

    # Update metadata
    metadata = src_raster.meta.copy()
    metadata.update(
        crs=dst_crs,
        transform=transform,
        width=width,
        height=height,
    )

    memfile = MemoryFile()
    dst = memfile.open(**metadata)
    for i in range(1, src_raster.count + 1):
        reproject(
            source=rasterio.band(src_raster, i),
            destination=rasterio.band(dst, i),
            src_transform=src_raster.transform,
            src_crs=src_raster.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return dst, memfile


def get_tiles(lat_center, lon_center, half_size_m=75, year=2024):
    """Find all 0.1deg tiles (referenced at 0.05deg) that overlap AOI."""

    # Convert half-size from meters to degrees (approximate)
    half_lat_deg = half_size_m / 111320.0
    half_lon_deg = half_size_m / (111320.0 * np.cos(np.radians(lat_center)))

    # AOI bounds
    lat_min = lat_center - half_lat_deg
    lat_max = lat_center + half_lat_deg
    lon_min = lon_center - half_lon_deg
    lon_max = lon_center + half_lon_deg

    tile_size = 0.1

    # Find tile indices overlapping the AOI
    i_min = int(np.floor(lat_min / tile_size))
    i_max = int(np.floor(lat_max / tile_size))
    j_min = int(np.floor(lon_min / tile_size))
    j_max = int(np.floor(lon_max / tile_size))

    tiles = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            ref_lat = i * tile_size + 0.05  # tile reference (center)
            ref_lon = j * tile_size + 0.05
            tiles.append((year, round(ref_lon, 10), round(ref_lat, 10)))

    return tiles


def get_tessera_embeds(
    lon: float,
    lat: float,
    name_loc: str,
    year: int,
    save_dir: str,
    tile_size: int,
    tessera_con: GeoTessera | None,
    padding: int = 100,
) -> None:
    """Obtain tessera embedding tile with specified size for a given coordinates.

    :param lon: longitude in WGS84
    :param lat: latitude in WGS84
    :param name_loc: data entry id used to reference back to model ready csv
    :param year: year of the embeddings
    :param save_dir: data directory to save embeddings
    :param tile_size: tile size in pixels
    :param tessera_con: GeoTessera instance
    :param padding: how many meters to pad initial bbox, fixes some inconsistencies when mosaicing
    :return: None
    """

    # Skip if tile exists
    embed_tile_name = os.path.join(save_dir, f"tessera_{name_loc}.npy")
    if os.path.exists(embed_tile_name):
        return

    # Local utm projection
    utm_crs = get_point_utm_crs(lon, lat)
    lon_utm, lat_utm = point_reprojection(lon, lat, "EPSG:4326", utm_crs)

    # Request to tessera
    radius = math.ceil(tile_size / 2) + padding
    tiles_to_fetch = get_tiles(
        lat_center=lat, lon_center=lon, half_size_m=radius * 10, year=int(year)
    )

    # Mosaic returned tiles for the bbox
    tiles = []
    memfiles = []

    for _, _, _, embedding, crs, transform in tessera_con.fetch_embeddings(tiles_to_fetch):
        memfile = MemoryFile()
        memfiles.append(memfile)

        tile = memfile.open(
            driver="GTiff",
            height=embedding.shape[0],
            width=embedding.shape[1],
            count=embedding.shape[2],
            dtype=embedding.dtype,
            crs=crs,
            transform=transform,
        )

        for c in range(embedding.shape[2]):
            tile.write(embedding[:, :, c], c + 1)

        reproject_tile, reproject_memfile = reproject_dataset(tile, utm_crs)
        tiles.append(reproject_tile)
        if reproject_memfile:
            memfiles.append(reproject_memfile)

    if len(tiles) == 0:
        for mf in memfiles:
            mf.close()
        raise NoTileError(f"No tiles found for {name_loc}")  # if no tiles, add to skipped.txt

    mosaic, mosaic_transform = merge(tiles)
    mosaic = mosaic.transpose(1, 2, 0)

    for tile in tiles:
        tile.close()
    for mf in memfiles:
        mf.close()

    # Crop patch tile
    c, r = crs_to_pixel_coords(lon_utm, lat_utm, mosaic_transform)
    half = tile_size // 2
    row_min = r - half
    row_max = r + half
    col_min = c - half
    col_max = c + half

    if row_min < 0 or row_max < 0 or col_min < 0 or col_max < 0:
        # retry with bigger padding
        if padding > 500:
            raise NoTileError(f"Padding {padding} > 500")
        get_tessera_embeds(
            lon, lat, name_loc, year, save_dir, tile_size, tessera_con, padding=padding + 100
        )

    crop = mosaic[row_min:row_max, col_min:col_max, :]
    if not crop.shape == (tile_size, tile_size, 128):
        if crop.min() == 0.0 and crop.max() == 0.0:
            raise NoTileError(f"No tiles found for {name_loc}")
        raise PartialTileError(f"Crop {name_loc}, size is {crop.shape}")

    if crop.min() == 0.0 and crop.max() == 0.0:
        raise NoTileError(f"Crop {name_loc} has embeddings of 0.0s with tiles: {tiles_to_fetch}")

    # Save array
    os.makedirs(save_dir, exist_ok=True)
    np.save(embed_tile_name, crop)
    print(f"Array saved as {embed_tile_name}")

    # Temp save tif too
    crop_transform = mosaic_transform * Affine.translation(col_min, row_min)
    height, width, channels = crop.shape

    with rasterio.open(
        embed_tile_name[:-4] + ".tif",
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=channels,
        dtype=crop.dtype,
        crs=utm_crs,
        transform=crop_transform,
    ) as dst:
        for i in range(channels):
            dst.write(crop[:, :, i], i + 1)
    print(f"tif saved to {embed_tile_name[:-4]}.tif")

    # Log its metadata
    meta_df = pd.DataFrame(
        {"id": [name_loc], "year": [year], "lon": [lon], "lat": [lat], "crs": [utm_crs]}
    )

    meta_file = f"{save_dir}/meta.csv"

    with _meta_csv_lock:
        try:
            if os.path.exists(meta_file):
                meta_df = pd.concat([meta_df, pd.read_csv(meta_file)], ignore_index=True)
            meta_df.to_csv(meta_file, index=False)
            print(f"Meta data logged to {meta_file}")
        except Exception as e:
            print(f"Warning: could not update meta.csv ({e}). Tile was saved successfully.")


def tessera_from_df(
    model_ready_df: pd.DataFrame,
    data_dir: str,
    year: int,
    tile_size: int = 256,
    cache_dir: str = "temp/",
    logs_dir: str = "logs",
) -> None:
    """Obtains Tessera embeddings from a CSV file for each (lon, lat).

    :param model_ready_df: pandas dataframe with model ready rentries
    :param data_dir: path to data directory
    :param year: year for the embeddings
    :param tile_size: tile size in meters
    :param cache_dir: path to cache directory
    :return: None
    """

    # Tessera connection
    cache_dir = os.path.join(cache_dir, "tessera")
    gt = GeoTessera(cache_dir=cache_dir, embeddings_dir=cache_dir, dataset_version="v1")

    # Iter each coord
    n = len(model_ready_df)
    for i, row in model_ready_df.iterrows():
        print(f"{i}/{n}")
        try:
            get_tessera_embeds(row.lon, row.lat, row.name_loc, year, f"{data_dir}/", tile_size, gt)
        except Exception as e:
            if isinstance(e, NoTileError):
                path = os.path.join(logs_dir, "tessera_skipped.txt")
                with open(path, "a") as f:
                    f.write(f"{row.name_loc}\n")
            else:
                print(f"{row.name_loc} did not get embedded because: {e}")


def inspect_np_arr_as_tiff(
    arr: np.ndarray,
    center_lon: float,
    center_lat: float,
    pixel_size: int,
    file_path: str,
    utm_crs: str = None,
) -> None:
    """Saves numpy array as tiff for inspection in e.g. qgis.

    :param arr: numpy array
    :param center_lon: center longitude in WGS84
    :param center_lat: center latitude in WGS84
    :param pixel_size: pixel size in meters
    :param file_path: file path to save tiff
    :param utm_crs: local UTM crs
    :return: None
    """

    # Get local utm crs
    if utm_crs is None:
        utm_crs = get_point_utm_crs(center_lon, center_lat)

    # Convert coords to utm
    center_lon, center_lat = point_reprojection(center_lon, center_lat, "EPSG:4326", utm_crs)

    # Handle shape
    if arr.ndim == 2:
        height, width = arr.shape
        count = 1
        arr_to_write = arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        height, width, count = arr.shape
        arr_to_write = arr.transpose(2, 0, 1)  # (bands, H, W)
    else:
        raise ValueError("Array must be 2D or 3D")

    # Compute top-left corner coordinates
    top_left_x = center_lon - (width / 2) * pixel_size
    top_left_y = center_lat + (height / 2) * pixel_size
    transform = from_origin(top_left_x, top_left_y, pixel_size, pixel_size)

    # Save to GeoTIFF
    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=arr.dtype,
        crs=utm_crs,
        transform=transform,
    ) as dst:
        for i in range(0, count):
            dst.write(arr_to_write[i], i + 1)

    print(f"Tiff version of np array saved to {file_path}")


if __name__ == "__main__":
    # os.chdir('../..')

    print(os.getcwd())

    # df = pd.read_csv("data/heat_guatemala/model_ready_heat_guatemala.csv")
    # df = pd.read_csv("/lustre/backup/SHARED/AIN/aether/data/s2bms/model_ready_s2bms.csv")
    df = pd.read_csv("data/s2bms/model_ready_s2bms.csv")
    # df.sort_values(by="name_loc", inplace=True, ascending=False)
    if os.path.exists("logs/tessera_skipped.txt"):
        with open(os.path.join("logs", "tessera_skipped.txt")) as f:
            skipped = set(f.read().splitlines())
        df = df[~df.name_loc.isin(skipped)]
    # df.sort_values('name_loc', ascending=False, inplace=True)

    tessera_from_df(
        df,
        "data/s2bms/eo/tessera",
        year=2024,
        tile_size=256,
        cache_dir="data/cache",
    )
