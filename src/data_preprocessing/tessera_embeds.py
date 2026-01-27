import math
import os

import numpy as np
import pandas as pd
import rasterio
from geotessera import GeoTessera
from rasterio import MemoryFile
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.data_preprocessing.crs_utils import (
    create_bbox_with_radius,
    crs_to_pixel_coords,
    get_point_utm_crs,
    point_reprojection,
)


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


def get_tessera_embeds(
    lon: float,
    lat: float,
    name_loc: str,
    year: int,
    save_dir: str,
    tile_size: int,
    tessera_con: GeoTessera | None,
) -> None:
    """Obtain tessera embedding tile with specified size for a given coordinates.

    :param lon: longitude in WGS84
    :param lat: latitude in WGS84
    :param name_loc: data entry id used to reference back to model ready csv
    :param year: year of the embeddings
    :param save_dir: data directory to save embeddings
    :param tile_size: tile size in pixels
    :param tessera_con: GeoTessera instance
    :return: None
    """

    embed_tile_name = os.path.join(save_dir, f"tessera_{name_loc}.npy")
    if os.path.exists(embed_tile_name):
        return

    # Local utm projection
    utm_crs = get_point_utm_crs(lon, lat)
    lon_utm, lat_utm = point_reprojection(lon, lat, "EPSG:4326", utm_crs)

    # Bounding box
    radius = math.ceil(tile_size / 2) + 10
    bbox = create_bbox_with_radius(lon, lat, radius=radius, utm_crs=utm_crs, return_wgs=True)

    # Request to tessera
    tiles_to_fetch = tessera_con.registry.load_blocks_for_region(
        bounds=bbox.bounds, year=int(year)
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

    mosaic, mosaic_transform = merge(tiles)
    mosaic = mosaic.transpose(1, 2, 0)

    for tile in tiles:
        tile.close()
    for mf in memfiles:
        mf.close()

    # Crop patch tile
    col, row = crs_to_pixel_coords(lon_utm, lat_utm, mosaic_transform)
    half = tile_size // 2
    row_min = row - half
    row_max = row + half
    col_min = col - half
    col_max = col + half
    crop = mosaic[row_min:row_max, col_min:col_max, :]

    # Save array
    os.makedirs(save_dir, exist_ok=True)
    np.save(embed_tile_name, crop)
    print(f"Array saved as {embed_tile_name}")

    # Log its metadata
    meta_df = pd.DataFrame(
        {"id": [name_loc], "year": [year], "lon": [lon], "lat": [lat], "crs": [utm_crs]}
    )

    meta_file = f"{save_dir}/meta.csv"

    if os.path.exists(meta_file):
        meta_df = pd.concat([meta_df, pd.read_csv(meta_file)], ignore_index=True)

    meta_df.to_csv(meta_file, index=False)
    print(f"Meta data logged to {meta_file}")


def tessera_from_df(
    model_ready_df: pd.DataFrame,
    data_dir: str,
    year: int,
    tile_size: int = 256,
    cache_dir: str = "temp/",
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
    gt = GeoTessera(cache_dir=cache_dir)

    # Iter each coord
    n = len(model_ready_df)
    for i, row in model_ready_df.iterrows():
        print(f"{i}/{n}")
        # Get tessera embeds
        get_tessera_embeds(row.lon, row.lat, row.name_loc, year, f"{data_dir}/", tile_size, gt)


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
