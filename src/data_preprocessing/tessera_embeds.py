import math

import numpy as np
import pandas as pd
import rasterio
from geotessera import GeoTessera
from rasterio import MemoryFile
from rasterio.merge import merge
from rasterio.transform import from_origin

from src.data_preprocessing.crs_utils import (
    create_bbox_with_radius,
    crs_to_pixel_coords,
    get_point_utm_crs,
    point_reprojection,
)


def get_tessera_embeds(
    lon: float,
    lat: float,
    id: str,
    year: int,
    save_dir: str,
    tile_size: int,
    tessera_con: GeoTessera,
) -> None:
    """Obtain tessera embedding tile with specified size for a given coordinates.

    :param lon: longitude in WGS84
    :param lat: latitude in WGS84
    :param id: data entry id used to reference back to model ready csv
    :param year: year of the embeddings
    :param save_dir: data directory to save embeddings
    :param tile_size: tile size in pixels
    :param tessera_con: GeoTessera instance
    :return: None
    """

    embed_tile_name = f"{save_dir}/tessera_{id}_{year}.npy"
    if os.path.exists(embed_tile_name):
        return

    # Local utm projection
    utm_crs = get_point_utm_crs(lon, lat)
    lon_utm, lat_utm = point_reprojection(lon, lat, "EPSG:4326", utm_crs)

    # Bounding box
    radius = math.ceil(tile_size / 2) + 10
    bbox = create_bbox_with_radius(lon, lat, radius=radius, utm_crs=utm_crs, return_wgs=True)

    # Request to tessera
    tiles_to_fetch = tessera_con.registry.load_blocks_for_region(bounds=bbox.bounds, year=year)

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
        tiles.append(tile)

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
        {"id": [id], "year": [year], "lon": [lon], "lat": [lat], "crs": [utm_crs]}
    )

    meta_file = f"{save_dir}/meta.csv"

    if os.path.exists(meta_file):
        meta_df = pd.concat([meta_df, pd.read_csv(meta_file)], ignore_index=True)

    meta_df.to_csv(meta_file, index=False)
    print(f"Meta data logged to {meta_file}")


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
        dst.write(arr_to_write[0], 1)

    print(f"Tiff version of np array saved to {file_path}")


def tessera_from_df(
    model_ready_csv_path: str, data_dir: str, year: int, tile_size: float = 256
) -> None:
    """Obtains Tessera embeddings from a CSV file for each (lon, lat).

    :param model_ready_csv_path: path to model ready csv file
    :param data_dir: path to data directory
    :param year: year for the embeddings
    :param tile_size: tile size in meters
    :return: None
    """
    if not os.path.exists(model_ready_csv_path):
        raise FileNotFoundError(f"File {model_ready_csv_path} does not exist")

    # Tessera connection
    gt = GeoTessera()

    # Data frame for coords
    df = pd.read_csv(model_ready_csv_path)

    # Iter each coord
    n = len(df)
    for i, row in df.iterrows():
        print(f"{i}/{n}")
        lon, lat = row.lon, row.lat
        id = row.name_loc  # TODO standardise col names across UC

        # Get tessera embeds
        get_tessera_embeds(
            lon, lat, id, year, f"{data_dir}/S2BMS/tessera_{tile_size}", tile_size, gt
        )


if __name__ == "__main__":
    import os

    from dotenv import dotenv_values

    config = dotenv_values("../../.env")
    os.chdir(config.get("PROJECT_ROOT", "../"))

    # Obtain all tiles
    tessera_from_df(
        f'{config.get("DATA_DIR", 'data')}/model_ready/s2bms_presence_with_aux_data.csv',
        data_dir=config.get("DATA_DIR", "data"),
        year=2019,
        tile_size=256,
    )
