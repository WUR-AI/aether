import json
import os

import shapely
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform


def get_point_utm_crs(lon: float, lat: float) -> str:
    """Determine local UTM crs code from given latitude and longitude.

    :param lon: longitude in WGS84
    :param lat: latitude in WGS84
    :return: UTM crs code
    """
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    utm_crs = f"EPSG:{32600 + utm_zone if is_northern else 32700 + utm_zone}"
    return utm_crs


def point_reprojection(lon: float, lat: float, src_crs: str, dst_crs: str):
    """Reproject a point from one to another CRS systems.

    :param lon: longitude
    :param lat: latitude
    :param src_crs: source CRS
    :param dst_crs: destination CRS
    :return: (lon, lat) in reprojection coordinates
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(lon, lat)


def crs_to_pixel_coords(x, y, transform):
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return col, row


def create_bbox_with_radius(
    lon: float, lat: float, radius: float, utm_crs: str = None, return_wgs: bool = False
) -> shapely.geometry.Polygon:
    """Creates a square bounding box of given radius (meters) around lon/lat.

    :param lon: Longitude (EPSG:4326)
    :param lat: Latitude (EPSG:4326)
    :param radius: Radius in meters
    :param utm_crs: Optional EPSG code for UTM CRS (e.g. "EPSG:32633")
    :param return_wgs: If True, returns WGS84 GeoJSON, else UTM Polygon
    """

    # Determine UTM CRS
    utm_crs = utm_crs or get_point_utm_crs(lon, lat)

    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x, y = to_utm.transform(lon, lat)

    # Create bbox in UTM
    square_utm = box(x - radius, y - radius, x + radius, y + radius)

    if return_wgs:
        to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        square_wgs = transform(to_wgs.transform, square_utm)
        return square_wgs

    return square_utm


def geom_to_geojson(geom: shapely.geometry.Polygon) -> dict:
    """Converts shapely geometry to GeoJSON."""
    return {"type": "Feature", "geometry": geom.__geo_interface__, "properties": {}}


def save_geojson(json_dict: dict, filename: str) -> None:
    """Saves GeoJSON to filename based on given dictionary.

    This function overwrites existing GeoJSON.
    :param json_dict: a dictionary with GeoJSON attributes
    :param filename: filename to save GeoJSON
    :return: None
    """
    if os.path.exists(filename):
        print(f"Overwriting {filename}")

    with open(filename, "w") as f:
        json.dump(json_dict, f, indent=2)

    print(f"Saved GeoJSON to {filename}")
