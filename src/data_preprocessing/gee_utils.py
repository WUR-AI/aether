import os, sys
import numpy as np
import ee, geemap
import utm
import shapely

from src.data_preprocessing import data_utils as du

ONLINE_ACCESS_TO_GEE = True 
if ONLINE_ACCESS_TO_GEE:
    gee_api_key = os.environ.get('GEE_API')
    if gee_api_key is None:
        print('WARNING: GEE_API environment variable not set, not using GEE API')
    else:
        ee.Authenticate()
        ee.Initialize(project=gee_api_key)
        geemap.ee_initialize()
else:
    print('WARNING: ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE')

def get_epsg_from_latlon(lat, lon):
    """Get the UTM EPSG code from latitude and longitude.
    https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    """
    utm_result = utm.from_latlon(lat, lon)
    zone_number = utm_result[2]
    hemisphere = '326' if lat >= 0 else '327'
    epsg_code = int(hemisphere + str(zone_number).zfill(2))
    return epsg_code

def create_aoi_from_coord_buffer(coords, buffer_deg=0.01, buffer_m=1000, bool_buffer_in_deg=False):
    """Create an Earth Engine AOI (Geometry) from a coordinate and buffer in meters."""
    point = shapely.geometry.Point(coords)
    if bool_buffer_in_deg:  # not ideal https://gis.stackexchange.com/questions/304914/python-shapely-intersection-with-buffer-in-meter
        print('WARNING: using buffer in degrees, which is not ideal for large latitudes.')
        point = shapely.geometry.Point(coords)
        polygon = point.buffer(buffer_deg, cap_style=3)  ## buffer in degrees
        xy_coords = np.array(polygon.exterior.coords.xy).T 
        aoi = ee.Geometry.Polygon(xy_coords.tolist())
    else:
        point = ee.Geometry.Point(coords)
        aoi = point.buffer(buffer_m).bounds()
    assert aoi is not None
    return aoi

def get_bioclim_from_coord(coords):
    assert ONLINE_ACCESS_TO_GEE, "ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE"
    aoi = create_aoi_from_coord_buffer(coords, buffer_m=1000, bool_buffer_in_deg=False)
    im_gee = ee.Image("WORLDCLIM/V1/BIO").clip(aoi) 
    point = ee.Geometry.Point(coords)  # redefine point for sampling
    values = im_gee.sample(region=point.buffer(1000), scale=1000).first().toDictionary().getInfo()
    return values 

def convert_bioclim_to_units(bioclim_dict):
    assert len(bioclim_dict) == 19, "bioclim_dict should have 19 variables"
    for k in range(1, 20):
        assert f'bio{str(k).zfill(2)}' in bioclim_dict, f'bio{str(k).zfill(2)} not in bioclim_dict'
    _, df_bioclim = du.bioclim_schema()
    for k, v in bioclim_dict.items():
        scale = df_bioclim.loc[df_bioclim['name'] == k, 'scale'].values[0]
        bioclim_dict[k] = v * scale

    bioclim_dict = {f'bioclim_{k.lstrip("bio")}': float(v) for k, v in bioclim_dict.items()}
    return bioclim_dict

def get_lc_from_coord(coords, patch_size=2000, year=2017):
    aoi = create_aoi_from_coord_buffer(coords, buffer_m=patch_size // 2, bool_buffer_in_deg=False)
    lon, lat = coords
    epsg_code = get_epsg_from_latlon(lat=lat, lon=lon)
    collection = ee.ImageCollection("COPERNICUS/CORINE/V20/100m")
    im_gee = ee.Image(collection
                      .filterBounds(aoi)
                      .filterDate(f'{year}-01-01', f'{year}-12-31')
                      .first()
                      .reproject(f'EPSG:{epsg_code}', scale=10)
                      .clip(aoi))
    return im_gee
    
def convert_corine_lc_im_to_tab(lc_im):
    """Convert a land cover image to a tabular format with pixel counts per class."""
    assert ONLINE_ACCESS_TO_GEE, "ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE"
    pixel_counts = (
        lc_im
        .reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=lc_im.geometry(),
            scale=10,  # match the image resolution
            maxPixels=1e9
        )
    )
    assert len(pixel_counts.getInfo()) == 1 and 'landcover' in pixel_counts.getInfo(), "Land cover band not found in the image."
    pixel_counts = pixel_counts.get('landcover').getInfo()  # has str keys ('211', etc)
    pixel_counts = {int(k): v for k, v in pixel_counts.items()}  # convert keys to int

    _, df_lc_classes = du.corine_lc_schema()
    for k, v in pixel_counts.items():
        assert k in df_lc_classes['code'].values, f"Land cover code {k} not found in land cover classes."

    sum_counts = sum(pixel_counts.values())
    assert sum_counts > 0, "No pixels found in the land cover image."
    dict_lc_counts = {f'corine_frac_{int(k)}': 0 if k not in pixel_counts else pixel_counts[k] / sum_counts for k in df_lc_classes['code'].values}
    return dict_lc_counts

if __name__ == "__main__":
    print('This is a utility script for GEE data preprocessing.')