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
        print('WARNING: using buffer in degrees, which distorts images for large latitudes.')
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

def get_gee_image_from_coord(coords, collection_name='corine', patch_size=2000, year=2017,
                             sentinel_month_start=3, sentinel_month_end=9):
    aoi = create_aoi_from_coord_buffer(coords, buffer_m=patch_size // 2, bool_buffer_in_deg=False)
    lon, lat = coords
    epsg_code = get_epsg_from_latlon(lat=lat, lon=lon)
    if collection_name == 'corine':
        collection = ee.ImageCollection("COPERNICUS/CORINE/V20/100m")
    elif collection_name == 'alphaearth':
        collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    elif collection_name == 'sentinel2':
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    elif collection == 'dynamic_world':
        collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    else:
        raise NotImplementedError(f"Unknown collection_name: {collection_name}")
    if collection is None: raise ValueError(f"Could not access {collection_name} collection in GEE.")

    if collection_name in ['corine', 'alphaearth']:
        im_gee = ee.Image(collection
                        .filterBounds(aoi)
                        .filterDate(f'{year}-01-01', f'{year}-12-31')
                        .first()
                        .reproject(f'EPSG:{epsg_code}', scale=10)
                        .clip(aoi))
    elif collection_name in ['sentinel2']:
        month_start_str = str(sentinel_month_start).zfill(2)
        month_end_str = str(sentinel_month_end).zfill(2)    
        im_gee = ee.Image(collection 
                        .filterBounds(aoi) 
                        .filterDate(ee.Date(f'{year}-{month_start_str}-01'), ee.Date(f'{year}-{month_end_str}-01')) 
                        .select(['B4', 'B3', 'B2', 'B8'])  # 10m bands, RGB and NIR
                        .sort('CLOUDY_PIXEL_PERCENTAGE')
                        .first()  # get the least cloudy image
                        .reproject(f'EPSG:{epsg_code}', scale=10)
                        .clip(aoi))
    elif collection_name == 'dynamic_world':
        prob_bands = ["water", "trees", "grass", "flooded_vegetation",
                      "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
        im_gee = ee.Image(collection 
                        .filterBounds(aoi) 
                        .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31'))
                        .select(prob_bands)  # get all probability bands
                        .mean()  # mean over the year
                        .reproject(f'EPSG:{epsg_code}', scale=10)  # reproject to 10m
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