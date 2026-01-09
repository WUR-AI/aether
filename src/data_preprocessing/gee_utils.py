'''
Google Earth Engine (GEE) utility functions for data preprocessing. 
Many functions were adapted from github.com/vdplasthijs/NeurEO.
'''


import os, sys
import numpy as np
import ee, geemap
import utm
import shapely
import json
from tqdm import tqdm

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
                             sentinel_month_start=3, sentinel_month_end=9, threshold_size: int | None = None):
    '''Get a GEE image from coordinates, for a given collection.
    Collections can have slightly different parameters/logic, hence they are split up in different if statements.'''
    aoi = create_aoi_from_coord_buffer(coords, buffer_m=patch_size // 2, bool_buffer_in_deg=False)
    lon, lat = coords
    epsg_code = get_epsg_from_latlon(lat=lat, lon=lon)
    if collection_name == 'corine':
        assert year == 2017, "GEE CORINE collection only has data for year 2017 I believe"
        collection = ee.ImageCollection(f"COPERNICUS/CORINE/V20/100m")
    elif collection_name == 'alphaearth':
        collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    elif collection_name == 'sentinel2':
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    elif collection_name == 'dynamicworld':
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
        sentinel_month_start = str(sentinel_month_start).zfill(2)
        sentinel_month_end = str(sentinel_month_end).zfill(2)    
        im_gee = ee.Image(collection 
                        .filterBounds(aoi) 
                        .filterDate(ee.Date(f'{year}-{sentinel_month_start}-01'), ee.Date(f'{year}-{sentinel_month_end}-01')) 
                        .select(['B4', 'B3', 'B2', 'B8'])  # 10m bands, RGB and NIR
                        .sort('CLOUDY_PIXEL_PERCENTAGE')
                        .first()  # get the least cloudy image
                        .reproject(f'EPSG:{epsg_code}', scale=10)
                        .clip(aoi))
    elif collection_name == 'dynamicworld':
        prob_bands = ["water", "trees", "grass", "flooded_vegetation",
                      "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
        im_gee = ee.Image(collection 
                        .filterBounds(aoi) 
                        .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31'))
                        .select(prob_bands)  # get all probability bands
                        .mean()  # mean over the year
                        .reproject(f'EPSG:{epsg_code}', scale=10)  # reproject to 10m
                        .clip(aoi)) 
        
    if threshold_size is not None:
        im_dims = im_gee.getInfo()["bands"][0]["dimensions"]
        if im_dims[0] < threshold_size or im_dims[1] < threshold_size:
            print('WARNING: image too small, returning None')
            return None
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

def create_filename(base_name, collection_name='sentinel2', year=2024,
                    sentinel_month_start='06', sentinel_month_end='09'):
    '''Create a filename for the GEE image based on collection and parameters.
    base_name should correspond to a unique identifier for the location/sample.'''
    if collection_name == 'sentinel2':
        filename = f'{base_name}_sent2-4band_y-{year}_m-{sentinel_month_start}-{sentinel_month_end}.tif'
    elif collection_name == 'alphaearth':
        filename = f'{base_name}_alphaearth_y-{year}.tif'
    elif collection_name == 'worldclimbio':
        filename = f'{base_name}_worldclimbio_v1.json'
    elif collection_name == 'dynamicworld':
        filename = f'{base_name}_dynamicworld_y-{year}.tif'
    elif collection_name == 'dsm':
        filename = f'{base_name}_dsm_y-{year}.tif'
    elif collection_name == 'corine':
        filename = f'{base_name}_corine_y-{year}.tif'
    else:
        raise NotImplementedError(f"Unknown collection_name: {collection_name}")
    return filename

def download_gee_image(coords, name: str, path_save: str,
                       pixel_patch_size=128, verbose=0, year=2019,
                        sentinel_month_start='06', sentinel_month_end='09',
                        collection_name='sentinel2', resize_image=True):
    '''Download a GEE image for given coordinates and save it locally.
    Steps:
    - Given a desired patch size, create an AOI around the coordinates with an added buffer.
    - Retrieve the image from the GEE collection. 
    - Download using geemap.
    - Open the image, crop to desired size (if resize_image is True), and save again.'''
    assert collection_name in ['sentinel2', 'alphaearth', 'dynamicworld', 'corine'], f'image collection {collection_name} not recognised.'
    assert type(path_save) and os.path.exists(path_save), f'path_save must be a valid path, got {path_save}'
    gsd_resolution = 10
    patch_size = (pixel_patch_size + 20) * gsd_resolution  # adding a bit extra in case of minor misalignment
    im_gee = get_gee_image_from_coord(coords=coords, patch_size=patch_size, year=year, 
                                    sentinel_month_start=sentinel_month_start, sentinel_month_end=sentinel_month_end,
                                    collection_name=collection_name,
                                    threshold_size=pixel_patch_size)
    if im_gee is None:  ## if image was too small it was discarded
        if verbose:
            print('No image downloaded, returning None')
        return None, None

    if verbose:
        print('Image selected. Saving now.')
    filename = create_filename(base_name=name, collection_name=collection_name, year=year,
                               sentinel_month_start=sentinel_month_start, sentinel_month_end=sentinel_month_end)
    filepath = os.path.join(path_save, filename)
    
    geemap.ee_export_image(
        im_gee, filename=filepath, 
        scale=10,  # 10m bands
        file_per_band=False,# crs='EPSG:32630'
        verbose=False
    )

    if resize_image:  # load & crop & save to size correctly (because of buffer): 
        remove_if_too_small = True  # deletes image entirely if too small (and hence not able to resize)
        im = du.load_tiff(filepath, datatype='da')
        if verbose:
            print('Original size: ', im.shape)
        if im.shape[1] < pixel_patch_size or im.shape[2] < pixel_patch_size:
            print('WARNING: image too small, returning None')
            if remove_if_too_small:
                os.remove(filepath)
            return None, None

        ## crop:
        padding_1 = (im.shape[1] - pixel_patch_size) // 2
        padding_2 = (im.shape[2] - pixel_patch_size) // 2
        im_crop = im[:, padding_1:pixel_patch_size + padding_1, padding_2:pixel_patch_size + padding_2]
        assert im_crop.shape[0] == im.shape[0] and im_crop.shape[1] == pixel_patch_size and im_crop.shape[2] == pixel_patch_size, im_crop.shape
        if verbose:
            print('New size: ', im_crop.shape)
        im_crop.rio.to_raster(filepath)
        im_gee = im_crop 

    return im_gee, filepath

def download_list_coord(coord_list, name_list=None, path_save: str | None = None, pixel_patch_size=128,
                        name_group='sample', start_index=0, stop_index=None, resize_image=True, 
                        list_collections=['sentinel2', 'alphaearth']):
    '''For a list of coordinates (and optional names), download GEE images for each coordinate and save them locally.'''
    assert type(coord_list) == list
    assert path_save is not None and type(path_save) == str, "path_save must be provided"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        print(f'Created folder {path_save}')
    else:
        print(f'WARNING: folder {path_save} already exists. OVERWRITING any existing files with same names!')
    if name_list is not None:
        assert len(name_list) == len(coord_list), "name_list and coord_list must have the same length"

    ## save list coords:
    filename_coords = os.path.join(path_save, f'{name_group}_coords.json')
    with open(filename_coords, 'w') as f:
        json.dump(coord_list, f)

    inds_none = []
    for i, coords in enumerate(tqdm(coord_list)):
        if i < start_index:
            continue
        if stop_index is not None and i >= stop_index:
            break
        if name_list is not None:
            name = f'{name_group}_{name_list[i]}'
        else:
            name = f'{name_group}_{i}'
        for im_collection in list_collections:
            try:
                im, path_im = download_gee_image(coords=coords, name=name, 
                                                 pixel_patch_size=pixel_patch_size,
                                                path_save=path_save, verbose=0,
                                                resize_image=resize_image,
                                                collection_name=im_collection)
            except Exception as e:
                print(f'Image {name}, {im_collection} could not be downloaded, error: {e}')
                im = None
            if im is None:
                inds_none.append(f'{i}_{im_collection}')
        
    if len(inds_none) > 0:
        print(f'Successfully downloaded {len(coord_list) - len(inds_none)} out of {len(coord_list)} images.')
        print(f'Images that could not be downloaded: {inds_none}')
    else:
        print(f'Successfully downloaded all {len(coord_list)} images.')
    return inds_none

if __name__ == "__main__":
    print('This is a utility script for GEE data preprocessing.')