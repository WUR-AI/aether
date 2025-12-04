import os, sys
import numpy as np 
import pandas as pd
import loadpaths
path_dict = loadpaths.loadpaths()
import shapely
from tqdm import tqdm
sys.path.append('../content/')
import data_utils as du

ONLINE_ACCESS_TO_GEE = True 
if ONLINE_ACCESS_TO_GEE:
    import api_keys
    import ee, geemap 
    ee.Authenticate()
    ee.Initialize(project=api_keys.GEE_API)
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

def get_bioclim_lc_from_coords(coords):
    """Get both bioclimatic and land cover data from coordinates."""
    bioclim_data = get_bioclim_from_coord(coords)
    bioclim_data = convert_bioclim_to_units(bioclim_data)
    lc_im = get_lc_from_coord(coords)
    lc_data = convert_corine_lc_im_to_tab(lc_im)
    return {**bioclim_data, **lc_data}

def get_bioclim_lc_from_coords_list(coords_list, name_list=None, save_file=False,
                                    save_folder=os.path.join(path_dict['repo'], 'outputs/'), 
                                    save_filename='bioclim_lc_data.csv'):
    """Get both bioclimatic and land cover data from a list of coordinates."""
    if name_list is not None:
        assert len(name_list) == len(coords_list), "name_list and coords_list must have the same length"
    if save_file:
        save_path = os.path.join(save_folder, save_filename)
        assert os.path.exists(save_folder), f"Save folder does not exist: {save_folder}"
        save_every_n = 100  # save every n samples to avoid data loss
        print(f'Will save bioclimatic and land cover data to {save_path} every {save_every_n} samples')
    results = {}
    with tqdm(total=len(coords_list), desc='Collecting bioclimatic and land cover data') as pbar:
        for i_coords, coords in enumerate(coords_list):
            try:
                result = get_bioclim_lc_from_coords(coords)
                result_keys = list(result.keys())
            except Exception as e:
                print(f"Error occurred while processing coordinates {i_coords}, {coords}: {e}")
                result = {k: np.nan for k in result_keys}
            if i_coords == 0:
                for k in result.keys():
                    results[k] = []
                results['coords'] = []
                if name_list is not None:
                    results['name'] = []
            if name_list is not None:
                results['name'].append(name_list[i_coords])
            results['coords'].append(coords)
            for k, v in result.items():
                results[k].append(v)
            pbar.update(1)

        if save_file and (i_coords + 1) % save_every_n == 0:
            temp_results = pd.DataFrame(results)
            temp_results.to_csv(save_path, index=False)
            print(f"Intermediate save of bioclimatic and land cover data to {save_path} at {i_coords + 1} samples")

    results = pd.DataFrame(results)
    if save_file:
        results.to_csv(save_path, index=False)
        print(f"Saved bioclimatic and land cover data to {save_path}")
    return results

if __name__ == "__main__":
    df_s2bms_presence = du.load_s2bms_presence()
    get_bioclim_lc_from_coords_list(coords_list=df_s2bms_presence.tuple_coords.values,
                                   name_list=df_s2bms_presence.name_loc.values,
                                   save_file=True, save_filename='s2bms_bioclim_lc_data.csv')