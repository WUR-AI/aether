import os, sys
import numpy as np 
import pandas as pd 
import geopandas as gpd
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


def corine_lc_schema():
    '''From https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_CORINE_V20_100m#bands'''
    corine_classes = [
        {"code": 111, "color": "#e6004d", "category": "Artificial surfaces > Urban fabric > Continuous urban fabric"},
        {"code": 112, "color": "#ff0000", "category": "Artificial surfaces > Urban fabric > Discontinuous urban fabric"},
        {"code": 121, "color": "#cc4df2", "category": "Artificial surfaces > Industrial, commercial, and transport units > Industrial or commercial units"},
        {"code": 122, "color": "#cc0000", "category": "Artificial surfaces > Industrial, commercial, and transport units > Road and rail networks and associated land"},
        {"code": 123, "color": "#e6cccc", "category": "Artificial surfaces > Industrial, commercial, and transport units > Port areas"},
        {"code": 124, "color": "#e6cce6", "category": "Artificial surfaces > Industrial, commercial, and transport units > Airports"},
        {"code": 131, "color": "#a600cc", "category": "Artificial surfaces > Mine, dump, and construction sites > Mineral extraction sites"},
        {"code": 132, "color": "#a64dcc", "category": "Artificial surfaces > Mine, dump, and construction sites > Dump sites"},
        {"code": 133, "color": "#ff4dff", "category": "Artificial surfaces > Mine, dump, and construction sites > Construction sites"},
        {"code": 141, "color": "#ffa6ff", "category": "Artificial surfaces > Artificial, non-agricultural vegetated areas > Green urban areas"},
        {"code": 142, "color": "#ffe6ff", "category": "Artificial surfaces > Artificial, non-agricultural vegetated areas > Sport and leisure facilities"},
        {"code": 211, "color": "#ffffa8", "category": "Agricultural areas > Arable land > Non-irrigated arable land"},
        {"code": 212, "color": "#ffff00", "category": "Agricultural areas > Arable land > Permanently irrigated land"},
        {"code": 213, "color": "#e6e600", "category": "Agricultural areas > Arable land > Rice fields"},
        {"code": 221, "color": "#e68000", "category": "Agricultural areas > Permanent crops > Vineyards"},
        {"code": 222, "color": "#f2a64d", "category": "Agricultural areas > Permanent crops > Fruit trees and berry plantations"},
        {"code": 223, "color": "#e6a600", "category": "Agricultural areas > Permanent crops > Olive groves"},
        {"code": 231, "color": "#e6e64d", "category": "Agricultural areas > Pastures > Pastures"},
        {"code": 241, "color": "#ffe6a6", "category": "Agricultural areas > Heterogeneous agricultural areas > Annual crops associated with permanent crops"},
        {"code": 242, "color": "#ffe64d", "category": "Agricultural areas > Heterogeneous agricultural areas > Complex cultivation patterns"},
        {"code": 243, "color": "#e6cc4d", "category": "Agricultural areas > Heterogeneous agricultural areas > Land principally occupied by agriculture, with significant areas of natural vegetation"},
        {"code": 244, "color": "#f2cca6", "category": "Agricultural areas > Heterogeneous agricultural areas > Agro-forestry areas"},
        {"code": 311, "color": "#80ff00", "category": "Forest and semi natural areas > Forests > Broad-leaved forest"},
        {"code": 312, "color": "#00a600", "category": "Forest and semi natural areas > Forests > Coniferous forest"},
        {"code": 313, "color": "#4dff00", "category": "Forest and semi natural areas > Forests > Mixed forest"},
        {"code": 321, "color": "#ccf24d", "category": "Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Natural grasslands"},
        {"code": 322, "color": "#a6ff80", "category": "Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Moors and heathland"},
        {"code": 323, "color": "#a6e64d", "category": "Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Sclerophyllous vegetation"},
        {"code": 324, "color": "#a6f200", "category": "Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Transitional woodland-shrub"},
        {"code": 331, "color": "#e6e6e6", "category": "Forest and semi natural areas > Open spaces with little or no vegetation > Beaches, dunes, sands"},
        {"code": 332, "color": "#cccccc", "category": "Forest and semi natural areas > Open spaces with little or no vegetation > Bare rocks"},
        {"code": 333, "color": "#ccffcc", "category": "Forest and semi natural areas > Open spaces with little or no vegetation > Sparsely vegetated areas"},
        {"code": 334, "color": "#000000", "category": "Forest and semi natural areas > Open spaces with little or no vegetation > Burnt areas"},
        {"code": 335, "color": "#a6e6cc", "category": "Forest and semi natural areas > Open spaces with little or no vegetation > Glaciers and perpetual snow"},
        {"code": 411, "color": "#a6a6ff", "category": "Wetlands > Inland wetlands > Inland marshes"},
        {"code": 412, "color": "#4d4dff", "category": "Wetlands > Inland wetlands > Peat bogs"},
        {"code": 421, "color": "#ccccff", "category": "Wetlands > Maritime wetlands > Salt marshes"},
        {"code": 422, "color": "#e6e6ff", "category": "Wetlands > Maritime wetlands > Salines"},
        {"code": 423, "color": "#a6a6e6", "category": "Wetlands > Maritime wetlands > Intertidal flats"},
        {"code": 511, "color": "#00ccf2", "category": "Water bodies > Inland waters > Water courses"},
        {"code": 512, "color": "#80f2e6", "category": "Water bodies > Inland waters > Water bodies"},
        {"code": 521, "color": "#00ffa6", "category": "Water bodies > Marine waters > Coastal lagoons"},
        {"code": 522, "color": "#a6ffe6", "category": "Water bodies > Marine waters > Estuaries"},
        {"code": 523, "color": "#e6f2ff", "category": "Water bodies > Marine waters > Sea and ocean"},
    ]

    dict_all = {x: [] for x in ['code', 'color', 'category', 'category_level_1', 'category_level_2', 'category_level_3']}
    for item in corine_classes:
        for key, val in item.items():   
            dict_all[key].append(val)
            if key == 'category':
                levels = val.split(' > ')
                dict_all['category_level_1'].append(levels[0])
                dict_all['category_level_2'].append(levels[1])
                dict_all['category_level_3'].append(levels[2])
    df_all = pd.DataFrame(dict_all)
    return corine_classes, df_all

def bioclim_schema():
    '''From https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_BIO'''
    bioclim_variables = [
        {"name": "bio01", "units": "°C", "min": -29, "max": 32, "scale": 0.1, "pixel_size": "meters", "description": "Annual mean temperature"},
        {"name": "bio02", "units": "°C", "min": 0.9, "max": 21.4, "scale": 0.1, "pixel_size": "meters", "description": "Mean diurnal range (mean of monthly (max temp - min temp))"},
        {"name": "bio03", "units": "%", "min": 7, "max": 96, "scale": 1, "pixel_size": "meters", "description": "Isothermality (bio02/bio07 * 100)"},
        {"name": "bio04", "units": "°C", "min": 0.62, "max": 227.21, "scale": 0.01, "pixel_size": "meters", "description": "Temperature seasonality (Standard deviation * 100)"},
        {"name": "bio05", "units": "°C", "min": -9.6, "max": 49, "scale": 0.1, "pixel_size": "meters", "description": "Max temperature of warmest month"},
        {"name": "bio06", "units": "°C", "min": -57.3, "max": 25.8, "scale": 0.1, "pixel_size": "meters", "description": "Min temperature of coldest month"},
        {"name": "bio07", "units": "°C", "min": 5.3, "max": 72.5, "scale": 0.1, "pixel_size": "meters", "description": "Temperature annual range (bio05 - bio06)"},
        {"name": "bio08", "units": "°C", "min": -28.5, "max": 37.8, "scale": 0.1, "pixel_size": "meters", "description": "Mean temperature of wettest quarter"},
        {"name": "bio09", "units": "°C", "min": -52.1, "max": 36.6, "scale": 0.1, "pixel_size": "meters", "description": "Mean temperature of driest quarter"},
        {"name": "bio10", "units": "°C", "min": -14.3, "max": 38.3, "scale": 0.1, "pixel_size": "meters", "description": "Mean temperature of warmest quarter"},
        {"name": "bio11", "units": "°C", "min": -52.1, "max": 28.9, "scale": 0.1, "pixel_size": "meters", "description": "Mean temperature of coldest quarter"},
        {"name": "bio12", "units": "mm", "min": 0, "max": 11401, "scale": 1, "pixel_size": "meters", "description": "Annual precipitation"},
        {"name": "bio13", "units": "mm", "min": 0, "max": 2949, "scale": 1, "pixel_size": "meters", "description": "Precipitation of wettest month"},
        {"name": "bio14", "units": "mm", "min": 0, "max": 752, "scale": 1, "pixel_size": "meters", "description": "Precipitation of driest month"},
        {"name": "bio15", "units": "Coefficient of Variation", "min": 0, "max": 265, "scale": 1, "pixel_size": "meters", "description": "Precipitation seasonality"},
        {"name": "bio16", "units": "mm", "min": 0, "max": 8019, "scale": 1, "pixel_size": "meters", "description": "Precipitation of wettest quarter"},
        {"name": "bio17", "units": "mm", "min": 0, "max": 2495, "scale": 1, "pixel_size": "meters", "description": "Precipitation of driest quarter"},
        {"name": "bio18", "units": "mm", "min": 0, "max": 6090, "scale": 1, "pixel_size": "meters", "description": "Precipitation of warmest quarter"},
        {"name": "bio19", "units": "mm", "min": 0, "max": 5162, "scale": 1, "pixel_size": "meters", "description": "Precipitation of coldest quarter"},
    ]
    dict_all = {x: [] for x in ['name', 'units', 'min', 'max', 'scale', 'pixel_size', 'description']}
    for item in bioclim_variables:
        for key, val in item.items():   
            dict_all[key].append(val)
    df_all = pd.DataFrame(dict_all)
    return bioclim_variables, df_all

def create_aoi_from_coord_buffer(coords, buffer_deg=0.01, buffer_m=1000, bool_buffer_in_deg=True):
    """Create an Earth Engine AOI (Geometry) from a coordinate and buffer in meters."""
    point = shapely.geometry.Point(coords)
    if bool_buffer_in_deg:  # not ideal https://gis.stackexchange.com/questions/304914/python-shapely-intersection-with-buffer-in-meter
        polygon = point.buffer(buffer_deg, cap_style=3)  ## buffer in degrees
        xy_coords = np.array(polygon.exterior.coords.xy).T 
        aoi = ee.Geometry.Polygon(xy_coords.tolist())
    else:
        assert False, 'verify this part of the code'
         ## buffer in meters
        point = ee.Geometry.Point(coords)
        aoi = point.buffer(buffer_m)
    assert aoi is not None
    return aoi

def get_bioclim_from_coord(coords):
    assert ONLINE_ACCESS_TO_GEE, "ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE"
    aoi = create_aoi_from_coord_buffer(coords, buffer_deg=0.01, bool_buffer_in_deg=True)
    im_gee = ee.Image("WORLDCLIM/V1/BIO").clip(aoi) 
    point = ee.Geometry.Point(coords)  # redefine point for sampling
    values = im_gee.sample(region=point.buffer(1000), scale=1000).first().toDictionary().getInfo()
    return values 

def convert_bioclim_to_units(bioclim_dict):
    assert len(bioclim_dict) == 19, "bioclim_dict should have 19 variables"
    for k in range(1, 20):
        assert f'bio{str(k).zfill(2)}' in bioclim_dict, f'bio{str(k).zfill(2)} not in bioclim_dict'
    _, df_bioclim = bioclim_schema()
    for k, v in bioclim_dict.items():
        scale = df_bioclim.loc[df_bioclim['name'] == k, 'scale'].values[0]
        bioclim_dict[k] = v * scale

    bioclim_dict = {f'bioclim_{k.lstrip("bio")}': float(v) for k, v in bioclim_dict.items()}
    return bioclim_dict

def get_lc_from_coord(coords, patch_size=None):
    aoi = create_aoi_from_coord_buffer(coords, bool_buffer_in_deg=True)

    collection = ee.ImageCollection("COPERNICUS/CORINE/V20/100m")
    im_gee = ee.Image(collection
                      .filterBounds(aoi)
                      .filterDate('2017-01-01', '2018-12-31')
                      .first()
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

    _, df_lc_classes = corine_lc_schema()
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