import os, sys, datetime, ast
import numpy as np 
import pandas as pd 
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr
import loadpaths
path_dict = loadpaths.loadpaths()
import shapely
from tqdm import tqdm

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

def get_path_s2bms(path_dict=path_dict):
    """Get the path to the Sentinel-2 BMS data directory."""
    if 's2bms_images' in path_dict:
        im_path = path_dict['s2bms_images']
        assert os.path.exists(im_path), f"Sentinel-2 BMS image path does not exist: {im_path}"
    else:
        im_path = None
    if 's2bms_presence' in path_dict:
        presence_path = path_dict['s2bms_presence']
    else:
        presence_path = os.path.join(path_dict['repo'], 'content/S2BMS/ukbms_species-presence/bms_presence_y-2018-2019_th-200.shp')
    assert os.path.exists(presence_path), f"Sentinel-2 BMS presence path does not exist: {presence_path}"
    return im_path, presence_path

def load_s2bms_presence(path_dict=path_dict):
    """Load the Sentinel-2 BMS species presence GeoDataFrame."""
    _, s2bms_presence_path = get_path_s2bms()
    df_s2bms_presence = gpd.read_file(s2bms_presence_path)
    ## convert to WGS84
    df_s2bms_presence = df_s2bms_presence.to_crs(epsg=4326)
    df_s2bms_presence['lat'] = df_s2bms_presence.geometry.y
    df_s2bms_presence['lon'] = df_s2bms_presence.geometry.x
    df_s2bms_presence['tuple_coords'] = [tuple([ast.literal_eval(x) for x in df_s2bms_presence.tuple_coor[ii].lstrip('(').rstrip(')').split(', ')]) for ii in range(len(df_s2bms_presence))]
    df_s2bms_presence.drop(columns=['row_id', 'tuple_coor'], inplace=True)
    

    return df_s2bms_presence

def load_tiff(tiff_file_path, datatype='np', verbose=0):
    '''Load tiff file as np or da'''
    with rasterio.open(tiff_file_path) as f:
        if verbose > 0:
            print(f.profile)
        if datatype == 'np':  # handle different file types 
            im = f.read()
            assert type(im) == np.ndarray
        elif datatype == 'da':
            im = rxr.open_rasterio(f)
            assert type(im) == xr.DataArray
        else:
            assert False, 'datatype should be np or da'
    return im 

def create_timestamp(include_seconds=False):
    dt = datetime.datetime.now()
    timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
    if include_seconds:
        timestamp += ':' + str(dt.second).zfill(2)
    return timestamp

def load_aux_data(filepath='../outputs/s2bms_bioclim_lc_data.csv',
                  col_identifier='name'):
    """Load auxiliary bioclimatic and land cover data."""
    assert os.path.exists(filepath), f"Auxiliary data file does not exist: {filepath}"
    df_aux = pd.read_csv(filepath)
    if col_identifier is not None:
        assert col_identifier in df_aux.columns, f"Column identifier '{col_identifier}' not found in auxiliary data."
    return df_aux

def get_article(str_follow):
    assert type(str_follow) == str, f'str_follow must be a string, but got {type(str_follow)}'
    vowels = ['a', 'e', 'i', 'o', 'u']
    if str_follow[0].lower() in vowels:
        return 'an ' + str_follow
    else:
        return 'a ' + str_follow

def create_simple_caption(data_single_loc, include_lc=True, include_bioclim=True):
    # assert type(data_single_loc) == pd.Series or type(data_single_loc) == dict, f'Input must be a pandas Series or dict, but got {type(data_single_loc)}'
    assert type(data_single_loc) == pd.Series, f'Input must be a pandas Series, but got {type(data_single_loc)}'  ## assuming series rather than dict because it will be a bit faster
    _, corine_dict = corine_lc_schema()
    corine_names = dict(zip(corine_dict['code'], corine_dict['category_level_3']))
    _, bioclim_dict = bioclim_schema()
    bioclim_names = dict(zip(bioclim_dict['name'], bioclim_dict['description']))
    bioclim_units = dict(zip(bioclim_dict['name'], bioclim_dict['units']))
    key_names = data_single_loc.index
    bioclim_keys = [k for k in key_names if 'bioclim_' in k]
    corine_keys = [k for k in key_names if 'corine_frac_' in k]
 
    caption = 'Location with '
    if include_lc:
        top_3_lc = data_single_loc[corine_keys].sort_values(ascending=False)[:3]
        for i_lc, (lc_key, lc_frac) in enumerate(top_3_lc.items()):
            lc_class = lc_key.replace('corine_frac_', '').replace('_', ' ')
            lc_frac_percent = round(lc_frac * 100)
            if lc_frac_percent == 0:
                continue
            if i_lc == len(top_3_lc) - 1:
                caption += ' and '
            elif i_lc > 0:
                caption += ', '
            caption += f'{corine_names[int(lc_class)].lower()} ({lc_frac_percent}%)'
    if include_bioclim:
        caption += ', with '
        bio_keys_include = ['bio01', 'bio05', 'bio06', 'bio12']
        for i_bio, bio_key in enumerate(bio_keys_include):
            bio_df_key = bio_key.replace('bio', 'bioclim_')
            bio_value = data_single_loc[bio_df_key]
            if i_bio == len(bio_keys_include) - 1:
                caption += ' and '
            elif i_bio > 0:
                caption += ', '
            caption += f'{get_article(bioclim_names[bio_key].lower())} of {round(bio_value, 1)} {bioclim_units[bio_key]}'
        
    caption += '.'
    return caption

if __name__ == "__main__":
    print('This is a utility script for creating and processing the dataset.')