import os, sys
import numpy as np 
import pandas as pd 
import geopandas as gpd
import loadpaths
path_dict = loadpaths.loadpaths()
import shapely
from tqdm import tqdm

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
    df_s2bms_presence.drop(columns=['row_id', 'tuple_coor'], inplace=True)
    return df_s2bms_presence