import os, sys
import numpy as np 
import pandas as pd 
import geopandas as gpd
import loadpaths
path_dict = loadpaths.loadpaths()
import shapely
from tqdm import tqdm

ONLINE_ACCESS_TO_GEE = True 
if ONLINE_ACCESS_TO_GEE:
    import api_keys
    import ee, geemap 
    ee.Authenticate()
    ee.Initialize(project=api_keys.GEE_API)
    geemap.ee_initialize()
else:
    print('WARNING: ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE')