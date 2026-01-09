import os, datetime, ast
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
import rioxarray as rxr
from hydra import initialize, compose
from omegaconf import OmegaConf


# TODO: is this still used? works?
def get_hydra_paths():
    assert False, 'Deprecated. Use hydra config or environment variables instead.'
    cwd = os.getcwd() + '/..'
    output_dir = os.path.join(cwd, "outputs/temp/")
    # repo_dir = os.environ.get('PROJECT_ROOT', cwd)
    with initialize(config_path='../../configs/paths', version_base="1.1"):
        cfg = compose(
            config_name="local.yaml",
            overrides=[
                f"hydra.run.dir={cwd}",
                f"hydra.job.num=0",  # required to resolve hydra.job.* interpolations
                f"output_dir={output_dir}",
                f"work_dir={cwd}",
            ],
        )

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    path_dict = resolved_cfg  # ['paths']
    path_dict['repo'] = path_dict['root_dir']
    return path_dict


def process_corine_classes(input_path, output_path):
    '''Creates processed csv file with all corine classes'''

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'File {input_path} for Corine classes does not exist')

    # read-in org file
    with open(input_path, 'r') as f:
        corine_classes = json.load(f)

    # format it into pandas
    keys = list(corine_classes[0].keys())
    keys.extend(['category_level_1', 'category_level_2', 'category_level_3'])

    # collect all keys and vals, split descriptions based on categories
    dict_all = {x: [] for x in keys}
    for item in corine_classes:
        for key, val in item.items():
            if key == 'code':
                val = f'aux_corine_frac_{val}'
            dict_all[key].append(val)
            if key == 'category':
                levels = val.split(' > ')
                dict_all['category_level_1'].append(levels[0])
                dict_all['category_level_2'].append(levels[1])
                dict_all['category_level_3'].append(levels[2])
    df = pd.DataFrame(dict_all)

    # save csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Processed Corine classes were saved to {output_path}')


def process_bioclim_classes(input_path, output_path):
    '''Creates processed csv file with all bioclim classes'''

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'File {input_path} for bioclimatic scheme does not exist')

    with open(input_path, 'r') as f:
        bioclim_classes = json.load(f)

    # Create and save pandas df
    df = pd.DataFrame(bioclim_classes)
    df['name'] = df['name'].apply(lambda x: x.replace('bio', 'aux_bioclim_'))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Processed bioclimatic classes were saved to {output_path}')


def corine_lc_schema(data_dir='data/'):
    '''From https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_CORINE_V20_100m#bands'''
    if not os.path.isfile(os.path.join(data_dir, "caption_templates/corine_classes.csv")):
        process_corine_classes(
            os.path.join(data_dir, "source/corine_classes.json"),
            os.path.join(data_dir, "caption_templates/corine_classes.csv"),
        )
    df = pd.read_csv(os.path.join(data_dir, ("caption_templates/corine_classes.csv")))

    corine_classes = df.to_dict('records')
    for c in corine_classes:
        c['code'] = int(c['code'].replace('aux_corine_frac_', ''))

    return corine_classes, df


def bioclim_schema(data_dir='data/'):
    '''From https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_BIO'''

    if not os.path.isfile(os.path.join(data_dir, "caption_templates/bioclim_classes.csv")):
        process_bioclim_classes(
            os.path.join(data_dir, "source/bioclim_classes.json"),
            os.path.join(data_dir, "caption_templates/bioclim_classes.csv"),
        )

    df = pd.read_csv(os.path.join(data_dir, "caption_templates/bioclim_classes.csv"))
    df.sort_values(by=["name"], inplace=True)
    bioclim_variables = df.to_dict('records')
    for v in bioclim_variables:
        v['name'] = v['name'].replace('aux_bioclim_', 'bio')

    return bioclim_variables, df


def get_path_s2bms():
    """Get the path to the Sentinel-2 BMS data directory."""
    if 'S2BMS_IMAGES' in os.environ:
        im_path = os.environ.get('S2BMS_IMAGES')
        assert os.path.exists(im_path), f"Sentinel-2 BMS image path does not exist: {im_path}"
    else:
        im_path = None
    if 'S2BMS_PRESENCE' in os.environ:
        presence_path = os.environ.get('S2BMS_PRESENCE')
    else:
        presence_path = os.path.join(
            os.environ.get('PROJECT_ROOT', '.'),
            'data/source/butterflies/S2BMS/ukbms_species-presence/bms_presence_y-2018-2019_th-200.shp',
        )
    assert os.path.exists(presence_path), f"Sentinel-2 BMS presence path does not exist: {presence_path}"
    return im_path, presence_path


def load_s2bms_presence():
    """Load the Sentinel-2 BMS species presence GeoDataFrame."""
    _, s2bms_presence_path = get_path_s2bms()
    df_s2bms_presence = gpd.read_file(s2bms_presence_path)
    ## convert to WGS84
    df_s2bms_presence = df_s2bms_presence.to_crs(epsg=4326)
    df_s2bms_presence['lat'] = df_s2bms_presence.geometry.y
    df_s2bms_presence['lon'] = df_s2bms_presence.geometry.x
    df_s2bms_presence['tuple_coords'] = [
        tuple([ast.literal_eval(x) for x in df_s2bms_presence.tuple_coor[ii].lstrip('(').rstrip(')').split(', ')])
        for ii in range(len(df_s2bms_presence))
    ]
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


def load_aux_data(filepath='../data/model_ready/s2bms_bioclim_lc_data.csv', col_identifier='name'):
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


if __name__ == "__main__":
    print('This is a utility script for creating and processing the dataset.')
