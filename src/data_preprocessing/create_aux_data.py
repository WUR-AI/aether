import os, sys
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.data_preprocessing import gee_utils as gu
from src.data_preprocessing import data_utils as du

def get_bioclim_lc_from_coords(coords):
    """Get both bioclimatic and land cover data from coordinates."""
    bioclim_data = gu.get_bioclim_from_coord(coords)
    bioclim_data = gu.convert_bioclim_to_units(bioclim_data)
    lc_im = gu.get_gee_image_from_coord(coords, collection_name='corine')
    lc_data = gu.convert_corine_lc_im_to_tab(lc_im)
    return {**bioclim_data, **lc_data}

def get_bioclim_lc_from_coords_list(coords_list, name_list=None, save_file=False,
                                    save_folder=os.path.join(os.environ['PROJECT_ROOT'], 'data/source/butterflies/'), 
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

def create_butterfly_aux_data(download_aux_data=False, data_dir=None, filename='s2bms_bioclim_lc_data.csv',
                              prefix_aux='aux_', prefix_target='target_', save_file=True):
    '''Create auxiliary dataset for S2BMS butterfly data. 
    Steps:
    - Load S2BMS presence data.
    - Download bioclimatic and land cover data if needed, else open.
    - Merge the presence (target) & auxiliary data. to one csv. Some renaming etc.'''
    assert type(prefix_aux) == str, "prefix_aux must be a string"
    assert type(prefix_target) == str, "prefix_target must be a string"
    df_s2bms_presence = du.load_s2bms_presence()
    if download_aux_data:
        get_bioclim_lc_from_coords_list(coords_list=df_s2bms_presence.tuple_coords.values,
                                        name_list=df_s2bms_presence.name_loc.values,
                                        save_file=False, save_filename=filename)
   
    if data_dir is None:
        data_dir = os.path.join(os.environ['PROJECT_ROOT'], 'data')
    path_butterfly_aux_target = os.path.join(data_dir, 'source', 'butterflies', filename)
    assert os.path.exists(path_butterfly_aux_target), f"Butterfly auxiliary data file does not exist: {path_butterfly_aux_target}"
    df_bioclim_lc = pd.read_csv(path_butterfly_aux_target)
    corine_keys = [k for k in df_bioclim_lc.iloc[0].index if 'corine_frac_' in k]

    ## rename columns:
    df_bioclim_lc.rename(columns={'name': 'name_loc'}, inplace=True)
    if prefix_aux != '':
        for c in df_bioclim_lc.columns:
            if c in ['name_loc']:
                continue
            df_bioclim_lc.rename(columns={c: f'{prefix_aux}{c}'}, inplace=True)

    df_s2bms_presence.drop(columns=['geometry', 'n_visits'], inplace=True)
    df_s2bms_presence.rename(columns={'tuple_coords':'coords'}, inplace=True)
    if prefix_target != '':
        for c in df_s2bms_presence.columns:
            if c in ['name_loc', 'lat', 'lon']:
                continue
            df_s2bms_presence.rename(columns={c: f'{prefix_target}{c.replace(" ", "_")}'}, inplace=True)

    df_merged = pd.merge(df_s2bms_presence, df_bioclim_lc, left_on='name_loc', right_on='name_loc', how='left')
    coord_columns = [col for col in df_merged.columns if 'coords' in col]
    df_merged.drop(columns=coord_columns, inplace=True)

    columns_ordered = ['name_loc', 'lat', 'lon'] + \
        sorted([c for c in df_merged.columns if c not in ['name_loc', 'lat', 'lon']])
    df_merged = df_merged[columns_ordered]

    if save_file:
        os.makedirs('../data/model_ready/', exist_ok=True)
        df_merged.to_csv('../data/model_ready/s2bms_presence_with_aux_data.csv', index=False)
    return df_merged

if __name__ == "__main__":
    df_s2bms_presence = du.load_s2bms_presence()
    get_bioclim_lc_from_coords_list(coords_list=df_s2bms_presence.tuple_coords.values,
                                   name_list=df_s2bms_presence.name_loc.values,
                                   save_file=True, save_filename='s2bms_bioclim_lc_data.csv')