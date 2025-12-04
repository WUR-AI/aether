import os, sys
from tqdm import tqdm
import numpy as np
import pandas as pd

from . import data_utils as du 
from . import gee_utils as gu

path_dict = du.get_hydra_paths()

def get_bioclim_lc_from_coords(coords):
    """Get both bioclimatic and land cover data from coordinates."""
    bioclim_data = gu.get_bioclim_from_coord(coords)
    bioclim_data = gu.convert_bioclim_to_units(bioclim_data)
    lc_im = gu.get_lc_from_coord(coords)
    lc_data = gu.convert_corine_lc_im_to_tab(lc_im)
    return {**bioclim_data, **lc_data}

def get_bioclim_lc_from_coords_list(coords_list, name_list=None, save_file=False,
                                    save_folder=os.path.join(path_dict['repo'], 'data/source/butterflies/'), 
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