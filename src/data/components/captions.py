import os, sys 
sys.path.append('../../data_processing')  # tidy up  
import data_utils as du

def create_simple_caption(data_single_loc, include_lc=True, include_bioclim=True):
    # assert type(data_single_loc) == pd.Series or type(data_single_loc) == dict, f'Input must be a pandas Series or dict, but got {type(data_single_loc)}'
    assert type(data_single_loc) == pd.Series, f'Input must be a pandas Series, but got {type(data_single_loc)}'  ## assuming series rather than dict because it will be a bit faster
    _, corine_dict = du.corine_lc_schema()
    corine_names = dict(zip(corine_dict['code'], corine_dict['category_level_3']))
    _, bioclim_dict = du.bioclim_schema()
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
            caption += f'{du.get_article(bioclim_names[bio_key].lower())} of {round(bio_value, 1)} {bioclim_units[bio_key]}'
        
    caption += '.'
    return caption