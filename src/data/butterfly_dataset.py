from typing import override, Dict, Any
import os
import torch
import numpy as np
from src.data.base_dataset import BaseDataset
from src.data.utils import open_file_as_tensor
import src.data_preprocessing.data_utils as du

path_dict = du.get_hydra_paths()

class ButterflyDataset(BaseDataset):
    def __init__(
            self,
            path_csv: str = os.path.join(path_dict['data_dir'], 'model_ready/s2bms_presence_with_aux_data.csv'),
            modalities: list[str] = ['coords'],
            use_target_data: bool = True,
            use_aux_data: bool = False,
            random_state: int = 42,
            path_s2_im: str = None,
            n_bands: int = 4,
            zscore_im: bool = True
    ) -> None:
        super().__init__(path_csv, modalities, use_target_data, use_aux_data, 'Butterflies', random_state)

        # Placeholder for filtered columns
        columns = ['id']
        self.df.rename(columns={'name_loc': 'id'}, inplace=True)

        for m in self.modalities:
            assert m in ['s2', 'coords'], f'Unsupported modality: {m}'

        if 'coords' in self.modalities:
            columns.extend(['lat', 'lon'])
        if 's2' in self.modalities:
            if path_s2_im is None:
                path_s2_im = os.path.join(os.environ.get('S2BMS_PATH', None), 'sentinel2_satellite-images/y-2018-2019_m-06-09')  ## default path from S2BMS dataset (on Zotero). Assuming S2BMS_PATH points to the parent folder
                assert path_s2_im is not None, "S2BMS_PATH environment variable not set, please provide path_s2_im"
            assert os.path.exists(path_s2_im), f'S2BMS path does not exist: {path_s2_im}'
            self.path_s2_im = path_s2_im
            columns.append('s2_path')
            self.add_s2_paths()
            self.n_bands = n_bands
            self.zscore_im = zscore_im
            if self.zscore_im:
                self.init_norm_stats()

        if self.use_target_data:
            self.target_names = [c for c in self.df.columns if 'target' in c]
            columns.extend(self.target_names)
            self.num_classes = len(self.target_names)

        if self.use_aux_data:
            self.aux_names = [c for c in self.df.columns if c not in columns and c != 'name_loc']
            columns.extend(self.aux_names)

        self.records = self.df.loc[:, columns].to_dict('records')

    def init_norm_stats(self, means: list[float] = None, stds: list[float] = None):
        if means is None or stds is None:
            print('Using S2BMS default zscore means and stds')
            means = np.array([661.1047,  770.6800,  531.8330, 3228.5588]).astype(np.float32)  ## computed across entire ds
            stds = np.array([640.2482,  571.8545,  597.3570, 1200.7518]).astype(np.float32) 
        if self.n_bands == 3:
            means = means[:3]
            stds = stds[:3]
        self.norm_means = means[:, None, None]
        self.norm_std = stds[:, None, None]

    def find_image_path(self, name_loc, prefix_images: str='', suffix_images: list[str]=['']):
        if len(suffix_images) == 1:
            im_file_name = f'{prefix_images}_{name_loc}_{suffix_images[0]}'
            im_file_path = os.path.join(self.path_s2_im, im_file_name)
            if os.path.exists(im_file_path):
                return im_file_path
        else:
            for s in suffix_images:
                im_file_name = f'{prefix_images}_{name_loc}_{s}'
                im_file_path = os.path.join(self.path_s2_im, im_file_name)
                if os.path.exists(im_file_path):
                    return im_file_path
        return None

    def add_s2_paths(self):
        assert os.path.exists(self.path_s2_im), f"Image folder does not exist: {self.path_s2_im}"
        content_image_folder = os.listdir(self.path_s2_im)
        suffix_images = list(set((['_'.join(x.split('_')[3:]) for x in content_image_folder])))
        prefix_images = list(set(([x.split('_')[0] for x in content_image_folder])))
        assert len(prefix_images) == 1, f'Multiple prefixes found in image folder: {prefix_images}'
        prefix_images = prefix_images[0]
        list_paths = [] 
        for loc in self.df['id'].values:
            im_path = self.find_image_path(loc, prefix_images=prefix_images, suffix_images=suffix_images)
            if im_path is None:
                ## could be changed to a warning instead of error, if downstream code can handle missing images
                raise FileNotFoundError(f'No image found for location {loc} in folder {self.path_s2_im}')
            else:
                list_paths.append(im_path)
        self.df['s2_path'] = list_paths

    def zscore_image(self, im: np.ndarray):
        '''Apply preprocessing function to a single image. 
        raw_sent2_means = torch.tensor([661.1047,  770.6800,  531.8330, 3228.5588])
        raw_sent2_stds = torch.tensor([640.2482,  571.8545,  597.3570, 1200.7518])
        '''
        im = (im - self.norm_means) / self.norm_std
        return im

    def load_image(self, filepath: str):
        im = du.load_tiff(filepath, datatype='np')
        
        if self.n_bands == 4:
            pass 
        elif self.n_bands == 3:
            im = im[:3, :, :]
        else:
            assert False, f'Number of bands {self.n_bands} not implemented.'

        if self.zscore_im:
            im = im.astype(np.int32)
            im = self.zscore_image(im)
        else:
            im = np.clip(im, 0, 2000)
            im = im / 2000.0
        return torch.tensor(im).float()

    @override
    def __getitem__(
            self,
            idx: int
    ) -> Dict[str, Any]:
        row = self.records[idx]

        formatted_row = {'eo': {}}

        for modality in self.modalities:
            if modality in ['coords']:
                formatted_row['eo'][modality] = torch.tensor([row['lat'], row['lon']])
            elif modality == 's2':
                # formatted_row['eo'][modality] = open_file_as_tensor(row['s2_path'])
                formatted_row['eo'][modality] = self.load_image(row['s2_path'])

        if self.use_target_data:
            formatted_row['target'] = torch.tensor([row[k] for k in self.target_names], dtype=torch.float32)

        if self.use_aux_data:
            formatted_row['aux'] = torch.tensor([row[k] for k in self.aux_names], dtype= torch.float32)

        return formatted_row

if __name__ == '__main__':
    _ = ButterflyDataset(None, None, None, None, None)