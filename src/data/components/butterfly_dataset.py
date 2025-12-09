from typing import override, Dict, Any

import torch

from src.data.base_dataset import BaseDataset
from src.data.utils import open_file_as_tensor


class ButterflyDataset(BaseDataset):
    def __init__(
            self,
            df_path: str,
            modalities: list[str],
            target: bool = True,
            numericals: bool = False,
            random_state: int = 42
    ) -> None:
        super().__init__(df_path, modalities, target, numericals, 'Butterflies', random_state)

        # Placeholder for filtered columns
        columns = ['id']

        if 'coords' in self.modalities:
            columns.extend(['lat', 'lon'])

        if self.target:
            self.target_names = [c for c in self.df.columns if 'target' in c]
            columns.extend(self.target_names)
            self.num_classes = len(self.target_names)

        if self.numericals:
            self.numerical_names = [c for c in self.df.columns if c not in columns and c != 'name_loc']
            columns.extend(self.numerical_names)

        self.df.rename(columns={'name_loc': 'id'}, inplace=True)

        self.records = self.df.loc[:, columns].to_dict('records')
    @override
    def __getitem__(
            self,
            idx: int
    ) -> Dict[str, Any]:
        row = self.records[idx]

        formated_row = {'eo': {}}

        for modality in self.modalities:
            if modality in ['coords']:
                formated_row['eo'][modality] = torch.tensor([row['lat'], row['lon']])
            elif modality == 's2':
                # TODO: tensor reading
                # formated_row['eo'][modality] = open_file_as_tensor(row['id'])
                pass

        if self.target:
            formated_row['target'] = torch.tensor([row[k] for k in self.target_names], dtype=torch.float32)

        if self.numericals:
            formated_row['numericals'] = torch.tensor([row[k] for k in self.numerical_names], dtype= torch.float32)

        return formated_row

if __name__ == '__main__':
    _ = ButterflyDataset(None, None, None, None, None)