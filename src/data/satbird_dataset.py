import os

from src.data.base_dataset import BaseDataset
from src.data_preprocessing.satbird import get_satbird_data


class SatBirdDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool,
        use_aux_data: bool,
        seed: int,
        study_site: str,
        cache_dir: str = None,
    ):
        # assert study_site in ["Kenya", "USA_summer", "USA_winter"]
        assert study_site in ["Kenya"]

        super().__init__(
            data_dir,
            modalities,
            use_target_data,
            use_aux_data,
            f"satbird-{study_site}",
            seed,
            cache_dir,
        )

        self.study_site = study_site

        # Placeholder for filtered columns
        columns = ["name_loc"]

        for modality, params in self.modalities.items():
            if modality == "coords":
                columns.extend(["lat", "lon"])
            else:
                self.add_modality_paths_to_df(modality, params["format"])
                columns.append(f"{modality}_path")

        if self.use_target_data:
            self.target_names = [c for c in self.df.columns if "target_" in c]
            columns.extend(self.target_names)
            self.num_classes = len(self.target_names)

        if self.use_aux_data:
            self.aux_names = [c for c in self.df.columns if "aux_" in c]
            columns.extend(self.aux_names)

        self.records = self.df.loc[:, columns].to_dict("records")

    def setup(self):
        # for mod in self.modalities.keys():
        #     dst_dir = os.path.join(self.data_dir, "eo", mod)
        #     if os.path.exists(dst_dir):
        #         dst_files = os.listdir(dst_dir)
        #         for p in self.df[f'{mod}_path']:
        #             if not os.path.basename(p) in dst_files:
        #                 raise FileNotFoundError(f"Missing S2 data: {p}")

        get_satbird_data(self.data_dir, self.cache_dir, self.study_site)

    def __getitem__(self, idx):
        record = self.records[idx]
        return record


if __name__ == "__main__":
    _ = SatBirdDataset(None, None, None, None, None, None, None)
    # t = SatBirdDataset('../../data', {'s2':{'format':'tif'}}, True, True, 12, "Kenya", '../../data/cache')
    # t.setup()
