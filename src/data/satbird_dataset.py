import logging
import os
from typing import Any, override

import numpy as np
import torch
from rasterio import open as ropen
from torchvision.transforms import v2

from src.data.base_dataset import BaseDataset
from src.data_preprocessing.satbird import setup_satbird_from_pooch
from utils.data_utils import center_crop_npy
from utils.errors import IllegalArgumentCombination

log = logging.getLogger(__name__)


class SatBirdDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        modalities: dict,
        use_target_data: bool = True,
        use_aux_data: Any = None,
        use_features: bool = False,
        seed: int = 12345,
        study_site: str = "Kenya",
        cache_dir: str = None,
        mock: bool = False,
        dtype: str = "float32",
        return_name_loc: bool = False,
        csv_name: str = None,
    ):
        """A dataset implementation for the Butterfly diversity use case.

        :param data_dir: path to data dir
        :param modalities: a list of modalities needed as EO data (for EO encoder)
        :param use_target_data: if target values should be returned
        :param use_aux_data: if auxiliary values should be returned
        :param seed: random seed
        :param cache_dir: path to cache dir
        :param study_site: study site name [Kenya, USA_summer, USA_winter]
        :param mock: whether to mock csv file
        """
        # assert study_site in ["Kenya", "USA-summer", "USA-winter"]
        assert study_site in ["Kenya", "USA-summer"]
        self.study_site = study_site

        if csv_name is not None:
            csv_name = csv_name
        else:
            csv_name = f"model_ready_satbird-{study_site}.csv"

        super().__init__(
            data_dir=data_dir,
            modalities=modalities,
            use_target_data=use_target_data,
            use_aux_data=use_aux_data,
            dataset_name=f"satbird-{study_site}",
            seed=seed,
            cache_dir=cache_dir,
            implemented_mod={"coords", "s2", "s2rgb", "aef_avr"},
            mock=mock,
            dtype=dtype,
            use_features=use_features,
            return_name_loc=return_name_loc,
            csv_name=csv_name,
        )

    @override
    def _setup(self):
        """Setups the whole dataset, makes available data of requested modalities."""

        # Set up each requested modality
        for mod in self.modalities.keys():
            if mod == "coords" and len(self.modalities.keys()) == 1:
                return
            if mod in ["s2", "s2rgb", "environmental"]:
                self.setup_satbird()
            elif mod == "tessera":
                self.setup_tessera()
            elif mod == "aef_avr":
                self.setup_embeds(mod)

    def setup_satbird(self):
        """Prepares (downloads, renames and moves) data for each requested modality."""
        print(f"\n\nSetting up SatBird {self.study_site} data...\n\n")

        check = False
        if check:
            # Check if data is already available
            dst_dirs = [
                os.path.join(self.data_dir, "eo", i) for i in ["s2", "s2rgb", "environmental"]
            ]

            # If data does not exist or is empty → full download
            for dst_dir in dst_dirs:
                if not os.path.exists(dst_dir) or len(os.listdir(dst_dir)) == 0:
                    setup_satbird_from_pooch(
                        self.data_dir, self.cache_dir, self.study_site, self.registry_path
                    )
                    return

    def center_crop_or_pad_npy(self, im, target_shape):
        """Center-crops dims larger than target, center-pads (zeros) dims smaller than target."""
        if len(im.shape) != len(target_shape):
            raise ValueError(
                f"arr has {len(im.shape)} dims but target_shape has {len(target_shape)}"
            )

        # Padding
        pad_widths = []
        for dim, target in zip(im.shape, target_shape):
            if dim < target:
                total_pad = target - dim
                before = total_pad // 2
                after = total_pad - before
                pad_widths.append((before, after))
            else:
                pad_widths.append((0, 0))
        im = np.pad(im, pad_widths, mode="constant", constant_values=0)

        # Crop
        slices = []
        for dim, target in zip(im.shape, target_shape):
            start = (dim - target) // 2  # 0 if dim == target
            end = start + target
            slices.append(slice(start, end))
        return im[tuple(slices)]

    def load_s2(self, path: str):
        """Loads S2 data from path."""
        # Modality settings
        size = self.modalities["s2"]["size"]
        np_dtype, is_bfloat16 = self.resolve_dtype(self.modalities["s2"]["dtype"])
        im = self.load_tiff(path, dtype=np.dtype("uint16"))

        if self.modalities["s2"].get("channels", "") == "4c":  # BGRNIR → RGBNIR
            im = im[[2, 1, 0, 3], :, :]
            c = 4
        elif self.modalities["s2"].get("channels", "") == "rgb":  # BGR → RGB
            im = im[[2, 1, 0], :, :]
            c = 3
        else:
            raise IllegalArgumentCombination(
                f"Channel specification {self.modalities["s2"].get("channels", 'null')} is not implemented."
            )

        if self.modalities["s2"].get("preprocessing") == "div_10000":
            im = im / 10000.0
            im = im.clip(0, 1)

        elif self.modalities["s2"].get("preprocessing") == "div_2000":
            im = im / 2000.0
            im = im.clip(0, 1)

        elif self.modalities["s2"].get("preprocessing") == "stretch_2_98":
            im = im.astype(np.float32)
            p2 = np.percentile(im, 2, axis=(1, 2), keepdims=True)
            p98 = np.percentile(im, 98, axis=(1, 2), keepdims=True)
            im = (im - p2) / np.clip(p98 - p2, 1e-6, None)
            im = im.clip(0, 1)
        else:
            log.warning("Data is not scaled.")

        im = im.astype(dtype=np_dtype)

        # Crop
        if im.shape[-2:] != (size, size):
            pad = True
            if pad:
                im = self.center_crop_or_pad_npy(im, (c, size, size))
            im = center_crop_npy(im, (c, size, size))

        tensor = torch.from_numpy(im)
        if is_bfloat16:
            tensor = tensor.to(torch.bfloat16)
        return tensor

    def load_s2rgb(self, path: str):
        img = ropen(path).read()
        tensor = v2.ToImage()(img)  # uint8, CxHxW
        tensor = v2.ToDtype(torch.float32, scale=True)(
            tensor
        )  # dtype preserved (e.g. uint8 stays uint8)
        tensor = tensor.permute(1, 2, 0)
        return tensor

    @override
    def __getitem__(self, idx):
        row = self.records[idx]

        formatted_row = {"eo": {}}

        for modality in self.modalities:
            if modality in ["coords"]:
                formatted_row["eo"][modality] = torch.tensor([row["lat"], row["lon"]])
            elif modality == "s2":
                s2 = self.load_s2(row[f"{modality}_path"])
                formatted_row["eo"][modality] = s2
            elif modality == "s2rgb":
                s2 = self.load_s2rgb(row[f"{modality}_path"])
                formatted_row["eo"][modality] = s2
            elif modality == "tessera":
                formatted_row["eo"][modality] = self.load_tessera(row["tessera_path"])
            elif modality == "aef_avr":
                formatted_row["eo"][modality] = self.aef_avr[row["name_loc"]]

        if self.use_target_data:
            formatted_row["target"] = torch.tensor(
                [row[k] for k in self.target_names], dtype=torch.float32
            )

        if self.use_aux_data:
            formatted_row["aux"] = {}
            for aux_cat, vals in self.use_aux_data.items():
                if aux_cat == "aux":
                    raw = torch.tensor([row[v] for v in vals], dtype=self.dtype)
                    formatted_row["aux"][aux_cat] = raw
                    if self._aux_mean is not None and self._aux_std is not None:
                        formatted_row["aux"]["aux_std"] = (raw - self._aux_mean) / self._aux_std
                else:
                    formatted_row["aux"][aux_cat] = [row[v] for v in vals]

        if self.use_features and self.feat_names:
            raw = torch.tensor([row[k] for k in self.feat_names], dtype=torch.float32)
            if self._feat_mean is not None and self._feat_std is not None:
                formatted_row["eo"]["tabular"] = (raw - self._feat_mean) / self._feat_std
            else:
                formatted_row["eo"]["tabular"] = raw

        if self.return_name_loc:
            formatted_row["name_loc"] = row["name_loc"]

        return formatted_row

    def plot(self, im):
        import matplotlib.pyplot as plt

        if isinstance(im, np.ndarray):
            rgb = im.transpose(1, 2, 0)
        else:
            rgb = im.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(rgb)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    _ = SatBirdDataset(None, None, None, None, None, None, None)
