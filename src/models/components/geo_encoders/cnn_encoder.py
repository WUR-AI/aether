from typing import Dict, override

import torch
import torchvision.models as models
from torch import nn
from torch.nn import functional as F

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder


class CNNEncoder(BaseGeoEncoder):
    """Convolutional neural network EO encoder. Adapted from PECL.

    :param backbone: backbone model to use (resnet)
    :param pretrained_cnn: pretrained weights to use (imagenet or None)
    :param resnet_version: resnet version to use (18, 34, 50)
    :param freezing_strategy: freezing strategy to use (all, none)
    :param eo_data_name: name of the EO data modality (s2, aef, tessera)
    :param output_dim: output dimension of the encoder
    """

    def __init__(
        self,
        backbone="resnet",
        pretrained_cnn="imagenet",
        resnet_version=18,
        freezing_strategy="all",
        geo_data_name="s2",
        input_n_bands: int | None = None,
        output_dim=512,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.pretrained_cnn = pretrained_cnn
        self.resnet_version = resnet_version
        self.freezing_strategy = freezing_strategy

        self.allowed_geo_data_names = ["s2", "aef", "tessera"]
        assert geo_data_name in self.allowed_geo_data_names
        self.geo_data_name = geo_data_name

        self.set_n_input_bands(input_n_bands)
        assert (
            self.input_n_bands >= 3 and type(self.input_n_bands) is int
        ), f"input_n_bands must be int >=3, got {self.input_n_bands}"
        self.output_dim = output_dim

        self.geo_encoder = self.get_backbone()

    def set_n_input_bands(self, n_bands: int | None = None) -> None:
        """Sets number of input bands based on geo_data_name if n_bands is None.

        :param n_bands: number of input bands
        :return: None
        """
        if n_bands is None:  # infer from geo_data_name
            if self.geo_data_name == "s2":
                self.input_n_bands = 4
            elif self.geo_data_name == "aef":
                self.input_n_bands = 64
            elif self.geo_data_name == "tessera":
                self.input_n_bands = 128
            print(
                f"[CNNEncoder] Inferred input_n_bands={self.input_n_bands} for geo_data_name='{self.geo_data_name}'"
            )
        else:
            self.input_n_bands = n_bands
        return None

    def get_backbone(self):
        """Gets backbone model given configuration stored in self.

        :return: backbone model
        """
        if self.backbone == "resnet":
            assert self.resnet_version in [
                18,
                34,
                50,
            ], f"Unsupported resnet version: {self.resnet_version}"
            assert self.pretrained_cnn in [
                "imagenet",
                "IMAGENET1K_V1",
                None,
            ], f"Unsupported pretrained_cnn: {self.pretrained_cnn}"
            if self.pretrained_cnn == "imagenet":
                self.pretrained_cnn = "IMAGENET1K_V1"
            if self.resnet_version == 18:
                model = models.resnet18(weights=self.pretrained_cnn)
            elif self.resnet_version == 34:
                model = models.resnet34(weights=self.pretrained_cnn)
            elif self.resnet_version == 50:
                model = models.resnet50(weights=self.pretrained_cnn)
            else:
                raise ValueError(f"Unsupported resnet version: {self.resnet_version}")

            # Modify the first conv layer to accept input_n_bands channels
            if self.pretrained_cnn is not None and self.input_n_bands != 3:
                weight = model.conv1.weight.clone()
            if self.input_n_bands != 3:
                model.conv1 = torch.nn.Conv2d(
                    self.input_n_bands, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                if self.pretrained_cnn is not None:  # copy pre-trained RGB bands
                    for i in range(self.input_n_bands):
                        model.conv1.weight.data[:, i, :, :] = weight[
                            :, i % 3, :, :
                        ]  # ensure this is not frozen
            model.fc = nn.Linear(model.fc.in_features, self.output_dim)

            assert self.freezing_strategy in [
                "all",
                "none",
            ], f"Unsupported freezing_strategy: {self.freezing_strategy}"
            layers_resnet = list(model.children())
            n_layers = len(layers_resnet)
            for i_c, child in enumerate(layers_resnet):
                if i_c == 0:  # train first layer if not 3 bands (or no freezing)
                    train_if = self.freezing_strategy == "none" or self.input_n_bands != 3
                    for param in child.parameters():
                        param.requires_grad = train_if
                elif i_c == n_layers - 1:  # always train last layer
                    for param in child.parameters():
                        param.requires_grad = True
                else:  # train other layers if no freezing
                    train_if = self.freezing_strategy == "none"
                    for param in child.parameters():
                        param.requires_grad = train_if

            return model
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")

    @override
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass of the CNN encoder.

        :param batch: input batch
        :return: extracted features
        """
        eo_data = batch.get("eo", {})

        dtype = self.dtype
        if eo_data.dtype != dtype:
            eo_data = eo_data.to(dtype)
        feats = self.geo_encoder(eo_data[self.geo_data_name])
        # n_nans = torch.sum(torch.isnan(feats)).item()
        # assert (
        #     n_nans == 0
        # ), f"CNNEncoder output contains {n_nans}/{feats.numel()} NaNs PRIOR to normalization with data min {eo_data[self.geo_data_name].min()} and max {eo_data[self.geo_data_name].max()}."

        return feats.to(dtype)


if __name__ == "__main__":
    _ = CNNEncoder(None, None, None, None, None, None, None, None)
