from typing import Dict, List, override

import torch
import torchvision.models as models
from torch import nn
from torchgeo.models import resnet50, ResNet50_Weights, ResNet18_Weights, resnet18

from src.models.components.geo_encoders.base_geo_encoder import BaseGeoEncoder
from src.utils.errors import IllegalArgumentCombination

RN_DIM = {
    18 : 512,
    34: 512,
    50: 2048
}

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
        geo_data_name="s2",
        input_n_bands: int | None = None,
    ) -> None:
        super().__init__()

        # Backbone configurations
        self.backbone = backbone
        if self.backbone == "resnet":
            assert resnet_version in [18, 34, 50], f"Unsupported resnet version: {resnet_version}"
            self.resnet_version = resnet_version

            assert pretrained_cnn in ["imagenet", "IMAGENET1K_V1", 'SSL4EO_RGB_MOCO', None], f"Unsupported pretrained_cnn: {pretrained_cnn}"
            self.pretrained_cnn = pretrained_cnn

            self.output_dim = RN_DIM[resnet_version]

        #  Input modality configurations
        self.allowed_geo_data_names = ["s2", "aef", "tessera"]
        assert geo_data_name in self.allowed_geo_data_names
        self.geo_data_name = geo_data_name

        self.set_n_input_bands(input_n_bands)
        assert (self.input_n_bands >= 3 and type(self.input_n_bands) is int), f"input_n_bands must be int >=3, got {self.input_n_bands}"

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

    @override
    def _setup(self) -> List[str]:
        """Gets backbone model given configuration stored in self.
        :return: backbone model
        """
        trainable_modules = []
        if self.backbone == "resnet":
            # Weights
            # SSL4EO
            if self.pretrained_cnn == "SSL4EO_RGB_MOCO":
                if self.resnet_version == 18:
                    self.geo_encoder = resnet18(weights=ResNet18_Weights.SENTINEL2_RGB_MOCO)
                elif self.resnet_version == 34:
                    raise IllegalArgumentCombination('SSL4EO_RGB_MOCO weights are not available for RN-34')
                else:
                    self.geo_encoder = resnet50(weights=ResNet50_Weights.SENTINEL2_RGB_MOCO)
            # Imagenet
            else:
                if self.pretrained_cnn == "imagenet":
                    self.pretrained_cnn = "IMAGENET1K_V1"
                elif self.pretrained_cnn == "imagenet_v2":
                    self.pretrained_cnn = "IMAGENET1K_V2"

                if self.resnet_version == 18:
                    self.geo_encoder = models.resnet18(weights=self.pretrained_cnn)
                elif self.resnet_version == 34:
                    self.geo_encoder = models.resnet34(weights=self.pretrained_cnn)
                else:
                    self.geo_encoder = models.resnet50(weights=self.pretrained_cnn)

            # Modify the first conv layer to accept input_n_bands channels
            if self.input_n_bands != 3:

                # Copy pre-trained weights
                if self.pretrained_cnn is not None:
                    weight = self.geo_encoder.conv1.weight.clone()

                # Replace 1st conv layer
                self.geo_encoder.conv1 = torch.nn.Conv2d(
                    self.input_n_bands, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

                # Copy pre-trained RGB bands
                if self.pretrained_cnn is not None:
                    with torch.no_grad():
                        for i in range(self.input_n_bands):
                            self.geo_encoder.conv1.weight[:, i, :, :] = weight[:, i % 3, :, :]

                # Ensure replaced layer is not frozen
                trainable_modules.append('geo_encoder.conv1')

            # I think for features fc often is replaced with identity?
            self.geo_encoder.fc = nn.Identity()

            # self.geo_encoder.fc = nn.Linear(self.geo_encoder.fc.in_features, self.output_dim)
            # trainable_modules.append('geo_encoder.fc')

            return trainable_modules
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
        eo_data = batch.get("eo", KeyError(f"Batch must contain batch['eo']"))
        eo_data = eo_data.get(self.geo_data_name,  KeyError(f"Batch must contain batch['eo']['{self.geo_data_name}']"))
        dtype = self.dtype

        if eo_data.dtype != dtype:
            eo_data = eo_data.to(dtype)
        feats = self.geo_encoder(eo_data)

        if self.extra_projector:
            feats = self.extra_projector(feats)

        return feats