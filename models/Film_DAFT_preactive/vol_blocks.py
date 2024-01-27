from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class FilmBase(nn.Module, metaclass=ABCMeta):
    """Absract base class for models that are related to FiLM of Perez et al"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float,
        stride: int,
        ndim_non_img: int,
        location: int,
        activation: str,
        scale: bool,
        shift: bool,
    ) -> None:

        super().__init__()

        # sanity checks
        if location not in set(range(5)):
            raise ValueError(f"Invalid location specified: {location}")
        if activation not in {"tanh", "sigmoid", "linear"}:
            raise ValueError(f"Invalid location specified: {location}")
        if (not isinstance(scale, bool) or not isinstance(shift, bool)) or (not scale and not shift):
            raise ValueError(
                f"scale and shift must be of type bool:\n    -> scale value: {scale}, "
                "scale type {type(scale)}\n    -> shift value: {shift}, shift type: {type(shift)}"
            )
        # ResBlock

        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=0.3)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True),
            )
        else:
            self.downsample = None

        self.global_pool = nn.AdaptiveAvgPool3d(1)


        # Film-specific variables
        self.location = location
        if self.location == 2 and self.downsample is None:
            raise ValueError("This is equivalent to location=1 and no downsampling!")
        # location decoding
        self.film_dims = 0
        if location in {0, 1, 2}:
            self.film_dims = in_channels
        elif location in {3, 4}:
            self.film_dims = out_channels
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

    @abstractmethod
    def rescale_features(self, feature_map, x_aux):
        """method to recalibrate feature map x"""

    def forward(self, feature_map, x_aux):

        if not (self.downsample is None):
            identity = self.downsample(feature_map)
        else:
            identity = feature_map

        out = self.rescale_features(feature_map, x_aux)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += identity
        return out


class FilmBlock(FilmBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.05,
        stride: int = 2,
        ndim_non_img: int = 15,
        location: int = 2,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 7,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None
        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        attention = self.aux(x_aux)

        assert (attention.size(0) == feature_map.size(0)) and (
            attention.dim() == 2
        ), f"Invalid size of output tensor of auxiliary network: {attention.size()}"

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift


class DAFTBlock(FilmBase):
    # Block for ZeCatNet
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.05,
        stride: int = 2,
        ndim_non_img: int = 15,
        location: int = 2,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
        bottleneck_dim: int = 7,
    ) -> None:

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        self.bottleneck_dim = bottleneck_dim
        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise AssertionError(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift