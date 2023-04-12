# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from Film_DAFT.model_base import BaseModel
from Film_DAFT.vol_blocks import ConvBnReLU, DAFTBlock, FilmBlock
from models import PreactivResBlock_bn


class FilmHNN_preactive_block(BaseModel):
    """
    adapted version of Perez et al. (AAAI, 2018)
    https://arxiv.org/abs/1709.07871
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_outputs: int = 3,
        bn_momentum: float = 0.1,
        init_features: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
        n_tabular_features=1,
        **kwargs
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {"ndim_non_img": n_tabular_features}

        self.split_size = 4 * init_features
        self.conv1 = ConvBnReLU(in_channels, init_features, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)  # 16
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)  # 8
        self.blockX = FilmBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2*init_features, n_outputs)

        self.relu = nn.ReLU()

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, x):
        image, tabular = x
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2(out)
        return out


class DAFT_preactive_block(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        n_outputs: int = 3,
        bn_momentum: float = 0.1,
        init_features: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
        n_tabular_features=1,
        **kwargs
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {"ndim_non_img": n_tabular_features}

        self.split_size = 4 * init_features
        self.conv1 = ConvBnReLU(in_channels, init_features, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)  # 16
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)  # 8
        self.blockX = DAFTBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2*init_features, n_outputs)

        self.relu = nn.ReLU()

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, x):
        image, tabular = x
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2(out)
        return out