from models import *
from models_hyper import Hyper4Fc, LinearLayer, Hyper4Conv3D, Conv3DLayer
from Film_DAFT_preactive.model_base import BaseModel
from Film_DAFT_preactive.vol_blocks import DAFTBlock, FilmBlock
from typing import Any, Dict, Optional, Sequence
import torch.nn as nn


class hyperMIX_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True,
                embedding_mlp_shapes=[10, 16, 8], hypers_bool=(False, False, False)):
        super().__init__()
        self.hyper1 = hypers_bool[0]
        self.hyper2 = hypers_bool[1]
        self.hyperdownsample = hypers_bool[2]

        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = Conv3DLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                hyper=self.hyper1, embedding_mlp_shapes=embedding_mlp_shapes)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = Conv3DLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 hyper=self.hyper2, embedding_mlp_shapes=embedding_mlp_shapes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv3DLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                            hyper=self.hyperdownsample, embedding_mlp_shapes=embedding_mlp_shapes),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None


    def forward(self, x):
        x, tabular = x


        if not (self.downsample is None):
            identity = self.downsample((x, tabular))
        else:
            identity = x

        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1((out, tabular))

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2((out, tabular))

        out += identity
        return out


class HyperMIX(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1,**kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = hyperMIX_ResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[4 * init_features + n_tabular_features] + kwargs["hiddenMLP4hyperMIX_block"], hypers_bool=kwargs["hypers_bool4hyperMIX_block"])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features,
                               hyper=kwargs["hypers_bool4fc1"], embedding_mlp_shapes=[8 * init_features + n_tabular_features] + kwargs["hiddenMLP4fc1"])
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs,
                               hyper=kwargs["hypers_bool4fc2"], embedding_mlp_shapes=[2 * init_features + n_tabular_features] + kwargs["hiddenMLP4fc2"])

        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        # concat between the tabular and the img data to mix them
        squeeze1 = self.global_pool(out)
        squeeze1 = squeeze1.view(squeeze1.size(0), -1)
        squeeze1 = torch.cat((squeeze1, tabular), dim=1)

        out = self.block4((out, squeeze1))

        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        squeeze2 = torch.cat((out, tabular), dim=1)
        out = self.fc1((out, squeeze2))
        out = self.relu(out)

        out = self.linear_drop2(out)
        squeeze3 = torch.cat((out, tabular), dim=1)
        out = self.fc2((out, squeeze3))

        return out


class HyperMIX_FFF_noembd_TT_noembd(HyperMIX):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1,**kwargs):
        kwargs.update({
            "hypers_bool4hyperMIX_block": (False, False, False),
            "hiddenMLP4hyperMIX_block": [],
            "hypers_bool4fc1": True,
            "hiddenMLP4fc1": [],
            "hypers_bool4fc2": True,
            "hiddenMLP4fc2": []

        })
        super().__init__(in_channels=in_channels, n_outputs=n_outputs, bn_momentum=bn_momentum,
                         init_features=init_features, n_tabular_features=n_tabular_features, **kwargs)

class HyperMIX_TFF_noembd_FF_noembd(HyperMIX):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1,**kwargs):
        kwargs.update({
            "hypers_bool4hyperMIX_block": (True, False, False),
            "hiddenMLP4hyperMIX_block": [],
            "hypers_bool4fc1": False,
            "hiddenMLP4fc1": [],
            "hypers_bool4fc2": False,
            "hiddenMLP4fc2": []

        })
        super().__init__(in_channels=in_channels, n_outputs=n_outputs, bn_momentum=bn_momentum,
                         init_features=init_features, n_tabular_features=n_tabular_features, **kwargs)

if __name__ == '__main__':
    from data_handler import get_dataloaders
    from tformNaugment import tform_dict
    ADNI_dir = "/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
    # ADNI_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/ADNI"
    # ADNI_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/zipped_processed_data/ADNI"
    metadata_path = "metadata_by_features_sets/set-8.csv"
    num_workers = 0
    load2ram = False
    data_fold = 0
    tform = "hippo_crop_lNr_l2r_tst"  # hippo_crop_2sides, center_crop  hippo_crop_lNr  hippo_crop_lNr_l2r


    loaders = get_dataloaders(batch_size=4, adni_dir=ADNI_dir, load2ram=load2ram,
                              metadata_path=metadata_path, fold=data_fold,
                              transform_train=tform_dict["hippo_crop_lNr"],
                              transform_valid=tform_dict["hippo_crop_2sides"],
                              with_skull=False, no_bias_field_correct=True, num_workers=num_workers)

    train_loader, valid_loader = loaders
    model = HyperMIX_FFF_noembd_hypfc12_noembd(in_channels=1, n_outputs=3, init_features=32, n_tabular_features=6)


    img, tabular, y = next(iter(train_loader))

    out = model((img, tabular))
    print(out.shape)