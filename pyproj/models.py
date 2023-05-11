import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from MLP_models import MLP
affine = True

def conv3d_bn3d_relu(in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1, conv_bias=True):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias)
        bn3d = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv3d, bn3d, relu)

def conv3d_instn3d_relu(in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1, conv_bias=True):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias)
        instn3d = nn.InstanceNorm3d(out_channels, affine=affine)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv3d, instn3d, relu)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1, conv_bias=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=conv_bias),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = ResBlock(init_features, init_features, bn_momentum=bn_momentum)
        self.block2 = ResBlock(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2) 
        self.block3 = ResBlock(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2)
        self.block4 = ResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    




class ResBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)

        self.dropout = nn.Dropout3d(p=dropout)
        self.relu = nn.PReLU()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=conv_bias),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.dropout(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet2(nn.Module):
    def __init__(self, in_channels=1, cnn_mlp_shapes=[3],
     bn_momentum=0.05, init_features=4, cnn_dropout=0.0,
     cnn_mlp_dropout=0.0, **kwargs):
        super().__init__()
        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = ResBlock2(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = ResBlock2(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = ResBlock2(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block4 = ResBlock2(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        cnn_mlp_shapes = [8 * init_features] + cnn_mlp_shapes
        self.ending = MLP(mlp_layers_shapes=cnn_mlp_shapes, dropout=cnn_mlp_dropout, bn_momentum=bn_momentum)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.ending(out)

        return out



class ResBlockInstnorm(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1, conv_bias=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.n1 = nn.InstanceNorm3d(out_channels, affine=affine)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias)
        self.n2 = nn.InstanceNorm3d(out_channels, affine=affine)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=conv_bias),
                nn.InstanceNorm3d(out_channels, affine=affine),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.n1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.n2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetInstnorm(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()

        self.conv_instn_relu = conv3d_instn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = ResBlockInstnorm(init_features, init_features, bn_momentum=bn_momentum)
        self.block2 = ResBlockInstnorm(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2) 
        self.block3 = ResBlockInstnorm(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2)
        self.block4 = ResBlockInstnorm(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_instn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ResNetDrop(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()
        cnn_dropout = 0.2

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = ResBlock2(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = ResBlock2(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = ResBlock2(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block4 = ResBlock2(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    




class PreactivResBlock_bn(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=conv_bias),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if not (self.downsample is None):
            identity = self.downsample(x)
        else:
            identity = x

        out = x
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


class PreactivResNet_bn(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, cnn_dropout=0.3, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class PreactivResNet_bn_3blks(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, cnn_dropout=0.3, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class PreactivResNet_bn_5blks(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, cnn_dropout=0.3, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block5 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class PreactivResNet_bn_5blks_incDrop(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()

        cnn_dropout=0.1
        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout*2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout*2)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout*3)
        self.block5 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout*3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop = nn.Dropout(cnn_dropout*2)
        self.fc = nn.Linear(16 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop(out)
        out = self.fc(out)

        return out


class PreactivResNet_bn_4blks_incDrop(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()

        cnn_dropout=0.1
        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout*2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout*2)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout*3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop = nn.Dropout(cnn_dropout*4)
        self.fc = nn.Linear(8 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop(out)
        out = self.fc(out)

        return out



class PreactivResNet_bn_4blks_incDrop_mlpend(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()

        # cnn_dropout=0.1
        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(8 * init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2*init_features, n_outputs)

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2(out)

        return out



class PreactivResBlock_instN(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=False):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=conv_bias),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if not (self.downsample is None):
            identity = self.downsample(x)
        else:
            identity = x

        out = x
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


class PreactivResNet_instN(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, cnn_dropout=0.1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_instn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_instN(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_instN(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = PreactivResBlock_instN(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block4 = PreactivResBlock_instN(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    

class PreactivResNet_instN_5blks(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, cnn_dropout=0.1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_instn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_instN(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_instN(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = PreactivResBlock_instN(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block4 = PreactivResBlock_instN(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.block5 = PreactivResBlock_instN(8 * init_features, 16 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class PreactivResNet_instN_3blks(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, cnn_dropout=0.1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_instn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_instN(init_features, init_features, bn_momentum=bn_momentum, dropout=cnn_dropout)
        self.block2 = PreactivResBlock_instN(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout) 
        self.block3 = PreactivResBlock_instN(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=cnn_dropout)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4 * init_features, n_outputs)

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    

if __name__ == "__main__":
    from data_handler import get_dataloaders
    from tformNaugment import tform_dict

    # Create the data loaders:
    loaders = get_dataloaders(batch_size=4, fold=0, num_workers=0,
                              transform=tform_dict["normalize"])
    train_loader, valid_loader = loaders

    img, tabular, y = next(iter(train_loader))

    model = PreactivResNet_instN()
    out = model((img, tabular))
    out.shape