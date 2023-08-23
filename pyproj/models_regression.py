import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from MLP_models import MLP
from models import *


class PreactivResNet_bn_4blks_diff_start_incDrop_mlpend_regression(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()

        # cnn_dropout=0.1
        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 *init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(16 * init_features, 4*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4*init_features, 1)

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



class AgePredModel_regression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dropout = 0.2
        self.conv1_a = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv1_b = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm1 = nn.BatchNorm3d(16)

        self.conv2_a = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2_b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(32)

        self.conv3_a = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3_b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm3 = nn.BatchNorm3d(64)

        self.dropout1 = nn.Dropout3d(dropout)
        self.linear1 = nn.Linear(in_features=39424, out_features=16)
        # relu
        self.linear2 = nn.Linear(in_features=16, out_features=32)
        # relu
        self.linear3 = nn.Linear(in_features=32, out_features=64)
        # relu
        self.final_layer = nn.Linear(in_features=64, out_features=1)


    def forward(self, x):
        # input shape = [batch size, channels, image shape]
        image, tabular = x

        out = self.conv1_a(image)
        out = F.relu(out)
        out = self.conv1_b(out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm1(out)

        out = self.conv2_a(out)
        out = F.relu(out)
        out = self.conv2_b(out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm2(out)

        out = self.conv3_a(out)
        out = F.relu(out)
        out = self.conv3_b(out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm3(out)

        out = self.dropout1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.final_layer(out)

        return out

class AgePredModel_adptPool_regression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dropout = 0.2
        self.conv1_a = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv1_b = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm1 = nn.BatchNorm3d(16)

        self.conv2_a = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2_b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(32)

        self.conv3_a = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3_b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm3 = nn.BatchNorm3d(64)

        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(4)

        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features=64 * (4 ** 3), out_features=16)
        # relu
        self.linear2 = nn.Linear(in_features=16, out_features=32)
        # relu
        self.linear3 = nn.Linear(in_features=32, out_features=64)
        # relu
        self.final_layer = nn.Linear(in_features=64, out_features=1)


    def forward(self, x):
        # input shape = [batch size, channels, image shape]
        image, tabular = x

        out = self.conv1_a(image)
        out = F.relu(out)
        out = self.conv1_b(out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm1(out)

        out = self.conv2_a(out)
        out = F.relu(out)
        out = self.conv2_b(out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm2(out)

        out = self.conv3_a(out)
        out = F.relu(out)
        out = self.conv3_b(out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm3(out)

        out = self.adaptive_avg_pool3d(out).view(out.size(0), -1)
        out = self.dropout1(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.final_layer(out)

        return out


if __name__ == "__main__":
    from data_handler import get_dataloaders
    from tformNaugment import tform_dict

    # # Create the data loaders:
    # loaders = get_dataloaders(batch_size=4, fold=0, num_workers=0,
    #                           transform=tform_dict["normalize"])
    # train_loader, valid_loader = loaders
    #
    # img, tabular, y = next(iter(train_loader))
    #
    # model = PreactivResNet_instN()
    # out = model((img, tabular))
    # out.shape