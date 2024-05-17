import torch.nn as nn
import torch.nn.functional as F
import torch
affine = True
# -----------------------------------------------------------------------------
# ---------------------- brain age prediction ---------------------------------
# -----------------------------------------------------------------------------

class Imaging_only_brainage(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

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

        self.dropout1 = nn.Dropout3d(0.2)
        self.linear1 = nn.Linear(in_features=39424, out_features=16)
        # relu
        self.linear2 = nn.Linear(in_features=16, out_features=32)
        # relu
        self.linear3 = nn.Linear(in_features=32, out_features=64)
        # relu
        self.final_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # input shape = [batch size, channels, image shape]
        img, tabular = x
        x = self.conv1_a(img)
        x = F.relu(x)
        x = self.conv1_b(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = self.batchnorm1(x)

        x = self.conv2_a(x)
        x = F.relu(x)
        x = self.conv2_b(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = self.batchnorm2(x)

        x = self.conv3_a(x)
        x = F.relu(x)
        x = self.conv3_b(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = self.batchnorm3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.final_layer(x)

        return x[:, 0]


# -----------------------------------------------------------------------------
# ------------------------ AD classification ----------------------------------
# -----------------------------------------------------------------------------
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


class PreactivResNet(nn.Module):
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
        self.fc2 = nn.Linear(4*init_features, n_outputs)

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


class MLP_8_bn_prl(nn.Module):
    def __init__(self, mlp_layers_shapes, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=mlp_layers_shapes[0], out_features=8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Linear(in_features=8, out_features=mlp_layers_shapes[-1]),
        )

        print(self)

    def forward(self, x):
        image, tabular = x
        return self.mlp(tabular)


if __name__ == "__main__":
    pass
