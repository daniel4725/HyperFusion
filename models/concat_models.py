from models.base_models import *

# -----------------------------------------------------------------------------
# ---------------------- brain age prediction ---------------------------------
# -----------------------------------------------------------------------------


class Brainage_concat(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()

        self.conv1_a = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv1_b = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.batchnorm1 = nn.BatchNorm3d(16)

        self.conv2_a = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2_b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(32)

        self.conv3_a = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3_b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm3d(64)

        self.dropout1 = nn.Dropout3d(dropout)
        self.linear1 = nn.Linear(in_features=39424, out_features=16)
        self.linear2 = nn.Linear(in_features=18, out_features=32)
        self.linear3 = nn.Linear(in_features=32, out_features=64)
        self.final_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
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
        out = torch.cat((out, tabular), dim=1)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.final_layer(out)

        return out[:, 0]


# -----------------------------------------------------------------------------
# ------------------------ AD classification ----------------------------------
# -----------------------------------------------------------------------------


class RES_Tab_concat1(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(16 * init_features + n_tabular_features, 4*init_features)
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
        out = torch.cat((out, tabular), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2(out)

        return out