from models import *
from HyperModels.hyper_base import *
from pl_wrap import PlModelWrap
import os

class ImageEmbedding(nn.Module):
    def __init__(self, in_channels=1, bn_momentum=0.1, init_features=4):
        super().__init__()
        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.bn = nn.BatchNorm1d(16 * init_features)
        self.prelu = nn.PReLU()
        self.output_size = 16 * init_features

    def forward(self, image):
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.bn(out)
        out = self.prelu(out)
        return out

class ImageAsHyper_TT(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        image_embedding = ImageEmbedding(in_channels=in_channels, bn_momentum=bn_momentum, init_features=init_features)
        embd_out_size = image_embedding.output_size

        fc1_hyper_kwargs = dict(embedding_model=image_embedding)
        fc2_hyper_kwargs = dict(embedding_model=image_embedding)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, n_tabular_features * 2, **fc1_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(n_tabular_features * 2)
        self.prelu = nn.PReLU()
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = LinearLayer(n_tabular_features * 2, n_outputs, **fc2_hyper_kwargs)

    def forward(self, x):
        image, tabular = x

        out = self.fc1((tabular, image))
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.drop1(out)
        out = self.fc2((out, image))
        return out



class ImageAsHyper_TF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        image_embedding = ImageEmbedding(in_channels=in_channels, bn_momentum=bn_momentum, init_features=init_features)
        embd_out_size = image_embedding.output_size

        fc1_hyper_kwargs = dict(embedding_model=image_embedding)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, n_tabular_features * 2, **fc1_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(n_tabular_features * 2)
        self.prelu = nn.PReLU()
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = LinearLayer(n_tabular_features * 2, n_outputs, **fc2_hyper_kwargs)

    def forward(self, x):
        image, tabular = x

        out = self.fc1((tabular, image))
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.drop1(out)
        out = self.fc2((out, image))
        return out


class ImageAsHyper_FT(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        image_embedding = ImageEmbedding(in_channels=in_channels, bn_momentum=bn_momentum, init_features=init_features)
        embd_out_size = image_embedding.output_size

        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=image_embedding)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, n_tabular_features * 2, **fc1_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(n_tabular_features * 2)
        self.prelu = nn.PReLU()
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = LinearLayer(n_tabular_features * 2, n_outputs, **fc2_hyper_kwargs)

    def forward(self, x):
        image, tabular = x

        out = self.fc1((tabular, image))
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.drop1(out)
        out = self.fc2((out, image))
        return out



class ImageAsHyper_T(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs

        embd_out_size = 32
        weights_init_method = None  # input_variance  embedding_variance  histogram  None
        # image_embedding = PreactivResNet_bn_4blks_incDrop_mlpend(in_channels=in_channels, n_outputs=embd_out_size, bn_momentum=bn_momentum, init_features=init_features, **kwargs)
        image_embedding = PreactivResNet_bn_4blks_diff_start_incDrop_mlpend(in_channels=in_channels, n_outputs=embd_out_size, bn_momentum=bn_momentum, init_features=init_features, **kwargs)

        fc1_hyper_kwargs = dict(embedding_model=image_embedding)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, n_outputs, **fc1_hyper_kwargs)


    def forward(self, x):
        image, tabular = x
        out = self.fc1((tabular, (image, 1)))
        return out



class ImageAsHyper_Hembd16_TF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs

        embd_out_size = 16
        weights_init_method = None  # input_variance  embedding_variance  histogram  None
        image_embedding = PreactivResNet_bn_4blks_diff_start_incDrop_mlpend(in_channels=in_channels, n_outputs=embd_out_size, bn_momentum=bn_momentum, init_features=init_features, **kwargs)

        fc1_hyper_kwargs = dict(embedding_model=image_embedding)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, 8, **fc1_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(8)
        self.prelu = nn.PReLU()
        self.fc2 = LinearLayer(8, n_outputs, **fc2_hyper_kwargs)


    def forward(self, x):
        image, tabular = x
        out = self.fc1((tabular, (image, 1)))
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.fc2((out, (image, 1)))
        return out

class ImageAsHyper_Hembd8_TF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs

        embd_out_size = 8
        weights_init_method = None  # input_variance  embedding_variance  histogram  None
        image_embedding = PreactivResNet_bn_4blks_diff_start_incDrop_mlpend(in_channels=in_channels, n_outputs=embd_out_size, bn_momentum=bn_momentum, init_features=init_features, **kwargs)

        fc1_hyper_kwargs = dict(embedding_model=image_embedding)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, 8, **fc1_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(8)
        self.prelu = nn.PReLU()
        self.fc2 = LinearLayer(8, n_outputs, **fc2_hyper_kwargs)


    def forward(self, x):
        image, tabular = x
        out = self.fc1((tabular, (image, 1)))
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.fc2((out, (image, 1)))
        return out

class ImageAsHyper_Hembd16_FT(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs

        embd_out_size = 16
        weights_init_method = None  # input_variance  embedding_variance  histogram  None
        image_embedding = PreactivResNet_bn_4blks_diff_start_incDrop_mlpend(in_channels=in_channels, n_outputs=embd_out_size, bn_momentum=bn_momentum, init_features=init_features, **kwargs)

        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=image_embedding)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, 8, **fc1_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(8)
        self.prelu = nn.PReLU()
        self.fc2 = LinearLayer(8, n_outputs, **fc2_hyper_kwargs)


    def forward(self, x):
        image, tabular = x
        out = self.fc1((tabular, (image, 1)))
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.fc2((out, (image, 1)))
        return out

class ImageAsHyper_Hembd8_FT(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs

        embd_out_size = 8
        weights_init_method = None  # input_variance  embedding_variance  histogram  None
        image_embedding = PreactivResNet_bn_4blks_diff_start_incDrop_mlpend(in_channels=in_channels, n_outputs=embd_out_size, bn_momentum=bn_momentum, init_features=init_features, **kwargs)

        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=image_embedding)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="image",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.fc1 = LinearLayer(n_tabular_features, 8, **fc1_hyper_kwargs)
        self.bn1 = nn.BatchNorm1d(8)
        self.prelu = nn.PReLU()
        self.fc2 = LinearLayer(8, n_outputs, **fc2_hyper_kwargs)


    def forward(self, x):
        image, tabular = x
        out = self.fc1((tabular, (image, 1)))
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.fc2((out, (image, 1)))
        return out