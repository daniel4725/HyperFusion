import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from MLP_models import MLP
from models import PreactivResNet_bn_4blks_incDrop_mlpend
import numpy as np

class img_as_hyper_tabMLP_AGE1to3(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        resnet_n_output = 32
        self.n_outputs = n_outputs

        self.resnet = PreactivResNet_bn_4blks_incDrop_mlpend(in_channels=in_channels, n_outputs=resnet_n_output, bn_momentum=bn_momentum, init_features=init_features, **kwargs)
        # self.linear_drop = nn.Dropout(0.2)

        self.num_weights = n_tabular_features * n_outputs  # num of weights and biases
        num_biases = n_outputs
        num_parameters = self.num_weights + num_biases
        self.weights_gen = nn.Linear(resnet_n_output, num_parameters)
        self.weights_shape = (n_outputs, n_tabular_features)


    def forward(self, x):
        image, tabular = x
        img_out = self.resnet(x)

        fc_parameters = self.weights_gen(img_out)  # creates #batch_size sets of parameters for the linear operation
        out = torch.zeros((tabular.shape[0], self.n_outputs), dtype=tabular.dtype, layout=tabular.layout, device=tabular.device)
        for i, param_set in enumerate(fc_parameters):
            # each input of the batch has different weights for the feedforward
            weights = param_set[: self.num_weights].reshape(self.weights_shape)
            biases = param_set[self.num_weights:]
            out[i] = F.linear(input=tabular[i], weight=weights, bias=biases)

        return out


class img_as_hyper_tabMLP_AGE1to3_drop(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        resnet_n_output = 32
        self.n_outputs = n_outputs

        self.resnet = PreactivResNet_bn_4blks_incDrop_mlpend(in_channels=in_channels, n_outputs=resnet_n_output, bn_momentum=bn_momentum, init_features=init_features, **kwargs)
        self.linear_drop = nn.Dropout(0.2)

        self.num_weights = n_tabular_features * n_outputs  # num of weights and biases
        num_biases = n_outputs
        num_parameters = self.num_weights + num_biases
        self.weights_gen = nn.Linear(resnet_n_output, num_parameters)
        self.weights_shape = (n_outputs, n_tabular_features)


    def forward(self, x):
        image, tabular = x
        img_out = self.resnet(x)
        img_out = self.linear_drop(img_out)

        fc_parameters = self.weights_gen(img_out)  # creates #batch_size sets of parameters for the linear operation
        out = torch.zeros((tabular.shape[0], self.n_outputs), dtype=tabular.dtype, layout=tabular.layout, device=tabular.device)
        for i, param_set in enumerate(fc_parameters):
            # each input of the batch has different weights for the feedforward
            weights = param_set[: self.num_weights].reshape(self.weights_shape)
            biases = param_set[self.num_weights:]
            out[i] = F.linear(input=tabular[i], weight=weights, bias=biases)

        return out



class img_as_hyper_tabMLP_set8_3(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=32, n_tabular_features=1, **kwargs):
        super().__init__()
        resnet_n_output = 32
        self.n_outputs = n_outputs

        self.resnet = PreactivResNet_bn_4blks_incDrop_mlpend(in_channels=in_channels, n_outputs=resnet_n_output, bn_momentum=bn_momentum, init_features=init_features, **kwargs)
        # self.linear_drop = nn.Dropout(0.2)

        self.num_weights = n_tabular_features * n_outputs  # num of weights and biases
        num_biases = n_outputs
        num_parameters = self.num_weights + num_biases
        self.weights_gen = nn.Linear(resnet_n_output, num_parameters)
        self.weights_shape = (n_outputs, n_tabular_features)


    def forward(self, x):
        image, tabular = x
        img_out = self.resnet(x)

        fc_parameters = self.weights_gen(img_out)  # creates #batch_size sets of parameters for the linear operation
        out = torch.zeros((tabular.shape[0], self.n_outputs), dtype=tabular.dtype, layout=tabular.layout, device=tabular.device)
        for i, param_set in enumerate(fc_parameters):
            # each input of the batch has different weights for the feedforward
            weights = param_set[: self.num_weights].reshape(self.weights_shape)
            biases = param_set[self.num_weights:]
            out[i] = F.linear(input=tabular[i], weight=weights, bias=biases)

        return out


if __name__ == '__main__':
    from data_handler import get_dataloaders
    from tformNaugment import tform_dict

    # Create the data loaders:
    loaders = get_dataloaders(batch_size=4, fold=0, num_workers=0,
         metadata_path="metadata_by_features_sets/set-8.csv")
    train_loader, valid_loader = loaders

    img, tabular, y = next(iter(train_loader))

    model = img_as_hyper_tabMLP_set8_3(n_tabular_features=train_loader.dataset.num_tabular_features)
    out = model((img, tabular))
    out.shape