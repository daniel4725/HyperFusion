import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from MLP_models import MLP
from models import ResBlock, conv3d_bn3d_relu
from models import *
import numpy as np


def Hyper4Fc(embedding_mlp_shapes=[10, 16, 8], in_features=1, out_features=256,
            bn_momentum=0.05, dropout=0.0):
    if type(embedding_mlp_shapes) == str: # "[16, 16, 3]" also works
        embedding_mlp_shapes = eval(embedding_mlp_shapes)
    
    hypernet = nn.Sequential()

    if len(embedding_mlp_shapes) > 1:  # there is embedding
        embd_out_size = embedding_mlp_shapes[-1]
        for i, (in_size, out_size) in enumerate(zip(embedding_mlp_shapes[:-1], embedding_mlp_shapes[1:])):
            # hypernet.add_module(f"bn_{i}", nn.BatchNorm1d(in_size, momentum=bn_momentum))
            hypernet.add_module(f"fc_{i}", nn.Linear(in_features=in_size, out_features=out_size)) 
            # TODO add dropout properly (place relative to bn and other layers)
            # hypernet.add_module(f"dropout_{i}",nn.Dropout(dropout))
            hypernet.add_module(f"activation_{i}", nn.PReLU())
    else:  # there is NO embedding
        embd_out_size = embedding_mlp_shapes[0]

    # final layer - generating the weights and biases
    num_weights = in_features * out_features  # num of weights and biases
    num_biases = out_features
    num_parameters = num_weights + num_biases
    hypernet.add_module("weights_gen", nn.Linear(in_features=embd_out_size, out_features=num_parameters))
    
    # TODO weights initialization
    # hypernet.weights_gen.weight.data.fill_(1)
    # hypernet.weights_gen.bias.data.fill_(0)
    # layer.weight.data = (torch.randn((out_feat, in_feat)))/100 + 1/out_feat
    # layer.bias.data = (torch.randn((out_feat)))/100

    # nn.init.kaiming_normal_(hypernet.weights_gen.weight, mode='fan_out', nonlinearity='relu')
    # hypernet.weights_gen.bias.data = (torch.randn(n_weights_out))/100

    stdv = 1. / np.sqrt(in_features)
    hypernet.weights_gen.weight.data.uniform_(-stdv, stdv) # the initialization of weights and biases of linear layer
    hypernet.weights_gen.bias.data.uniform_(-stdv/100, stdv/100) # to keep the outpot around the same distribution of the weights


    # lin = nn.Linear(in_features=in_features, out_features=out_features)
    # l = hypernet.weights_gen
    # l = lin
    # print(
    # l.weight.std(),
    # l.weight.mean(),
    # l.weight.min(),
    # l.weight.max()
    # )

    return hypernet

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, hyper=False, embedding_mlp_shapes=[10, 16, 8], bn_momentum=0.05, dropout=0.0):
        super().__init__()
        self.hyper = hyper
        if hyper:
            self.num_weights = in_features * out_features  # num of weights and biases
            self.num_biases = out_features
            self.num_out_features = out_features
            self.weights_shape = (out_features, in_features)
            n_weights_out = self.num_weights + self.num_biases
            self.hyper_net = Hyper4Fc(embedding_mlp_shapes=embedding_mlp_shapes, in_features=in_features, out_features=out_features, bn_momentum=bn_momentum, dropout=dropout)
            self.fc = None
        else:
            self.fc = nn.Linear(in_features=in_features, out_features=out_features)
            self.hyper_net = None


    def forward(self, x):
        x, features = x[0], x[1]
        
        if self.hyper:
            fc_parameters = self.hyper_net(features)  # creates #batch_size sets of parameters for the linear operation
            out = torch.zeros((x.shape[0], self.num_out_features), dtype=x.dtype, layout=x.layout, device=x.device)
            for i, param_set in enumerate(fc_parameters):
                # each input of the batch has different weights for the feedforward
                weights = param_set[: self.num_weights].reshape(self.weights_shape)
                biases = param_set[self.num_weights:]
                out[i] = F.linear(input=x[i], weight=weights, bias=biases)

        else:
            out = self.fc(x)

        return out


def Hyper4Conv3D(embedding_mlp_shapes=[10, 16, 8], in_channels=16, out_channels=32,
            kernel_size=3, bn_momentum=0.05, dropout=0.0):
    if type(embedding_mlp_shapes) == str: # "[16, 16, 3]" also works
        embedding_mlp_shapes = eval(embedding_mlp_shapes)
    
    hypernet = nn.Sequential()

    if len(embedding_mlp_shapes) > 1:  # there is embedding
        embd_out_size = embedding_mlp_shapes[-1]
        for i, (in_size, out_size) in enumerate(zip(embedding_mlp_shapes[:-1], embedding_mlp_shapes[1:])):
            # hypernet.add_module(f"bn_{i}", nn.BatchNorm1d(in_size, momentum=bn_momentum))
            hypernet.add_module(f"fc_{i}", nn.Linear(in_features=in_size, out_features=out_size)) 
            # TODO add dropout properly (place relative to bn and other layers)
            # hypernet.add_module(f"dropout_{i}",nn.Dropout(dropout))
            hypernet.add_module(f"activation_{i}", nn.PReLU())
    else:  # there is NO embedding
        embd_out_size = embedding_mlp_shapes[0]

    # final layer - generating the weights and biases
    num_weights = in_channels * out_channels * (kernel_size ** 3)
    num_biases = out_channels
    num_parameters = num_biases + num_weights
    hypernet.add_module("weights_gen", nn.Linear(in_features=embd_out_size, out_features=num_parameters))
    
    # weights initialization
    # weights_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
    # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(torch.zeros(weights_shape))
    # stdv = 1 / np.sqrt(fan_in)
    stdv = 1. / np.sqrt(in_channels * (kernel_size ** 3))
    hypernet.weights_gen.weight.data.uniform_(-stdv, stdv)  # the initialization of weights and biases of conv3d
    hypernet.weights_gen.bias.data.uniform_(-stdv/100, stdv/100)  # to keep the outpot around the same distribution of the weights
    # hypernet.weights_gen.bias.data.uniform_(-stdv, stdv)

    # the default pytorch initialization
    # def reset_parameters(self) -> None:
    #     # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    #     # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
    #     # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         if fan_in != 0:
    #             bound = 1 / math.sqrt(fan_in)
    #             init.uniform_(self.bias, -bound, bound)

    # conv = nn.Conv3d(in_channels, out_channels, kernel_size)
    # l = hypernet.weights_gen
    # l = conv
    # print(
    # l.weight.std(),
    # l.weight.mean(),
    # l.weight.min(),
    # l.weight.max(),
    # l.bias.std(),
    # l.bias.mean(),
    # l.bias.min(),
    # l.bias.max()
    # )

    return hypernet


class Conv3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3,  stride=1, padding=1,
                hyper=False, embedding_mlp_shapes=[10, 16, 8], bn_momentum=0.05, dropout=0.0):
        super().__init__()
        self.hyper = hyper
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if hyper:
            self.num_weights = in_channels * out_channels * (kernel_size ** 3)  # num of weights and biases
            self.num_biases = out_channels
            self.num_out_channels = out_channels
            self.weights_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
            self.hyper_net = Hyper4Conv3D(embedding_mlp_shapes=embedding_mlp_shapes, in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, bn_momentum=bn_momentum, dropout=dropout)
            self.conv3d = None
        else:
            
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.hyper_net = None


    def forward(self, x):
        x, features = x[0], x[1]
        
        if self.hyper:
            conv_parameters = self.hyper_net(features)  # creates #batch_size sets of parameters for the linear operation
            
            # first sample forward to determine the shape of the output
            param_set = conv_parameters[0]
            weights = param_set[: self.num_weights].reshape(self.weights_shape)
            biases = param_set[self.num_weights:]
            out0 = F.conv3d(input=x[0][None], weight=weights, bias=biases, stride=self.stride, padding=self.padding)

            out = torch.zeros([x.shape[0]] +  list(out0.shape[1:]), dtype=x.dtype, layout=x.layout, device=x.device)
            out[0] = out0
            for i, param_set in enumerate(conv_parameters[1:]):
                # each input of the batch has different weights for the feedforward
                weights = param_set[: self.num_weights].reshape(self.weights_shape)
                biases = param_set[self.num_weights:]
                out[i+1] = F.conv3d(input=x[i][None], weight=weights, bias=biases, stride=self.stride, padding=self.padding)
                
        else:
            out = self.conv3d(x)

        return out




class HE_RSNT_age_noembd(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out



class HE_RSNT_age_noembd_lastHyp(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class HE_RSNT_age_noembd_lastHyp_smothedAGE(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=38, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])

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
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class PreactivResBlock_bn_hyper_TTT(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True,
                embedding_mlp_shapes=[10, 16, 8]):
        super().__init__()
        self.hyper1 = True
        self.hyper2 = True
        self.hyperdownsample = True

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
        x, features = x
        if not (self.downsample is None):
            identity = self.downsample((x, features))
        else:
            identity = x

        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1((out, features))

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2((out, features))

        out += identity
        return out

class RSNT_age_noembd_lastblockHyp_TTT(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_TTT(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs)

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class PreactivResBlock_bn_hyper_FFT(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True,
                embedding_mlp_shapes=[10, 16, 8]):
        super().__init__()
        self.hyper1 = False
        self.hyper2 = False
        self.hyperdownsample = True

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
        x, features = x
        if not (self.downsample is None):
            identity = self.downsample((x, features))
        else:
            identity = x

        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1((out, features))

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2((out, features))

        out += identity
        return out

class RSNT_age_noembd_lastblockHyp_FFT(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_FFT(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs)

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out

class PreactivResBlock_bn_hyper_TTF(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True,
                embedding_mlp_shapes=[10, 16, 8]):
        super().__init__()
        self.hyper1 = True
        self.hyper2 = True
        self.hyperdownsample = False

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
        x, features = x
        if not (self.downsample is None):
            identity = self.downsample((x, features))
        else:
            identity = x

        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1((out, features))

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2((out, features))

        out += identity
        return out

class RSNT_age_noembd_lastblockHyp_TTF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_TTF(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs)

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out












# --------------------------lastblockHyp_TTF_smthAGE_embd_38_5_sumto1-----------------

class Hyper4Conv3D_sumto1(nn.Module):
    def __init__(self, embedding_mlp_shapes=[10, 16, 8], in_channels=16, out_channels=32,
            kernel_size=3, bn_momentum=0.05, dropout=0.0):
        super().__init__()
        if type(embedding_mlp_shapes) == str: # "[16, 16, 3]" also works
            embedding_mlp_shapes = eval(embedding_mlp_shapes)
        
        hypernet = nn.Sequential()

        if len(embedding_mlp_shapes) > 1:  # there is embedding
            embd_out_size = embedding_mlp_shapes[-1]
            for i, (in_size, out_size) in enumerate(zip(embedding_mlp_shapes[:-1], embedding_mlp_shapes[1:])):
                # hypernet.add_module(f"bn_{i}", nn.BatchNorm1d(in_size, momentum=bn_momentum))
                hypernet.add_module(f"fc_{i}", nn.Linear(in_features=in_size, out_features=out_size)) 
                # TODO add dropout properly (place relative to bn and other layers)
                # hypernet.add_module(f"dropout_{i}",nn.Dropout(dropout))
                hypernet.add_module(f"activation_{i}", nn.PReLU())
        else:  # there is NO embedding
            embd_out_size = embedding_mlp_shapes[0]

        self.hypernet = hypernet
        # final layer - generating the weights and biases
        num_weights = in_channels * out_channels * (kernel_size ** 3)
        num_biases = out_channels
        num_parameters = num_biases + num_weights
        self.weights_gen = nn.Linear(in_features=embd_out_size, out_features=num_parameters)
        
        # weights initialization
        stdv = 1. / np.sqrt(in_channels * (kernel_size ** 3))
        self.weights_gen.weight.data.uniform_(-stdv, stdv)  # the initialization of weights and biases of conv3d
        self.weights_gen.bias.data.uniform_(-stdv/100, stdv/100)  # to keep the outpot around the same distribution of the weights

    def forward(self, x):
        out = self.hypernet(x)
        out = out/out.sum()
        out = self.weights_gen(out)
        return out


class Conv3DLayer_sumto1(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3,  stride=1, padding=1,
                hyper=False, embedding_mlp_shapes=[10, 16, 8], bn_momentum=0.05, dropout=0.0):
        super().__init__()
        self.hyper = hyper
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if hyper:
            self.num_weights = in_channels * out_channels * (kernel_size ** 3)  # num of weights and biases
            self.num_biases = out_channels
            self.num_out_channels = out_channels
            self.weights_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
            self.hyper_net = Hyper4Conv3D_sumto1(embedding_mlp_shapes=embedding_mlp_shapes, in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, bn_momentum=bn_momentum, dropout=dropout)
            self.conv3d = None
        else:
            
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.hyper_net = None


    def forward(self, x):
        x, features = x[0], x[1]
        
        if self.hyper:
            conv_parameters = self.hyper_net(features)  # creates #batch_size sets of parameters for the linear operation
            
            # first sample forward to determine the shape of the output
            param_set = conv_parameters[0]
            weights = param_set[: self.num_weights].reshape(self.weights_shape)
            biases = param_set[self.num_weights:]
            out0 = F.conv3d(input=x[0][None], weight=weights, bias=biases, stride=self.stride, padding=self.padding)

            out = torch.zeros([x.shape[0]] +  list(out0.shape[1:]), dtype=x.dtype, layout=x.layout, device=x.device)
            out[0] = out0
            for i, param_set in enumerate(conv_parameters[1:]):
                # each input of the batch has different weights for the feedforward
                weights = param_set[: self.num_weights].reshape(self.weights_shape)
                biases = param_set[self.num_weights:]
                out[i+1] = F.conv3d(input=x[i][None], weight=weights, bias=biases, stride=self.stride, padding=self.padding)
                
        else:
            out = self.conv3d(x)

        return out



class PreactivResBlock_bn_hyper_TTF_sumto1(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True,
                embedding_mlp_shapes=[10, 16, 8]):
        super().__init__()
        self.hyper1 = True
        self.hyper2 = True
        self.hyperdownsample = False

        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = Conv3DLayer_sumto1(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                hyper=self.hyper1, embedding_mlp_shapes=embedding_mlp_shapes)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = Conv3DLayer_sumto1(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 hyper=self.hyper2, embedding_mlp_shapes=embedding_mlp_shapes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv3DLayer_sumto1(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                            hyper=self.hyperdownsample, embedding_mlp_shapes=embedding_mlp_shapes),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        x, features = x
        if not (self.downsample is None):
            identity = self.downsample((x, features))
        else:
            identity = x

        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1((out, features))

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2((out, features))

        out += identity
        return out

class lastblockHyp_TTF_smthAGE_embd_38_5_sumto1(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=38, **kwargs):
        super().__init__()
        assert n_tabular_features==38

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_TTF_sumto1(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features, 5])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs)

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


# ------------------------ combine conv hyper and fc hyper --------------------
class age_noembd_lastblockHyp_FFT_fcHyp_both(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_FFT(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class age_noembd_lastblockHyp_TTF_fcHyp_both(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_TTF(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out

class age_noembd_lastblockHyp_FFT_fcHyp_2(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_FFT(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class age_noembd_lastblockHyp_TTF_fcHyp_2(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2) 
        self.block1 = PreactivResBlock_bn(init_features, init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2) 
        self.block3 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn_hyper_TTF(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3,
                        embedding_mlp_shapes=[n_tabular_features])
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(8*init_features, 2*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(2*init_features, n_outputs, hyper=True,
                    embedding_mlp_shapes=[n_tabular_features])

        self.relu = nn.ReLU()


    def forward(self, x):
        image, tabular = x
        tabular = tabular + 1

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out



if __name__ == '__main__':
    from data_handler import get_dataloaders
    from tformNaugment import tform_dict

    # Create the data loaders:
    loaders = get_dataloaders(batch_size=4, fold=0, num_workers=0,
         metadata_path="metadata_by_features_sets/set-6.csv")
    train_loader, valid_loader = loaders

    img, tabular, y = next(iter(train_loader))

    model = lastblockHyp_TTF_smthAGE_embd_38_5_sumto1()
    out = model((img, tabular))
    out.shape