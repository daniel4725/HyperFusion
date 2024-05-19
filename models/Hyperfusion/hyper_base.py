import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .hyper_hist_init import hypernet_histloss_init


class HyperNetwork(nn.Module):
    def __init__(self, embedding_model, embedding_output_size, num_weights, num_biases):
        super().__init__()
        self.embedding_model = embedding_model
        self.embedding_model_params = [param for param in embedding_model.parameters() if param.requires_grad]
        self.num_weights = num_weights
        self.num_biases = num_biases
        self.weights_gen = nn.Linear(in_features=embedding_output_size, out_features=self.num_weights)
        self.bias_gen = nn.Linear(in_features=embedding_output_size, out_features=num_biases)
        self.parameters_generators_input_size = embedding_output_size

        # self.scale_weights = True
        # self.scale_factor_net = nn.Sequential(
        #     copy.deepcopy(embedding_model),
        #     nn.Linear(embedding_output_size, 2)
        # )


    def calc_variance4init(self, main_net_in_size, train_dataloader, hyper_input_type,
                           embd_vars=False, main_net_relu=True, main_net_biasses=True, var_hypernet_input=None):
        # initialize the weights and biasses of the weights geneerator
        if var_hypernet_input is None:
            # according to PRINCIPLED WEIGHT INITIALIZATION FOR HYPERNETWORKS
            hyper_input_type_dict = {"image": 0, "tabular": 1}
            if hyper_input_type == "tabular":
                only_tabular = train_dataloader.dataset.only_tabular
                train_dataloader.dataset.only_tabular = True
            variances = []
            for batch in iter(train_dataloader):
                # to choose the input for the hyper network - (image or tabular)
                values = batch[hyper_input_type_dict[hyper_input_type]]
                if embd_vars:  # calculates tha variance after the embedding model
                    values = self.embedding_model(values)
                for v in values:
                    variances += [np.array(v.view(-1).detach().cpu()).var()]
            if hyper_input_type == "tabular":
                train_dataloader.dataset.only_tabular = only_tabular

            var_hypernet_input = np.mean(variances)
            if var_hypernet_input == 0:
                var_hypernet_input = 1

        # calculate the needed variance
        dk = self.parameters_generators_input_size  # both dk and dl
        dj = main_net_in_size
        var_weights_generator = (2 ** main_net_relu) / ((2 ** main_net_biasses) * dj * dk * var_hypernet_input)
        var_biasses_generator = (2 ** main_net_relu) / (2 * dk * var_hypernet_input)
        return var_weights_generator, var_biasses_generator

    def variance_uniform_init(self, var_weights_generator, var_biasses_generator):
        # initialize the weights and biasses of the weights geneerator
        # according to PRINCIPLED WEIGHT INITIALIZATION FOR HYPERNETWORKS

        # apply the initialization
        ws_init = np.sqrt(3 * var_weights_generator)
        bs_init = np.sqrt(3 * var_biasses_generator)
        nn.init.uniform_(self.weights_gen.weight, -ws_init, ws_init)
        nn.init.uniform_(self.bias_gen.weight, -bs_init, bs_init)

        # init the biasses of the weights and biasses generators with 0
        nn.init.constant_(self.weights_gen.bias, 0)
        nn.init.constant_(self.bias_gen.bias, 0)

    def initialize_parameters(self, weights_init_method, fan_in, hyper_input_type,
                              for_conv=False, train_loader=None, GPU=None, var_hypernet_input=None):
        if weights_init_method == "input_variance":
            print("input_variance weights initialization")
            var_w, var_b = self.calc_variance4init(fan_in, train_loader, hyper_input_type, embd_vars=False,
                                                   var_hypernet_input=var_hypernet_input)
            self.variance_uniform_init(var_w, var_b)
        elif weights_init_method == "embedding_variance":
            print("embedding_variance weights initialization")
            var_w, var_b = self.calc_variance4init(fan_in, train_loader, hyper_input_type, embd_vars=True)
            self.variance_uniform_init(var_w, var_b)
        elif weights_init_method == "histogram":
            print("histogram weights initialization")
            hypernet_histloss_init(self, fan_in, train_loader, GPU)
        else:
            raise ValueError("HyperNetwork initialization type not implemented!")

    def freeze_embedding_model(self):
        for param in self.embedding_model_params:
            param.requires_grad = False

    def unfreeze_embedding_model(self):
        for param in self.embedding_model_params:
            param.requires_grad = True

    def forward(self, x):
        emb_out = self.embedding_model(x)
        weights = self.weights_gen(emb_out)
        biases = self.bias_gen(emb_out)
        return weights, biases


class HyperLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, embedding_model, embedding_output_size,
                 weights_init_method=None, train_loader=None, hyper_input_type=None, GPU=None, var_hypernet_input=None):
        super().__init__()
        num_weights = in_features * out_features
        num_biases = out_features
        self.hyper_net = HyperNetwork(embedding_model, embedding_output_size, num_weights, num_biases)

        self.num_out_features = out_features
        self.weights_shape = (out_features, in_features)

        # initialize the weights of the layer if there is weights_init_method value
        if not (weights_init_method is None):
            self.hyper_net.initialize_parameters(weights_init_method, in_features, hyper_input_type,
                                                 for_conv=False, train_loader=train_loader, GPU=GPU,
                                                 var_hypernet_input=var_hypernet_input)

    def forward(self, x):
        x, features = x[0], x[1]

        weights, biases = self.hyper_net(features)  # creates #batch_size sets of parameters for the linear operation
        out = torch.zeros((x.shape[0], self.num_out_features), dtype=x.dtype, layout=x.layout, device=x.device)
        for i, (w, b) in enumerate(zip(weights, biases)):
            # each input of the batch has different weights for the feedforward
            w = w.reshape(self.weights_shape)
            out[i] = F.linear(input=x[i], weight=w, bias=b)

        return out



class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, embedding_model=None, embedding_output_size=None,
                 weights_init_method=None, train_loader=None, hyper_input_type=None, GPU=None,
                 var_hypernet_input=None):
        super().__init__()
        self.hyper = not(embedding_model is None)  # if there is an embedding model that hyper is True
        if self.hyper:
            self.layer = HyperLinearLayer(in_features=in_features, out_features=out_features,
                                          embedding_model=embedding_model, embedding_output_size=embedding_output_size,
                                          weights_init_method=weights_init_method, train_loader=train_loader,
                                          hyper_input_type=hyper_input_type, GPU=GPU, var_hypernet_input=var_hypernet_input)
        else:
            self.layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        if not self.hyper:
            x, features = x[0], x[1]
        return self.layer(x)


class HyperConv3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, embedding_model, embedding_output_size,
                 weights_init_method=None, train_loader=None, hyper_input_type=None, GPU=None, var_hypernet_input=None,
                 stride=1, padding=1):
        super().__init__()
        num_weights = in_channels * out_channels * (kernel_size ** 3)  # num of weights and biases
        num_biases = out_channels
        self.stride = stride
        self.padding = padding

        self.hyper_net = HyperNetwork(embedding_model, embedding_output_size, num_weights, num_biases)

        self.num_out_channels = out_channels
        self.weights_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)

        # initialize the weights of the layer if there is weights_init_method value
        if not (weights_init_method is None):
            fan_in = in_channels * (kernel_size ** 3)
            self.hyper_net.initialize_parameters(weights_init_method, fan_in, hyper_input_type,
                                                 for_conv=True, train_loader=train_loader, GPU=GPU,
                                                 var_hypernet_input=var_hypernet_input)

    def forward(self, x):
        x, features = x[0], x[1]

        weights, biases = self.hyper_net(features)  # creates #batch_size sets of parameters for the linear operation

        # first sample forward to determine the shape of the output
        out0 = F.conv3d(input=x[0][None], weight=weights[0].reshape(self.weights_shape),
                        bias=biases[0], stride=self.stride, padding=self.padding)

        out = torch.zeros([x.shape[0]] + list(out0.shape[1:]), dtype=x.dtype, layout=x.layout, device=x.device)
        out[0] = out0
        if x.shape[0] > 1:  # if the batch is bigger than aize 1
            for i, (w, b) in enumerate(zip(weights[1:], biases[1:])):
                # each input of the batch has different weights for the feedforward
                w = w.reshape(self.weights_shape)
                out[i + 1] = F.conv3d(input=x[i][None], weight=w, bias=b, stride=self.stride,
                                      padding=self.padding)

        return out


class Conv3DLayer(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 embedding_model=None, embedding_output_size=None,
                 weights_init_method=None, train_loader=None, hyper_input_type=None, GPU=None,
                 var_hypernet_input=None):
        super().__init__()
        self.hyper = not(embedding_model is None)  # if there is an embedding model that hyper is True
        if self.hyper:
            self.layer = HyperConv3dLayer(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding,
                                          embedding_model=embedding_model, embedding_output_size=embedding_output_size,
                                          weights_init_method=weights_init_method, train_loader=train_loader,
                                          hyper_input_type=hyper_input_type, GPU=GPU, var_hypernet_input=var_hypernet_input)

        else:
            self.layer = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if not self.hyper:
            x, features = x[0], x[1]
        return self.layer(x)


class HyperPreactivResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1,
                 hyper_embedding_models=(None, None, None), **hyper_kwargs):
        super().__init__()
        # hyper_kwargs:
        # embedding_model = , embedding_output_size = , weights_init_method = ,
        # train_loader = , hyper_input_type = , GPU =)

        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = Conv3DLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                 embedding_model=hyper_embedding_models[0], **hyper_kwargs)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = Conv3DLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 embedding_model=hyper_embedding_models[1], **hyper_kwargs)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                Conv3DLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                            embedding_model=hyper_embedding_models[2], **hyper_kwargs),
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


class HyperPreactivResBlock_TTT(HyperPreactivResBlock):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, **hyper_kwargs):
        # kwargs:
        # embedding_model = , embedding_output_size = , weights_init_method = ,
        # train_loader = , hyper_input_type = )

        embedding_model = hyper_kwargs["embedding_model"]
        del hyper_kwargs["embedding_model"]

        # if embedding_model is None then the Hyper wont be active
        #                        [hyper1,         hyper2,         hyper Down-sample]
        hyper_embedding_models = [embedding_model, embedding_model, embedding_model]
        super().__init__(in_channels, out_channels, bn_momentum=bn_momentum, dropout=dropout, stride=stride,
                         hyper_embedding_models=hyper_embedding_models, **hyper_kwargs)


class HyperPreactivResBlock_FFT(HyperPreactivResBlock):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, **hyper_kwargs):
        # kwargs:
        # embedding_model = , embedding_output_size = , weights_init_method = ,
        # train_loader = , hyper_input_type = , GPU =)

        embedding_model = hyper_kwargs["embedding_model"]
        del hyper_kwargs["embedding_model"]

        # if embedding_model is None then the Hyper wont be active
        #                        [hyper1,  hyper2,  hyper Down-sample]
        hyper_embedding_models = [None, None, embedding_model]
        super().__init__(in_channels, out_channels, bn_momentum=bn_momentum, dropout=dropout, stride=stride,
                         hyper_embedding_models=hyper_embedding_models, **hyper_kwargs)


class HyperPreactivResBlock_TTF(HyperPreactivResBlock):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, **hyper_kwargs):
        # kwargs:
        # embedding_model = , embedding_output_size = , weights_init_method = ,
        # train_loader = , hyper_input_type = )

        embedding_model = hyper_kwargs["embedding_model"]
        del hyper_kwargs["embedding_model"]

        # if embedding_model is None then the Hyper wont be active
        #                        [hyper1,         hyper2,         hyper Down-sample]
        hyper_embedding_models = [embedding_model, embedding_model, None]
        super().__init__(in_channels, out_channels, bn_momentum=bn_momentum, dropout=dropout, stride=stride,
                         hyper_embedding_models=hyper_embedding_models, **hyper_kwargs)




if __name__ == "__main__":
    embedding_output_size = 3
    embedding = nn.Sequential(nn.Linear(10, embedding_output_size))

    model = LinearLayer(16, 4)
    # model = LinearLayer(16, 4, embedding_model=embedding, embedding_output_size=embedding_output_size,
    #                     weights_init_method=None, train_loader=None, hyper_input_type="tabular")

    x = torch.randn((5, 16))
    tabular = torch.randn((5, 10))
    batch = (x, tabular)

    print(model(batch).shape)

