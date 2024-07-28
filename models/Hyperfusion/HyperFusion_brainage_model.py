from models.base_models import *
from models.Hyperfusion.hyper_base import *
import os
from pl_wrap import PlModelWrapADcls


class HyperFusion_Brainage(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 1

        hyper_embeddings = []
        for i in range(4):
            layer = nn.Linear(in_features=2, out_features=1)
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)
            hyper_embeddings.append(layer)

        fc1_hyper_kwargs = dict(embedding_model=hyper_embeddings[0])
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings[1])
        fc3_hyper_kwargs = dict(embedding_model=hyper_embeddings[2])
        fc4_hyper_kwargs = dict(embedding_model=hyper_embeddings[3])

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
            var_hypernet_input=0.25
        )

        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)
        fc3_hyper_kwargs.update(general_hyper_kwargs)
        fc4_hyper_kwargs.update(general_hyper_kwargs)

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
        self.linear1 = LinearLayer(in_features=39424, out_features=16, **fc1_hyper_kwargs)
        self.linear2 = LinearLayer(in_features=16, out_features=32, **fc2_hyper_kwargs)
        self.linear3 = LinearLayer(in_features=32, out_features=64, **fc3_hyper_kwargs)
        self.final_layer = LinearLayer(in_features=64, out_features=1, **fc4_hyper_kwargs)

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
        out = self.linear1((out, tabular))
        out = F.relu(out)
        out = self.linear2((out, tabular))
        out = F.relu(out)
        out = self.linear3((out, tabular))
        out = F.relu(out)
        out = self.final_layer((out, tabular))

        return out[:, 0]


class HyperFusion_Brainage_nofill(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 1

        hyper_embeddings = []
        for i in range(4):
            layer = nn.Linear(in_features=2, out_features=1)
            # layer.weight.data.fill_(1)
            # layer.bias.data.fill_(0)
            hyper_embeddings.append(layer)

        fc1_hyper_kwargs = dict(embedding_model=hyper_embeddings[0])
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings[1])
        fc3_hyper_kwargs = dict(embedding_model=hyper_embeddings[2])
        fc4_hyper_kwargs = dict(embedding_model=hyper_embeddings[3])

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
            var_hypernet_input=0.25
        )

        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)
        fc3_hyper_kwargs.update(general_hyper_kwargs)
        fc4_hyper_kwargs.update(general_hyper_kwargs)

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
        self.linear1 = LinearLayer(in_features=39424, out_features=16, **fc1_hyper_kwargs)
        self.linear2 = LinearLayer(in_features=16, out_features=32, **fc2_hyper_kwargs)
        self.linear3 = LinearLayer(in_features=32, out_features=64, **fc3_hyper_kwargs)
        self.final_layer = LinearLayer(in_features=64, out_features=1, **fc4_hyper_kwargs)

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
        out = self.linear1((out, tabular))
        out = F.relu(out)
        out = self.linear2((out, tabular))
        out = F.relu(out)
        out = self.linear3((out, tabular))
        out = F.relu(out)
        out = self.final_layer((out, tabular))

        return out[:, 0]


class HyperFusion_Brainage_only_linear2_nofill(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 1

        hyper_embeddings = []
        for i in range(4):
            layer = nn.Linear(in_features=2, out_features=1)
            # layer.weight.data.fill_(1)
            # layer.bias.data.fill_(0)
            hyper_embeddings.append(layer)

        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings[1])
        fc3_hyper_kwargs = dict(embedding_model=None)
        fc4_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
            var_hypernet_input=0.25
        )

        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)
        fc3_hyper_kwargs.update(general_hyper_kwargs)
        fc4_hyper_kwargs.update(general_hyper_kwargs)

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
        self.linear1 = LinearLayer(in_features=39424, out_features=16, **fc1_hyper_kwargs)
        self.linear2 = LinearLayer(in_features=16, out_features=32, **fc2_hyper_kwargs)
        self.linear3 = LinearLayer(in_features=32, out_features=64, **fc3_hyper_kwargs)
        self.final_layer = LinearLayer(in_features=64, out_features=1, **fc4_hyper_kwargs)

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
        out = self.linear1((out, tabular))
        out = F.relu(out)
        out = self.linear2((out, tabular))
        out = F.relu(out)
        out = self.linear3((out, tabular))
        out = F.relu(out)
        out = self.final_layer((out, tabular))

        return out[:, 0]


class HyperFusion_Brainage_only_linear2_2to2(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 2

        hyper_embeddings = []
        for i in range(4):
            layer = nn.Linear(in_features=2, out_features=2)
            hyper_embeddings.append(layer)

        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings[1])
        fc3_hyper_kwargs = dict(embedding_model=None)
        fc4_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
            var_hypernet_input=0.25
        )

        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)
        fc3_hyper_kwargs.update(general_hyper_kwargs)
        fc4_hyper_kwargs.update(general_hyper_kwargs)

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
        self.linear1 = LinearLayer(in_features=39424, out_features=16, **fc1_hyper_kwargs)
        self.linear2 = LinearLayer(in_features=16, out_features=32, **fc2_hyper_kwargs)
        self.linear3 = LinearLayer(in_features=32, out_features=64, **fc3_hyper_kwargs)
        self.final_layer = LinearLayer(in_features=64, out_features=1, **fc4_hyper_kwargs)

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
        out = self.linear1((out, tabular))
        out = F.relu(out)
        out = self.linear2((out, tabular))
        out = F.relu(out)
        out = self.linear3((out, tabular))
        out = F.relu(out)
        out = self.final_layer((out, tabular))

        return out[:, 0]


class HyperFusion_Brainage_2to2(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 2

        hyper_embeddings = []
        for i in range(4):
            layer = nn.Linear(in_features=2, out_features=2)
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)
            hyper_embeddings.append(layer)

        fc1_hyper_kwargs = dict(embedding_model=hyper_embeddings[0])
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings[1])
        fc3_hyper_kwargs = dict(embedding_model=hyper_embeddings[2])
        fc4_hyper_kwargs = dict(embedding_model=hyper_embeddings[3])

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
            var_hypernet_input=0.25
        )

        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)
        fc3_hyper_kwargs.update(general_hyper_kwargs)
        fc4_hyper_kwargs.update(general_hyper_kwargs)

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
        self.linear1 = LinearLayer(in_features=39424, out_features=16, **fc1_hyper_kwargs)
        self.linear2 = LinearLayer(in_features=16, out_features=32, **fc2_hyper_kwargs)
        self.linear3 = LinearLayer(in_features=32, out_features=64, **fc3_hyper_kwargs)
        self.final_layer = LinearLayer(in_features=64, out_features=1, **fc4_hyper_kwargs)

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
        out = self.linear1((out, tabular))
        out = F.relu(out)
        out = self.linear2((out, tabular))
        out = F.relu(out)
        out = self.linear3((out, tabular))
        out = F.relu(out)
        out = self.final_layer((out, tabular))

        return out[:, 0]


class HyperFusion_Brainage_conv3_b(nn.Module):
    def __init__(self, dropout=0.2, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 1

        hyper_embeddings = []
        for i in range(5):
            layer = nn.Linear(in_features=2, out_features=1)
            layer.weight.data.fill_(1)
            layer.bias.data.fill_(0)
            hyper_embeddings.append(layer)

        fc1_hyper_kwargs = dict(embedding_model=hyper_embeddings[0])
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings[1])
        fc3_hyper_kwargs = dict(embedding_model=hyper_embeddings[2])
        fc4_hyper_kwargs = dict(embedding_model=hyper_embeddings[3])
        conv3_b_hyper_kwargs = dict(embedding_model=hyper_embeddings[4])

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
            var_hypernet_input=0.25
        )

        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)
        fc3_hyper_kwargs.update(general_hyper_kwargs)
        fc4_hyper_kwargs.update(general_hyper_kwargs)
        conv3_b_hyper_kwargs.update(general_hyper_kwargs)

        self.conv1_a = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv1_b = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.batchnorm1 = nn.BatchNorm3d(16)

        self.conv2_a = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2_b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(32)

        self.conv3_a = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        # self.conv3_b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv3_b = Conv3DLayer(in_channels=64, out_channels=64, kernel_size=1, stride=1, **conv3_b_hyper_kwargs)
        self.batchnorm3 = nn.BatchNorm3d(64)

        self.dropout1 = nn.Dropout3d(dropout)
        self.linear1 = LinearLayer(in_features=39424, out_features=16, **fc1_hyper_kwargs)
        self.linear2 = LinearLayer(in_features=16, out_features=32, **fc2_hyper_kwargs)
        self.linear3 = LinearLayer(in_features=32, out_features=64, **fc3_hyper_kwargs)
        self.final_layer = LinearLayer(in_features=64, out_features=1, **fc4_hyper_kwargs)

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
        out = self.conv3_b((out, tabular))
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm3(out)

        out = self.dropout1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear1((out, tabular))
        out = F.relu(out)
        out = self.linear2((out, tabular))
        out = F.relu(out)
        out = self.linear3((out, tabular))
        out = F.relu(out)
        out = self.final_layer((out, tabular))

        return out[:, 0]



    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################

class HyperFusionBrainageGeneric(nn.Module):
    def __init__(self, fill, hyper_layers_dict, dropout=0.2, **kwargs):
        super().__init__()
        self.is_hyper = hyper_layers_dict
        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 1

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
            var_hypernet_input=0.25
        )

        def get_hyper_kwargs(is_hyper: bool):
            if is_hyper:
                layer = nn.Linear(in_features=2, out_features=1)
                if fill:
                    layer.weight.data.fill_(1)
                    layer.bias.data.fill_(0)
                hyper_kwargs = dict(embedding_model=layer)
            else:
                hyper_kwargs = dict(embedding_model=None)
            hyper_kwargs.update(general_hyper_kwargs)
            return hyper_kwargs

        if hyper_layers_dict['conv1_a']:
            self.conv1_a = Conv3DLayer(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0, **get_hyper_kwargs(hyper_layers_dict['conv1_a']))
        else:
            self.conv1_a = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        if hyper_layers_dict['conv1_b']:
            self.conv1_b = Conv3DLayer(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, **get_hyper_kwargs(hyper_layers_dict['conv1_b']))
        else:
            self.conv1_b = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.batchnorm1 = nn.BatchNorm3d(16)

        if hyper_layers_dict['conv2_a']:
            self.conv2_a = Conv3DLayer(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, **get_hyper_kwargs(hyper_layers_dict['conv2_a']))
        else:
            self.conv2_a = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)

        if hyper_layers_dict['conv2_b']:
            self.conv2_b = Conv3DLayer(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, **get_hyper_kwargs(hyper_layers_dict['conv2_b']))
        else:
            self.conv2_b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(32)

        if hyper_layers_dict['conv3_a']:
            self.conv3_a = Conv3DLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, **get_hyper_kwargs(hyper_layers_dict['conv3_a']))
        else:
            self.conv3_a = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        if hyper_layers_dict['conv3_b']:
            self.conv3_b = Conv3DLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, **get_hyper_kwargs(hyper_layers_dict['conv3_b']))
        else:
            self.conv3_b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm3d(64)

        self.dropout1 = nn.Dropout3d(dropout)
        if hyper_layers_dict['fc1']:
            self.linear1 = LinearLayer(in_features=39424, out_features=16, **get_hyper_kwargs(hyper_layers_dict['fc1']))
        else:
            self.linear1 = nn.Linear(in_features=39424, out_features=16)

        if hyper_layers_dict['fc2']:
            self.linear2 = LinearLayer(in_features=16, out_features=32, **get_hyper_kwargs(hyper_layers_dict['fc2']))
        else:
            self.linear2 = nn.Linear(in_features=16, out_features=32)

        if hyper_layers_dict['fc3']:
            self.linear3 = LinearLayer(in_features=32, out_features=64, **get_hyper_kwargs(hyper_layers_dict['fc3']))
        else:
            self.linear3 = nn.Linear(in_features=32, out_features=64)

        if hyper_layers_dict['fc4']:
            self.final_layer = LinearLayer(in_features=64, out_features=1, **get_hyper_kwargs(hyper_layers_dict['fc4']))
        else:
            self.final_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        image, tabular = x

        out = self.conv1_a((image, tabular) if self.is_hyper['conv1_a'] else image)
        out = F.relu(out)
        out = self.conv1_b((out, tabular) if self.is_hyper['conv1_b'] else out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm1(out)

        out = self.conv2_a((out, tabular) if self.is_hyper['conv2_a'] else out)
        out = F.relu(out)
        out = self.conv2_b((out, tabular) if self.is_hyper['conv2_b'] else out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm2(out)

        out = self.conv3_a((out, tabular) if self.is_hyper['conv3_a'] else out)
        out = F.relu(out)
        out = self.conv3_b((out, tabular) if self.is_hyper['conv3_b'] else out)
        out = F.relu(out)
        out = F.max_pool3d(out, kernel_size=2, stride=2)
        out = self.batchnorm3(out)

        out = self.dropout1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear1((out, tabular) if self.is_hyper['fc1'] else out)
        out = F.relu(out)
        out = self.linear2((out, tabular) if self.is_hyper['fc2'] else out)
        out = F.relu(out)
        out = self.linear3((out, tabular) if self.is_hyper['fc3'] else out)
        out = F.relu(out)
        out = self.final_layer((out, tabular) if self.is_hyper['fc4'] else out)

        return out[:, 0]


class B_nofill__TTTT(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = False
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': False,

            'fc1': True,
            'fc2': True,
            'fc3': True,
            'fc4': True,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_fill__FT_FFF(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = True
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': True,

            'fc1': False,
            'fc2': False,
            'fc3': False,
            'fc4': False,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_nofill__FT_FFFF(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = False
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': True,

            'fc1': False,
            'fc2': False,
            'fc3': False,
            'fc4': False,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_fill__TFFF(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = True
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': False,

            'fc1': True,
            'fc2': False,
            'fc3': False,
            'fc4': False,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_fill__FTFF(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = True
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': False,

            'fc1': False,
            'fc2': True,
            'fc3': False,
            'fc4': False,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_fill__FFTF(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = True
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': False,

            'fc1': False,
            'fc2': False,
            'fc3': True,
            'fc4': False,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_fill__FFFT(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = True
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': False,

            'fc1': False,
            'fc2': False,
            'fc3': False,
            'fc4': True,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_nofill__FFFT(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = False
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': False,

            'fc1': False,
            'fc2': False,
            'fc3': False,
            'fc4': True,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)


class B_fill__FF_FFFF(HyperFusionBrainageGeneric):
    def __init__(self, **kwargs):
        fill = True
        hyper_layers_dict = {
            'conv1_a': False,
            'conv1_b': False,

            'conv2_a': False,
            'conv2_b': False,

            'conv3_a': False,
            'conv3_b': False,

            'fc1': False,
            'fc2': False,
            'fc3': False,
            'fc4': False,
        }
        super().__init__(fill=fill, hyper_layers_dict=hyper_layers_dict, **kwargs)

