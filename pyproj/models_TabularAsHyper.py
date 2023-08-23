from models import *
from HyperModels.hyper_base import *
import os


class TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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

class TabularAsHyper_R_R_R_FTF_FF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, hyper_embeddings_tab, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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

class TabularAsHyper_R_R_R_TFF_FF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[hyper_embeddings_tab, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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

class TabularAsHyper_R_R_R_FFF_TF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, None])
        fc1_hyper_kwargs = dict(embedding_model=hyper_embeddings_tab)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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

class TabularAsHyper_R_R_R_FFF_FT_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings_tab)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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

class TabularAsHyper_R_R_FFT_FFF_FF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        resblock2_hyper_kwargs = dict(hyper_embedding_models=[None, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        resblock2_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block3 = HyperPreactivResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock2_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3((out, tabular))
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out

class TabularAsHyper_R_R_FTF_FFF_FF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, hyper_embeddings_tab, None])
        resblock2_hyper_kwargs = dict(hyper_embedding_models=[None, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        resblock2_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block3 = HyperPreactivResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock2_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3((out, tabular))
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out

class TabularAsHyper_R_R_TFF_FFF_FF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[hyper_embeddings_tab, None, None])
        resblock2_hyper_kwargs = dict(hyper_embedding_models=[None, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        resblock2_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block3 = HyperPreactivResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock2_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3((out, tabular))
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out

class TabularAsHyper_R_R_FFT_FFT_FF_embd_trainedTabular8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )
        model2 = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab2 = nn.Sequential(
            model2[0],
            model2[1],
            model2[2]
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        resblock2_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab2])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        resblock2_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block3 = HyperPreactivResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock2_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3((out, tabular))
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out



# ----------------------------------------------------------------------------------------------
class TabularAsHyper_R_R_R_FFT_FF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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


class TabularAsHyper_R_R_R_FFT_FF_embd8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 8
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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


class TabularAsHyper_R_R_R_FFT_FF_embd8_2losses(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 8
        self.hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        self.tabular_head = nn.Linear(in_features=embd_tab_out_size, out_features=n_outputs)

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, self.hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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

        tabular_out = self.hyper_embeddings_tab(tabular)
        tabular_out = self.tabular_head(tabular_out)

        return out, tabular_out


class TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 3
        embd_model_path_end = f'baseline-tabular_embd8_v2-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        hyper_embeddings_tab = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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



class TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular8Freeze(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_embd8_v2-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp
        hyper_embeddings_tab = nn.Sequential(
            model[0],
            model[1],
            model[2]
        )
        for param in hyper_embeddings_tab.parameters():
            param.requires_grad = False

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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



class TabularAsHyper_R_R_R_FFT_FF_TabularLoss(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()
        from pl_wrap import PlModelWrap

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 3
        embd_model_path_end = f'baseline-tabular_embd8_v2-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        self.hyper_embeddings_tab = PlModelWrap.load_from_checkpoint(embd_model_path).model.mlp

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, self.hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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

        tabular_out = self.hyper_embeddings_tab(tabular)

        return 0.7 * out + 0.3 * tabular_out



class TabularAsHyper_R_R_R_FFT_FF_embd4(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 4
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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


class TabularAsHyper_R_R_FFT_R_FF_embd8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1,
                 **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 8
        hyper_embeddings_tab = nn.Sequential(
            nn.Linear(n_tabular_features, embd_tab_out_size),
            nn.BatchNorm1d(embd_tab_out_size),
            nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block3 = HyperPreactivResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block4 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3((out, tabular))
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class TabularAsHyper_R_R_FFT_FFT_FF_embd8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1,
                 **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 8
        hyper_embeddings_tab = nn.Sequential(
            nn.Linear(n_tabular_features, embd_tab_out_size),
            nn.BatchNorm1d(embd_tab_out_size),
            nn.PReLU()
        )
        hyper_embeddings_tab2 = nn.Sequential(
            nn.Linear(n_tabular_features, embd_tab_out_size),
            nn.BatchNorm1d(embd_tab_out_size),
            nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        resblock2_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab2])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        resblock2_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block3 = HyperPreactivResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock2_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3((out, tabular))
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class TabularAsHyper_R_R_FFT_FFT_FF_embd16(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1,
                 **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
            nn.Linear(n_tabular_features, embd_tab_out_size),
            nn.BatchNorm1d(embd_tab_out_size),
            nn.PReLU()
        )
        hyper_embeddings_tab2 = nn.Sequential(
            nn.Linear(n_tabular_features, embd_tab_out_size),
            nn.BatchNorm1d(embd_tab_out_size),
            nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        resblock2_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab2])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        resblock2_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block3 = HyperPreactivResBlock(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock2_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3((out, tabular))
        out = self.block4((out, tabular))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class TabularAsHyper_R_FFT_R_R_FF_embd8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1,
                 **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 8
        hyper_embeddings_tab = nn.Sequential(
            nn.Linear(n_tabular_features, embd_tab_out_size),
            nn.BatchNorm1d(embd_tab_out_size),
            nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = HyperPreactivResBlock(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2, **resblock_hyper_kwargs)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2,
                                          dropout=0.2)
        self.block4 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16 * init_features, 4 * init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4 * init_features, n_outputs, **fc2_hyper_kwargs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x

        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2((out, tabular))
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



class TabularAsHyper_R_R_R_FFT_FF_histinit(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "histogram"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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


class TabularAsHyper_R_R_R_FFT_TT(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )
        hyper_embeddings_tab2 = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )
        hyper_embeddings_tab3 = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=hyper_embeddings_tab2)
        fc2_hyper_kwargs = dict(embedding_model=hyper_embeddings_tab3)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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


class TabularAsHyper_R_R_R_TFT_FF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )
        hyper_embeddings_tab2 = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[hyper_embeddings_tab2, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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


class TabularAsHyper_R_R_R_TFF_FF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[hyper_embeddings_tab, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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



class TabularAsHyperMIX16_R_R_R_TFF_FF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features + (8 * init_features), embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[hyper_embeddings_tab, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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
        squeeze = self.global_pool(out)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, tabular), dim=1)

        out = self.block4((out, squeeze))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out

class TabularAsHyperMIX_R_R_R_TFF_FF(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 8
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features + (8 * init_features), embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[hyper_embeddings_tab, None, None])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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
        squeeze = self.global_pool(out)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, tabular), dim=1)

        out = self.block4((out, squeeze))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out


class TabularAsHyperMIX_R_R_R_FFT_FF_embd8(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None
        embd_tab_out_size = 8
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features + (8 * init_features), embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        #                                               [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)


        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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
        squeeze = self.global_pool(out)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, tabular), dim=1)

        out = self.block4((out, squeeze))
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1((out, tabular))
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2((out, tabular))

        return out



class Hyper_testing(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 16
        hyper_embeddings_tab = nn.Sequential(
                nn.Linear(n_tabular_features, embd_tab_out_size),
                nn.BatchNorm1d(embd_tab_out_size),
                nn.PReLU()
        )

        # [conv1 hyper, conv2 hyper, down-sample hyper]
        resblock_hyper_kwargs = dict(hyper_embedding_models=[None, None, hyper_embeddings_tab])
        fc1_hyper_kwargs = dict(embedding_model=None)
        fc2_hyper_kwargs = dict(embedding_model=None)

        general_hyper_kwargs = dict(
            embedding_output_size=embd_tab_out_size,
            weights_init_method=weights_init_method,
            hyper_input_type="tabular",
            train_loader=kwargs["train_loader"],
            GPU=kwargs["GPU"],
        )
        resblock_hyper_kwargs.update(general_hyper_kwargs)
        fc1_hyper_kwargs.update(general_hyper_kwargs)
        fc2_hyper_kwargs.update(general_hyper_kwargs)

        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 * init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = HyperPreactivResBlock(8 * init_features, 16 * init_features, bn_momentum=bn_momentum,
                                            stride=2, dropout=0.3, **resblock_hyper_kwargs)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = LinearLayer(16*init_features, 4*init_features, **fc1_hyper_kwargs)

        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = LinearLayer(4*init_features, n_outputs, **fc2_hyper_kwargs)

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
