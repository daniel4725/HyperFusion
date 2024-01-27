from models.base_models import *
from models.Hyperfusion.hyper_base import *
import os
from pl_wrap import PlModelWrapADcls


class HyperFusion(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, n_tabular_features=1, **kwargs):
        super().__init__()

        weights_init_method = "input_variance"  # input_variance  embedding_variance  histogram  None

        embd_tab_out_size = 8
        embd_model_path_end = f'baseline-tabular_v1-seed{kwargs["split_seed"]}-fs{kwargs["features_set"]}/fold_{kwargs["data_fold"]}/best_val.ckpt'
        embd_model_path = os.path.join(kwargs["checkpoint_dir"], embd_model_path_end)
        model = PlModelWrapADcls.load_from_checkpoint(embd_model_path).model.mlp
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

