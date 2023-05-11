import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


def MLP(mlp_layers_shapes=[10, 64, 64, 3], activation=nn.ReLU,
        dropout=0, bn_momentum=0.1, **kwargs):
    """
    layers_shapes = [input shape, hidden1, hidden2 ..., output shape]
    """
    assert len(mlp_layers_shapes) >=2  # must have at least input shape and output shape

    output_shape = mlp_layers_shapes.pop()  # pop the last element (the output shape)
    mlp = nn.Sequential()
    for i, (in_size, out_size) in enumerate(zip(mlp_layers_shapes[:-1], mlp_layers_shapes[1:])):
        mlp.add_module(f"bn_{i}", nn.BatchNorm1d(in_size, momentum=bn_momentum))
        mlp.add_module(f"fc_{i}", nn.Linear(in_features=in_size, out_features=out_size)) 
        mlp.add_module(f"activation_{i}", activation())
        mlp.add_module(f"dropout_{i}",nn.Dropout(dropout))
    mlp.add_module("output", nn.Linear(in_features=mlp_layers_shapes[-1], out_features=output_shape))
    return mlp


class MLP4Tabular(nn.Module):
    def __init__(self, mlp_layers_shapes=[10, 64, 64, 3], activation=nn.PReLU,
                dropout=0, bn_momentum=0.1, **kwargs):
        super().__init__()
        self.mlp = MLP(mlp_layers_shapes=mlp_layers_shapes, activation=activation,
                        dropout=dropout, bn_momentum=bn_momentum)

        print(self)

    def forward(self, x):
        image, tabular = x        
        return self.mlp(tabular)

if __name__ == "__main__":
    batch_size = 12
    in_features = 4
    out_features = 3

    # model = MLP4Tabular(
    #      in_features=10,
    #      out_features=3,
    #      hidden_shapes=[16, 32, 64, 16],
    #     activation=nn.PReLU,
    #     skip_connection=False,
    #     dropout=0.3,
    #     bn_momentum=0.1
    # )
    # data = ("", torch.randn(13, in_features))
    # print(model(data))

    model = MLP4Tabular(
        mlp_layers_shapes=[in_features, out_features],
        activation=nn.PReLU,
        dropout=0.3,
        bn_momentum=0.1
    )
    # print(model)
    # data = torch.randn(batch_size, in_features)

    # print(model(data))

