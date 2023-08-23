import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from utils import nonsquared_conf_mat
import wandb
from tformNaugment import tform_dict
import matplotlib.pyplot as plt


# pytorch_lightning:
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html   

class PlModelWrap4Regression(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = kwargs["model"]
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.L2 = kwargs["L2"]

        self.best_val_MAE = 1000

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # self.model.train()
        imgs, tabular, y = batch
        y_hat = self((imgs, tabular))

        loss = F.mse_loss(y_hat.to(torch.float64).view(-1), y)
        mae = torchmetrics.functional.mean_absolute_error(y_hat.view(-1), y) * 100

        self.log('train/loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('train/MAE', mae, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss

    def training_epoch_end(self, training_step_outputs):
        # all_preds = torch.stack(training_step_outputs)
        pass

    def _shared_eval_step(self, batch, batch_idx, evaluation_type="val"):
        # self.model.eval()

        imgs, tabular, y = batch

        # the validation is both hippocampuses so we need to split them and flip the right one
        y_hat = self((imgs, tabular))

        if evaluation_type == "val":
            loss = F.mse_loss(y_hat, y)

            self.log(f'{evaluation_type}/loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return y_hat, y

    def _shared_eval_epoch_end(self, step_outputs, type_evaluiation="val"):
        y_hat = step_outputs[0][0]
        y = step_outputs[0][1]
        for element in step_outputs[1:]:
            y_hat = torch.cat((y_hat, element[0]))
            y = torch.cat((y, element[1]))

        mae = torchmetrics.functional.mean_absolute_error(y_hat.view(-1), y) * 100
        # logging the best balance acc and the other metrics at this point
        if self.best_val_MAE > mae:
            self.best_val_MAE = mae

        self.log(f'{type_evaluiation}/best_MAE', self.best_val_MAE, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/MAE', mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, evaluation_type="val")

    def validation_epoch_end(self, validation_step_outputs):
        self._shared_eval_epoch_end(step_outputs=validation_step_outputs, type_evaluiation="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, evaluation_type="test")

    def test_epoch_end(self, test_step_outputs):
        self._shared_eval_epoch_end(step_outputs=test_step_outputs, type_evaluiation="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)


class PlModelWrap4testRegression(PlModelWrap4Regression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_dataloader(self):
        return [1]


def get_results(list_of_softmax_outputs):
    # find each model prediction (good for majority vote)
    torch.stack(list_of_softmax_outputs).argmax(dim=2)

    # mean over each sample prediction
    torch.stack(list_of_softmax_outputs).mean(dim=0)

