import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from utils import get_img_conf_matrix
import wandb
from tformNaugment import tform_dict


# pytorch_lightning:
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html   

class PlModelWrap(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = kwargs["model"]
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.L2 = kwargs["L2"]
        self.class_weights = kwargs["class_weights"]
        self.num_classes = len(kwargs["class_weights"])
        self.class_names = kwargs["class_names"]

        self.best_val_balanced_acc = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, tabular, y = batch
        y_hat = self((imgs, tabular))

        loss = F.cross_entropy(y_hat, y, weight=self.class_weights.to(self.device))
        acc = torchmetrics.functional.accuracy(y_hat, y)
        AUC = torchmetrics.functional.auroc(y_hat.softmax(dim=-1), y, num_classes=self.num_classes)       
        balanced_acc = torchmetrics.functional.accuracy(y_hat, y, num_classes=self.num_classes, average='macro')

        self.log('train/loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('train/acc', acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('train/AUC', AUC, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('train/balanced_acc', balanced_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return loss

    def training_epoch_end(self, training_step_outputs):
        # all_preds = torch.stack(training_step_outputs)
        pass

    def validation_step(self, batch, batch_idx):
        imgs, tabular, y = batch

        # the validation is both hippocampuses so we need to split them and flip the right one
        mid_x = imgs.shape[4]//2
        y_hat1 = self((imgs[...,mid_x:], tabular))
        y_hat2 = self((imgs[...,:mid_x].flip(dims=(4,)), tabular))
        y_hat = (y_hat1 + y_hat2)/2
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights.to(self.device))
        acc = torchmetrics.functional.accuracy(y_hat, y)

        self.log('val/loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/acc', acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return y_hat, y

    def validation_epoch_end(self, validation_step_outputs):
        y_hat = validation_step_outputs[0][0]
        y = validation_step_outputs[0][1]
        for element in validation_step_outputs[1:]:
            y_hat = torch.cat((y_hat, element[0]))
            y = torch.cat((y, element[1]))

        balanced_acc = torchmetrics.functional.accuracy(y_hat, y, num_classes=self.num_classes, average='macro')
        f1_macro = torchmetrics.functional.f1_score(y_hat, y, num_classes=self.num_classes, average='macro')
        f1_micro = torchmetrics.functional.f1_score(y_hat, y, num_classes=self.num_classes, average='micro')

        # cm = torchmetrics.functional.confusion_matrix(y_hat, y, num_classes=self.num_classes)
        # mean_val_acc = torch.diag(cm).sum()/len(y)
        CN_acc, MCI_acc, AD_acc = torchmetrics.functional.accuracy(y_hat, y, num_classes=self.num_classes, average='none')
        self.log('val/CN_acc', CN_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/MCI_acc', MCI_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/AD_acc', AD_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/f1_macro', f1_macro, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/f1_micro', f1_micro, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        # wandb.log({"val/conf_mat" : wandb.plot.confusion_matrix(
        #             y_true=np.array(y.cpu()), preds=np.array(y_hat.argmax(1).cpu()),
        #             class_names=self.class_names)})
        if self.best_val_balanced_acc < balanced_acc:
            self.best_val_balanced_acc = balanced_acc
        self.log('val/best_balanced_acc', self.best_val_balanced_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

        AUC = torchmetrics.functional.auroc(y_hat.softmax(dim=-1), y, num_classes=self.num_classes)       

        self.log('val/best_acc', self.best_val_balanced_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/AUC', AUC, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/balanced_acc', balanced_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)



