import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from utils.utils import nonsquared_conf_mat
from easydict import EasyDict

# pytorch_lightning:
# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

class PlModelWrapADcls(pl.LightningModule):
    def __init__(self, **wrapper_kwargs):
        super().__init__()
        self.save_hyperparameters()
        wrapper_kwargs = EasyDict(self.hparams)
        self.model = wrapper_kwargs.model
        self.batch_size = wrapper_kwargs.batch_size
        self.lr = wrapper_kwargs.optimizer.lr
        self.weight_decay = wrapper_kwargs.optimizer.weight_decay
        self.class_weights = wrapper_kwargs.loss.class_weights
        self.num_classes = len(wrapper_kwargs.loss.class_weights)
        self.class_names = wrapper_kwargs.class_names

        self.best_val_balanced_acc = -1

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # self.model.train()
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

    def _shared_eval_step(self, batch, batch_idx, evaluation_type="val"):
        # self.model.eval()

        imgs, tabular, y = batch

        # the validation is both hippocampuses so we need to split them and flip the right one
        mid_x = imgs.shape[4]//2
        y_hat1 = self((imgs[..., mid_x:], tabular))
        y_hat2 = self((imgs[..., :mid_x].flip(dims=(4,)), tabular))
        if evaluation_type == 'val':
            y_hat = (y_hat1.softmax(dim=1) + y_hat2.softmax(dim=1))/2
        elif evaluation_type == 'test':
            y_hat = (y_hat1 + y_hat2) / 2
            # y_hat = get_results(y_hat1 + y_hat2)
        # y_hat = self((imgs, tabular))
        if evaluation_type == "val":
            loss = F.cross_entropy(y_hat, y, weight=self.class_weights.to(self.device))
            acc = torchmetrics.functional.accuracy(y_hat, y)

            self.log(f'{evaluation_type}/loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
            self.log(f'{evaluation_type}/acc', acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

        return y_hat, y

    def log_nonsquared_conf_mat(self, y_hat, y):
        self.confmat_fig_normalized, _, _ = nonsquared_conf_mat(y_hat, y, labels=self.class_names, normalize='true')
        self.confmat_fig, _, _ = nonsquared_conf_mat(y_hat, y, labels=self.class_names)
        self.logger.log_image(f'test_confmat/norm', [self.confmat_fig_normalized])
        self.logger.log_image(f'test_confmat/not_norm', [self.confmat_fig])

        if len(y.unique()) < len(y_hat[0]):
            self.confmat_fig_normalized, _, _ = nonsquared_conf_mat(y_hat, y, labels=self.class_names, normalize='true', classes_3=True)
            self.confmat_fig, _, _ = nonsquared_conf_mat(y_hat, y, labels=self.class_names, classes_3=True)
            self.logger.log_image(f'test_confmat/norm-3_classes_collapse', [self.confmat_fig_normalized])
            self.logger.log_image(f'test_confmat/not_norm-3_classes_collapse', [self.confmat_fig])


    def _shared_eval_epoch_end(self, step_outputs, type_evaluiation="val"):
        y_hat = step_outputs[0][0]
        y = step_outputs[0][1]
        for element in step_outputs[1:]:
            y_hat = torch.cat((y_hat, element[0]))
            y = torch.cat((y, element[1]))

        if len(y.unique()) != len(y_hat[0]):
            self.log_nonsquared_conf_mat(y_hat, y)
            return

        AUC = torchmetrics.functional.auroc(y_hat, y, num_classes=self.num_classes, average='macro')

        balanced_acc = torchmetrics.functional.accuracy(y_hat, y, num_classes=self.num_classes, average='macro')
        f1_macro = torchmetrics.functional.f1_score(y_hat, y, num_classes=self.num_classes, average='macro')
        f1_micro = torchmetrics.functional.f1_score(y_hat, y, num_classes=self.num_classes, average='micro')

        # mean_val_acc = torch.diag(cm).sum()/len(y)
        CN_acc, MCI_acc, AD_acc = torchmetrics.functional.accuracy(y_hat, y, num_classes=self.num_classes, average='none')
        self.log(f'{type_evaluiation}/CN_acc', CN_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/MCI_acc', MCI_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/AD_acc', AD_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/f1_macro', f1_macro, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/f1_micro', f1_micro, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)

        self.log(f'{type_evaluiation}/AUC', AUC, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/balanced_acc', balanced_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)


        # logging the best balance acc and the other metrics at this point
        if self.best_val_balanced_acc < balanced_acc:
            self.best_val_balanced_acc = balanced_acc
            self.CN_acc_at_best_point = CN_acc
            self.MCI_acc_at_best_point = MCI_acc
            self.AD_acc_at_best_point = AD_acc
            self.f1_macro_at_best_point = f1_macro
            self.f1_micro_acc_at_best_point = f1_micro
            self.AUC_at_best_point = AUC
            self.confmat_fig_normalized, _, _ = nonsquared_conf_mat(y_hat, y, labels=self.class_names, normalize='true')
            self.confmat_fig, self.confmat, self.precision_at_best_point = nonsquared_conf_mat(y_hat, y, labels=self.class_names)

        self.log(f'{type_evaluiation}/best_balanced_acc', self.best_val_balanced_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.batch_size)

        self.log(f'{type_evaluiation}/CN_acc_at_best_point', self.CN_acc_at_best_point, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/MCI_acc_at_best_point', self.MCI_acc_at_best_point, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/AD_acc_at_best_point', self.AD_acc_at_best_point, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/f1_macro_at_best_point', self.f1_macro_at_best_point, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/f1_micro_at_best_point', self.f1_micro_acc_at_best_point, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/AUC_at_best_point', self.AUC_at_best_point, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log(f'{type_evaluiation}/precision_at_best_point', self.precision_at_best_point, prog_bar=False, on_step=False, on_epoch=True, batch_size=self.batch_size)


        self.logger.log_table(f'{type_evaluiation}_confmat/raw_confmat', data=self.confmat.tolist())
        self.logger.log_image(f'{type_evaluiation}_confmat/norm', [self.confmat_fig_normalized])
        self.logger.log_image(f'{type_evaluiation}_confmat/not_norm', [self.confmat_fig])


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, evaluation_type="val")

    def validation_epoch_end(self, validation_step_outputs):
        self._shared_eval_epoch_end(step_outputs=validation_step_outputs, type_evaluiation="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, evaluation_type="test")

    def test_epoch_end(self, test_step_outputs):
        self._shared_eval_epoch_end(step_outputs=test_step_outputs, type_evaluiation="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class PlModelWrap4test(PlModelWrapADcls):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_dataloader(self):
        return [1]


def get_results(list_of_softmax_outputs):
    # find each model prediction (good for majority vote)
    torch.stack(list_of_softmax_outputs).argmax(dim=2)

    # mean over each sample prediction
    torch.stack(list_of_softmax_outputs).mean(dim=0)

