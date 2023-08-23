from torch import nn
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


class HistogramEmdLoss(nn.Module):
    def __init__(self, min_val, max_val, bins):
        super().__init__()
        self.bins = bins
        self.interval_length = (max_val - min_val) / bins
        self.intervals = torch.linspace(min_val, max_val, bins + 1)
        self.mid_intervals = (self.intervals + self.interval_length/2)[:-1]
        self.bandwidth = self.interval_length / 2.5

    def get_smooth_histogram(self, y):
        hist = torch.zeros((y.size(0), self.bins))  # shape: (batch, bins)
        for i, mid_interval in enumerate(self.mid_intervals):
            sig_plus = torch.sigmoid((y - mid_interval + self.interval_length/2)/self.bandwidth)
            sig_minus = torch.sigmoid((y - mid_interval - self.interval_length/2)/self.bandwidth)
            hist[:, i] = (sig_plus - sig_minus).sum(axis=1)
        hist = hist / y.size(1)
        return hist

    def get_regular_histogram(self, y):
        hist = torch.zeros((y.size(0), self.bins))  # shape: (batch, bins)
        for i, (bin_start, bin_end) in enumerate(zip(self.intervals[:-1], self.intervals[1:])):
            hist[:, i] = ((y >= bin_start) & (y < bin_end)).sum(axis=1)
        hist = hist / y.size(1)
        return hist

    def get_uniform_histogram(self):
        hist = torch.ones((self.bins))
        return hist / self.bins

    def get_normal_histogram(self, mu, sigma):
        hist = torch.exp(-0.5 * ((self.mid_intervals - mu) / sigma) ** 2)
        return (hist / hist.sum())

    def forward(self, y_hat, hist):
        y_hat_smooth_hist = self.get_smooth_histogram(y_hat)
        cdf_y_hat = torch.cumsum(y_hat_smooth_hist, dim=1)
        cdf_hist = torch.cumsum(hist, dim=0)
        loss = ((cdf_y_hat - cdf_hist) ** 2).mean()
        return loss

    def plot_histogram(self, hist):
        hist = hist.detach().numpy()
        plt.bar(self.mid_intervals, hist, width=[self.interval_length] * len(hist))
        plt.show()


class PlWrap4HyperInitHistLoss(pl.LightningModule):
    def __init__(self, model, target_distribution, batch_size, histogram_loss_obj, lr=0.1):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.target_distribution = target_distribution
        self.histogram_loss = histogram_loss_obj
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, tabular, _ = batch
        weights_out, biasses_out = self(tabular)
        w_factor = weights_out.shape[1] / (biasses_out.shape[1] + weights_out.shape[1])
        b_factor = biasses_out.shape[1] / (biasses_out.shape[1] + weights_out.shape[1])

        loss_weights = self.histogram_loss(weights_out, self.target_distribution)
        loss_biases = self.histogram_loss(biasses_out, self.target_distribution)
        loss = (loss_weights * w_factor) + (loss_biases * b_factor)
        # if batch_idx == 0:
        #     self.histogram_loss.plot_histogram(self.target_distribution)
        #     weights_hist = self.histogram_loss.get_regular_histogram(weights_out)
        #     self.histogram_loss.plot_histogram(weights_hist[0])
        #     self.histogram_loss.plot_histogram(weights_hist[1])
        #     self.histogram_loss.plot_histogram(weights_hist[2])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def hypernet_histloss_init(hypernet, fan_in, train_dataloader, GPU, lr=0.0003, epochs=120):
    l2r = train_dataloader.dataset.data_in_ram
    only_tabular = train_dataloader.dataset.only_tabular
    dl = train_dataloader
    train_dataloader.dataset.data_in_ram = False
    train_dataloader.dataset.only_tabular = True
    train_dataloader = DataLoader(dataset=train_dataloader.dataset, batch_size=256, shuffle=True)

    # TODO automize parameters
    # bins = np.ceil(np.sqrt(hypernet.num_weights + hypernet.num_biases)).astype(int)
    bins = 30

    # TODO is that good? this stdv?
    torch_init_val = 1 / np.sqrt(fan_in)
    hist_loss = HistogramEmdLoss(min_val=-torch_init_val, max_val=torch_init_val, bins=bins)
    target_hist = hist_loss.get_uniform_histogram()

    # TODO is that good? this stdv/8? - meant to concentrate all in one place so while
    #  learning less weights will be outside of the distribution and will be squeezed to
    #  places when the side sigmoids are 0
    nn.init.uniform_(hypernet.weights_gen.weight, -torch_init_val / 100, torch_init_val / 100)
    nn.init.uniform_(hypernet.weights_gen.bias, -torch_init_val / 100, torch_init_val / 100)
    nn.init.uniform_(hypernet.bias_gen.weight, -torch_init_val / 100, torch_init_val / 100)
    nn.init.uniform_(hypernet.bias_gen.bias, -torch_init_val / 100, torch_init_val / 100)

    hypernet.freeze_embedding_model()
    pl_hyper = PlWrap4HyperInitHistLoss(model=hypernet, target_distribution=target_hist,
                                        batch_size=train_dataloader.batch_size, histogram_loss_obj=hist_loss, lr=lr)
    trainer = pl.Trainer(accelerator="gpu", devices=GPU, max_epochs=epochs, logger=False, checkpoint_callback=False)
    trainer.fit(pl_hyper, train_dataloader=train_dataloader)
    hypernet.unfreeze_embedding_model()

    dl.dataset.data_in_ram = l2r
    dl.dataset.only_tabular = only_tabular


# # target_hist = hist_loss.get_normal_histogram(mu=0, sigma=1)
#
#
# # hist_loss(y, target_hist)
# batch_size = 256
# loaders = get_dataloaders(batch_size=batch_size, features_set=5, adni_dir="/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI",
#                           fold=0, num_workers=0,
#                           transform_train=tform_dict["None"],
#                           transform_valid=tform_dict["None"], load2ram=False,
#                           num_classes=3, only_tabular=True,
#                           with_skull=True, no_bias_field_correct=False)
# train_loader, _ = loaders
#
#
# # --------------- the histogram loss --------------------
# main_net_in_features = 64
# main_net_out_features = 16
# stdv = 1 / np.sqrt(main_net_in_features)
# hist_loss = HistogramEmdLoss(min_val=-stdv, max_val=stdv, bins=40)
# target_hist = hist_loss.get_uniform_histogram()
#
# plt.figure()
# plt.title("target_hist")
# hist_loss.plot_histogram(target_hist)
#
#
# # --------------- the hyper network --------------------
# out_embbeding_size = 12
# embedding_model = nn.Sequential(
#     nn.Linear(train_loader.dataset.num_tabular_features, 10),
#     nn.Linear(10, 6),
#     nn.Linear(6, out_embbeding_size)
# )
# hyper_net = HyperNetwork(embedding_model=embedding_model, embedding_output_size=out_embbeding_size,
#                      num_weights=main_net_out_features * main_net_out_features, num_biases=main_net_out_features)
# nn.init.uniform_(hyper_net.weights_gen.weight, -stdv/8, stdv/8)
# hyper_net.freeze_embedding_model()
#
#
# # show histogram of the weights of the weight generator layer
# plt.hist(np.array(hyper_net.weights_gen.weight.view(-1).detach().cpu()))
# plt.title("weights of weights generator layer before")
# plt.show()
#
# # ---------- train the net to init the weights -----------
# pl_hyper = PlWrap4HyperInitHistLoss(model=hyper_net, target_distribution=target_hist,
#                                     batch_size=batch_size, histogram_loss_obj=hist_loss, lr=0.005)
# trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=100)
# trainer.fit(pl_hyper, train_dataloader=train_loader)
# hyper_net.unfreeze_embedding_model()
# # --------------------------------------------------------
#
# # show histogram of the weights of the weight generator layer
# plt.hist(np.array(hyper_net.weights_gen.weight.view(-1).detach().cpu()))
# plt.title("weights of weights generator layer after")
# plt.show()
#
# out_weights = []
# for _, tabular, _ in iter(train_loader):
#     weights, biasses = hyper_net(tabular)
#     for w in weights:
#         out_weights.append(w)
#
# for w in np.random.choice(out_weights, 10, replace=False):
#     plt.hist(w.detach().numpy(), bins=30)
#     plt.title("weight generated by a specific input")
#     plt.show()
#