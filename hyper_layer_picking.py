from utils.costum_callbacks import TimeEstimatorCallback
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
import os
import yaml
from easydict import EasyDict
import sys
from tqdm import tqdm
import pickle

# per model and data imports:
from utils.costum_callbacks import CheckpointCallbackBrainage, CheckpointCallbackAD
from utils.utils import get_class_weight
from pl_wrap import PlModelWrapADcls, PlModelWrapBrainAge, PlModelWrapADcls2Classes
from data_utils.ADNI_data_handler import ADNIDataModule
from data_utils.BrainAge_data_handler import BrainAgeDataModule
from models.Hyperfusion.HyperFusion_AD_model import *
from models.Hyperfusion.HyperFusion_brainage_model import *
from models.Film_DAFT_preactive.models_film_daft import *
from models.base_models import *
from models.concat_models import *


def main(config: EasyDict):
    # wandb logger
    logger = wandb_interface(config)

    # Create the data module:
    data_module_name = config.data_module.pop("data_module_name")
    data_module = globals()[data_module_name](config.data_module)
    config.data_module_instance = data_module

    # add and change some configurations w.r.t the task (config.task)
    arrange_config4task(config)

    # build the model
    device = torch.device(f"cuda:{config.trainer.gpu[0]}" if torch.cuda.is_available() else "cpu")
    model = PreactivResNet().to(device)

    model.eval()
    num_times = 400   # time calculations: 70 min per 100 (over all 14 layers)
    layers_names_dict = {
        'fc2': model.fc2,
        'fc1': model.fc1,
        'b4Do': model.block4.downsample[1],
        'b4c2': model.block4.conv2,
        'b4c1': model.block4.conv1,
        'b3Do': model.block3.downsample[1],
        'b3c2': model.block3.conv2,
        'b3c1': model.block3.conv1,
        'b2Do': model.block2.downsample[1],
        'b2c2': model.block2.conv2,
        'b2c1': model.block2.conv1,
        'b1Do': model.block1.downsample[1],
        'b1c2': model.block1.conv2,
        'b1c1': model.block1.conv1,
    }
    with torch.no_grad():
        for name, layer in layers_names_dict.items():
            print(f'\nworking on layer: {name}')
            losses = []
            for i in tqdm(range(num_times)):
                layer.reset_parameters()
                y_hat_accumulator, y_accumulator = [], []
                train_loader = data_module.train_dataloader()
                for batch in train_loader:
                    imgs, tabular, y = batch[0].to(device), batch[1], batch[2].to(device)
                    y_hat = model((imgs, tabular))
                    y_accumulator.append(y)
                    y_hat_accumulator.append(y_hat)
                loss = F.cross_entropy(torch.cat(y_hat_accumulator), torch.cat(y_accumulator))
                losses.append(loss.item())
                wandb.log({f'l-{name}-{i+1}': loss})
            with open(f'/media/rrtammyfs/Users/daniel/hyper_selection2/{name}.pkl', 'wb') as file:
                pickle.dump(losses, file)


import pickle
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

def calc_entropy(array, bins, bin_range, name):
    histogram, bin_edges = np.histogram(array, bins=bins, range=bin_range)
    histogram = histogram / len(array)
    plt.hist(array, bins=bins, density=True, range=bin_range)
    plt.title(name)
    plt.show()
    ent = entropy(histogram, base=2)
    return ent

def load_list(path):
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

pkl_dir1 = '/media/rrtammyfs/Users/daniel/hyper_selection'
pkl_dir2 = '/media/rrtammyfs/Users/daniel/hyper_selection2'
def print_all_entropies(pkl_dir1):
    all_losses, all_names = [], []
    for name in sorted(os.listdir(pkl_dir1)):
        path1 = os.path.join(pkl_dir1, name)
        # path2 = os.path.join(pkl_dir2, name)
        # all_losses.append(load_list(path1) + load_list(path2))
        all_losses.append(load_list(path1))
        all_names.append(name)
    max_loss = np.max(all_losses)
    min_loss = np.min(all_losses)
    print(f'min: {min_loss},  max: {max_loss}')
    for losses, name in zip(all_losses, all_names):
        # print(f'{name}: entropy - {calc_entropy(losses, bins=30, bin_range=(min_loss, max_loss), name=name)}')
        print(f'{calc_entropy(losses, bins=30, bin_range=(min_loss, max_loss), name=name)}')


def arrange_config4task(config: EasyDict):
    if config.task == "AD_classification":

        # for the model
        train_dataset = config.data_module_instance.train_ds
        config.model.n_tabular_features = train_dataset.num_tabular_features
        config.model.n_outputs = train_dataset.num_classes
        config.model.train_loader = config.data_module_instance.train_dataloader()
        config.model.mlp_layers_shapes = [config.model.n_tabular_features] + config.model.hidden_shapes + [train_dataset.num_classes]

        config.model.split_seed = config.data_module.dataset_cfg.split_seed
        config.model.features_set = config.data_module.dataset_cfg.features_set
        config.model.data_fold = config.data_module.dataset_cfg.fold
        config.model.checkpoint_dir = config.checkpointing.ckpt_dir

        config.model.GPU = config.trainer.gpu

        # for the Pl wrapper
        if config.lightning_wrapper.loss.class_weights == 'default':
            config.lightning_wrapper.loss.class_weights = get_class_weight(config.data_module_instance.train_dataloader(),
                                                                           config.data_module_instance.val_dataloader())
        else:
            config.lightning_wrapper.loss.class_weights = torch.Tensor(config.lightning_wrapper.loss.class_weights)

        print("class weights: ", config.lightning_wrapper.loss.class_weights)

        config.lightning_wrapper.batch_size = config.data_module.batch_size
        config.lightning_wrapper.class_names = config.data_module.class_names

        # for checkpointing
        config.checkpointing.CheckpointCallback = CheckpointCallbackAD
        config.checkpointing.callback_kwargs = dict(
            ckpt_dir=config.checkpointing.ckpt_dir,
            experiment_name=config.experiment_name,
            data_fold=config.data_module.dataset_cfg.fold
        )

    elif config.task == "brain_age_prediction":
        config.model.train_loader = config.data_module_instance.train_dataloader()
        config.model.GPU = config.trainer.gpu

        config.lightning_wrapper.batch_size = config.data_module.batch_size

        # for checkpointing

        config.checkpointing.CheckpointCallback = CheckpointCallbackBrainage
        config.checkpointing.callback_kwargs = dict(
            ckpt_dir=config.checkpointing.ckpt_dir,
            experiment_name=config.experiment_name,
        )



def wandb_interface(config: EasyDict):
    wandb_args = config.wandb
    if wandb_args.sweep or wandb_args.enable:
        if wandb_args.sweep:  # gets the args from wandb
            wandb.init()

            # taking the relevant args from the sweep's parameters
            for key in wandb.config.keys():
                wandb_args[key] = wandb.config[key]

            # the agent's CUDA_VISIBLE_DEVICES is alredy set:
            config.trainer.gpu = [0]

        # run_id = datetime.datetime.now().strftime("%d-%m-%Y %Hh%Mm%Ss - ") + args.experiment_name
        os.makedirs(wandb_args.logs_dir, exist_ok=True)
        if config.task == "AD_classification":
            exp_name = config.experiment_name + f"-f{config.data_module.dataset_cfg.fold}"
        else:
            exp_name = config.experiment_name
        logger = WandbLogger(project=wandb_args.project_name,
                             name=exp_name,
                             save_dir=wandb_args.logs_dir)

        def flatten_dict(d, parent_key='', sep='_'):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items

        logger.experiment.config.update(flatten_dict(config))

    else:
        logger = False
    return logger


if __name__ == '__main__':
    default_cfg_path = '/home/duenias/PycharmProjects/HyperFusion/configAD_layer_pick.yml'
    # default_cfg_path = os.path.join(os.getcwd(), "experiments", "brain_age_prediction", "default_train_config.yml")

    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', default=default_cfg_path, type=str, help="path to YAML config file")
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()

    assert os.path.exists(args.config_path), f"config file '{args.config_path}' does not exist!"
    with open(args.config_path, 'r') as file:
        config = EasyDict(yaml.safe_load(file))


    ide_debug_mode = any('pydevd' in s for s in sys.modules)
    if args.debug or ide_debug_mode:
        print("debug mode activated!")


        config.data_module.num_workers = 0

        config.trainer.gpu = [1]


    main(config)
