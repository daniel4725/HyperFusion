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

# per model and data imports:
from utils.costum_callbacks import CheckpointCallbackBrainage, CheckpointCallbackAD
from utils.utils import get_class_weight
from pl_wrap import PlModelWrapADcls, PlModelWrapBrainAge
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
    model_name = config.model.pop("model_name")
    model = globals()[model_name](**config.model)

    # wrap the model with its relevant pytorch lightning model
    config.lightning_wrapper.model = model
    lightning_wrapper_name = config.lightning_wrapper.pop("wrapper_name")
    pl_model = globals()[lightning_wrapper_name](**config.lightning_wrapper)

    # Callbacks:
    callbacks = [TimeEstimatorCallback(config.trainer.epochs)]
    if config.checkpointing.enable:
        callbacks += [config.checkpointing.CheckpointCallback(**config.checkpointing.callback_kwargs)]

    if len(config.trainer.gpu) > 1:
        strategy = "dp"
    else:
        strategy = None

    # Create the trainer:
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.trainer.gpu,
        strategy=strategy,
        default_root_dir=config.checkpointing.ckpt_dir,

        logger=logger,
        callbacks=callbacks,

        max_epochs=config.trainer.epochs,
        fast_dev_run=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        overfit_batches=config.trainer.overfit_batches,

        enable_checkpointing=config.checkpointing.enable,
    )

    if not config.checkpointing.continue_train_from_ckpt:
        trainer.fit(pl_model, datamodule=data_module)


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
    default_cfg_path = os.path.join(os.getcwd(), "experiments", "AD_classification", "default_train_config.yml")
    # default_cfg_path = os.path.join(os.getcwd(), "experiments", "brain_age_prediction", "default_train_config.yml")

    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', default=default_cfg_path, type=str, help="path to YAML config file")
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()

    assert os.path.exists(args.config_path), f"config file '{args.config_path}' does not exist!"
    with open(args.config_path, 'r') as file:
        config = EasyDict(yaml.safe_load(file))

    # debug mode using an IDE
    ide_debug_mode = any('pydevd' in s for s in sys.modules)
    if args.debug or ide_debug_mode:
        print("debug mode activated!")

        config_path = "/path/to/config.yml"
        with open(config_path, 'r') as file:
            config = EasyDict(yaml.safe_load(file))

        config.data_module.num_workers = 0
        config.trainer.gpu = [1]
        config.wandb.enable = False
        config.data_module.dataset_cfg.load2ram = False

    main(config)
