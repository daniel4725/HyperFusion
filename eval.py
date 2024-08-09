from train import *
from pl_wrap import *
from models.model_ensemble import ModelsEnsembleClassification, ModelsEnsembleRegression
import re

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
    model = get_ensemble_model(config)

    # wrap the model with its relevant pytorch lightning model
    config.lightning_wrapper.model = model
    test_lightning_wrapper_name = config.lightning_wrapper.pop("wrapper_name") + "4Test"
    pl_model = globals()[test_lightning_wrapper_name](**config.lightning_wrapper)

    # Callbacks:
    callbacks = [TimeEstimatorCallback(config.trainer.epochs)]

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

        enable_checkpointing=False,
    )
    trainer.test(pl_model, datamodule=data_module)

def get_ensemble_model(config):
    wrapper = globals()[config.lightning_wrapper.wrapper_name]
    versions = config.versions.split(",")
    model = None
    if config.task == "AD_classification":
        model = ModelsEnsembleClassification()
        experiment_base_name = re.sub(r"_v\d-", "{}-", config.experiment_name)
        for v in versions:
            experiment_name = experiment_base_name.format(v)
            print(f"loading experiment: {experiment_name}")
            experiment_dir = os.path.join(config.checkpointing.ckpt_dir, experiment_name)
            for fold_directory in os.listdir(experiment_dir):
                model_path = os.path.join(experiment_dir, fold_directory, "best_val.ckpt")
                m = wrapper.load_from_checkpoint(model_path).model
                model.append(m)

    elif config.task == "brain_age_prediction":
        model = ModelsEnsembleRegression()
        experiment_base_name = re.sub(r"_v\d", "{}", config.experiment_name)
        for v in versions:
            experiment_name = experiment_base_name.format(v)
            print(f"loading experiment: {experiment_name}")
            experiment_dir = os.path.join(config.checkpointing.ckpt_dir, experiment_name)
            model_path = os.path.join(experiment_dir, "best_val.ckpt")
            m = wrapper.load_from_checkpoint(model_path).model
            model.append(m)

    return model

def arrange_config4task(config: EasyDict):
    if config.task == "AD_classification":

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

    main(config)

