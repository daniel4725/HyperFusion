from data_handler import *
from models import *
from pl_wrap import *
from utils import *
from MLP_models import *
from models_concat import *
from Film_DAFT_preactive.models_film_daft import *
from tformNaugment import tform_dict
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import WandbLogger
import wandb
from costum_callbacks import *
from models_TabularAsHyper import *


def main(args):
    # torch.manual_seed(0)
    logger, args = wandb_interface(args)  # this line must be first
    model_name = args.model

    # Create the data loaders:
    loaders = get_dataloaders(batch_size=args.batch_size, features_set=args.features_set, adni_dir=args.adni_dir,
                              fold=args.data_fold, num_workers=args.num_workers,
                              transform_train=tform_dict[args.transform],
                              transform_valid=tform_dict[args.transform_valid], load2ram=args.load2ram,
                              only_tabular=args.only_tabular, num_classes=args.num_classes,
                              with_skull=args.with_skull, no_bias_field_correct=args.no_bias_field_correct,
                              dataset_class=globals()[args.dataset_class], split_seed=args.split_seed)
    train_loader, valid_loader = loaders

    # Create the model:
    if args.class_weights == [0]:
        args.class_weights = get_class_weight(train_loader, valid_loader)
    else:
        args.class_weights = torch.Tensor(args.class_weights)

    # args.class_weights = args.class_weights * 0 + 1
    print("class weights: ", args.class_weights)

    kwargs = args.__dict__
    kwargs["n_tabular_features"] = train_loader.dataset.num_tabular_features
    kwargs['n_outputs'] = len(args.class_weights)
    kwargs["cnn_mlp_shapes"][-1] = len(args.class_weights)
    kwargs["train_loader"] = train_loader

    # mlp_layers_shapes = [n_tabular_features, hidden_shapes, num_classes]
    kwargs["mlp_layers_shapes"] = [kwargs["n_tabular_features"]] + kwargs["hidden_shapes"] + [
        len(kwargs["class_weights"])]

    model = globals()[args.model](**kwargs)
    kwargs["model"] = model
    kwargs["class_names"] = list(train_loader.dataset.labels_dict.keys())

    pl_model = PlModelWrap(**kwargs)

    # Callbacks:
    callbacks = [TimeEstimatorCallback(**kwargs)]
    if kwargs["enable_checkpointing"]:
        callbacks += [CheckpointCallback(**kwargs)]


    if len(args.GPU) > 1:
        strategy = "dp"
    else:
        strategy = None

    # Create the trainer:
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.GPU,
        strategy=strategy,
        default_root_dir=args.checkpoint_dir,

        logger=logger,
        callbacks=callbacks,

        max_epochs=args.epochs,
        fast_dev_run=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        overfit_batches=args.overfit_batches,

        enable_checkpointing=args.enable_checkpointing,
    )
    if args.continue_from_checkpoint_path == "":
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    else:
        ckpt_path = args.continue_from_checkpoint_path
        pl_model = PlModelWrap.load_from_checkpoint(ckpt_path, **kwargs)
        trainer.fit(pl_model, ckpt_path=ckpt_path, train_dataloaders=train_loader, val_dataloaders=valid_loader)



def wandb_interface(args):
    if args.wandb_sweeping or args.wandb_log:
        if args.wandb_sweeping:  # gets the args from wandb
            wandb.init()

            # taking the relevant args from the sweep's parameters
            args_dict = args.__dict__
            for key in wandb.config.keys():
                if wandb.config[key] == "None":  # convert the string Nones
                    args_dict[key] = None
                else:
                    args_dict[key] = wandb.config[key]
            if type(args_dict["hidden_shapes"]) == str:  # "[16, 16, 3]" also works
                args_dict["hidden_shapes"] = eval(args_dict["hidden_shapes"])
            args = Namespace(**args_dict)

            # the agent's CUDA_VISIBLE_DEVICES is alredy set:
            args.GPU = [0]

        # run_id = datetime.datetime.now().strftime("%d-%m-%Y %Hh%Mm%Ss - ") + args.experiment_name
        logger = WandbLogger(project=args.project_name,
                             name=args.experiment_name + f"-f{args.data_fold}",
                             save_dir=args.logs_dir)
        # id=run_id)

        # exclude_lst = []
        # config_args = exclude_wandb_parameters(exclude_lst, args)
        logger.experiment.config.update(args)

    else:
        logger = False
    return logger, args


def exclude_wandb_parameters(exclude_lst, args):
    args_dict = args.__dict__
    for key in exclude_lst:
        del args_dict[key]
    return args_dict


if __name__ == '__main__':
    parser = ArgumentParser()

    # enviroment
    parser.add_argument("--GPU", nargs="+", type=int, default=[2])

    # wandb
    parser.add_argument("-wandb", "--wandb_log", action='store_true')
    parser.add_argument("--logs_dir", default='wandb_logs')
    parser.add_argument("--wandb_sweeping", action='store_true')
    parser.add_argument("-exname", "--experiment_name", default="0 tst")

    # training
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("-lr", "--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("-L2", "--L2", type=float, default=0)
    parser.add_argument("-nw", "--num_workers", type=int, default=24)

    # model
    parser.add_argument("--model", default="ResNet")
    parser.add_argument("--init_features", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bn_momentum", type=float, default=0.05)
    parser.add_argument("-cw", "--class_weights",  nargs="+", type=float, default=[0])  # [0] is auto weight calc

    # MLP
    parser.add_argument("--hidden_shapes", nargs="+", type=int, default=[])  # MLP hidden layers shapes

    # CNN
    parser.add_argument("--cnn_mlp_shapes", nargs="+", type=int, default=[3])  # end of cnn hidden layers shapes
    parser.add_argument("--cnn_mlp_dropout", type=float, default=0.0)
    parser.add_argument("--cnn_dropout", type=float, default=0.0)

    # data
    parser.add_argument("--dataset_class", default="ADNI_Dataset")
    parser.add_argument("--features_set", type=int, default=5)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--adni_dir", default="/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI")
    parser.add_argument("--data_fold", type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument("--num_classes", type=int, choices=[3, 5], default=3)
    parser.add_argument('-tform', "--transform", default="normalize")
    parser.add_argument('-tform_valid', "--transform_valid", default="hippo_crop_2sides")
    parser.add_argument('-l2r', '--load2ram', action='store_true')
    parser.add_argument('-ws', '--with_skull', action='store_true')
    parser.add_argument('-nbfc', '--no_bias_field_correct', action='store_true')
    parser.add_argument('--overfit_batches', type=float, default=0.0)
    parser.add_argument("--class_names", nargs="+", type=str, default=["CN", "MCI", "AD"])
    parser.add_argument("--only_tabular", action='store_true')

    # checkpoints save dir
    parser.add_argument("-cp_dir", "--checkpoint_dir", default="/media/rrtammyfs/Users/daniel/HyperProj_checkpoints")
    parser.add_argument("-ckpt_en", "--enable_checkpointing", action='store_true')

    # resume learning from checkpoint option
    parser.add_argument("-cp_cont_path", "--continue_from_checkpoint_path", default="")

    # project name dont change!
    parser.add_argument("--project_name", default="HyperNets_imgNtabular")

    # running from here or from shell :
    parser.add_argument('-rfs', '--runfromshell', action='store_true')

    args = parser.parse_args()

    if not args.runfromshell:
        print("Running from IDE")
        # --------------------------------------------------------------------------------
        # ----------------------------- input arguments ----------------------------------
        GPU = "2"

        exname = "tst"
        features_set = "15"
        num_classes = "3"
        model = "PreactivResNet_bn_4blks_diff_start_incDrop_mlpend"
        split_seed = "1"
        cnn_dropout = "0.1"
        init_features = "16"
        lr = "0.0001"
        L2 = "0.00001"
        epochs = "180"
        batch_size = "4"
        tform = "hippo_crop_lNr"
        tform_valid = "hippo_crop_2sides"
        num_workers = "0"

        # flags:
        with_skull = ""  # "--with_skull"  or ""
        no_bias_field_correct = "--no_bias_field_correct"  # "--no_bias_field_correct" or ""
        load2ram = ""  # "-l2r" or ""
        wandb_logging = ""  # "-wandb" or ""
        ckpt_en = ""  # -ckpt_en

        adni_dir = "/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
        # --------------------------------------------------------------------------------------------------
        # only for running from here:
        overfit_batches = "--overfit_batches 0.01"  # "--overfit_batches 0.02"
        data_fold = "0"
        # --------------------------------------------------------------------------------------------------

        args_string = f"-exname {exname} --model {model}  --cnn_dropout {cnn_dropout} --init_features {init_features}  -lr {lr} --L2 {L2}  --epochs {epochs} --batch_size {batch_size} --data_fold {data_fold} -tform {tform} --features_set {features_set} --GPU {GPU} {wandb_logging} --adni_dir {adni_dir} -nw {num_workers} {with_skull} {no_bias_field_correct} {load2ram} {overfit_batches} {ckpt_en} --num_classes {num_classes} --split_seed {split_seed}"
        args = parser.parse_args(args_string.split())

    else:
        print("Running from Shell")

    main(args)

