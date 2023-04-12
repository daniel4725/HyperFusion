from data_handler import *
from models import *
from pl_wrap import *
from utils import *
from MLP_models import *
from models_hyper import *
from models_concat import *
from models_img_hyper import *
from Film_DAFT.models_film_daft import *
from Film_DAFT_preactive_block.models_film_daft import *
from tformNaugment import tform_dict
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import WandbLogger
import wandb
from costum_callbacks import TimeEstimatorCallback, CheckpointCallback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    logger, args = wandb_interface(args)  # this line must be first

    # Create the data loaders:
    loaders = get_dataloaders(batch_size=args.batch_size, metadata_path=args.metadata_path, adni_dir=args.adni_dir,
                                fold=args.data_fold, num_workers=args.num_workers, transform_train=tform_dict[args.transform],
                                transform_valid=tform_dict['hippo_crop_2sides'], load2ram=args.load2ram, classes=args.class_names)
    train_loader, valid_loader = loaders


    # Create the model:
    args.class_weights = get_class_weight(train_loader, valid_loader)
    # args.class_weights *= 0
    # args.class_weights += 1
    print("class weights: ", args.class_weights)


    kwargs = args.__dict__
    kwargs["n_tabular_features"] = train_loader.dataset.num_tabular_features
    kwargs['n_outputs'] = len(args.class_weights)
    kwargs["cnn_mlp_shapes"][-1] = len(args.class_weights)
    
    # mlp_layers_shapes = [n_tabular_features, hidden_shapes, num_classes]
    kwargs["mlp_layers_shapes"] = [kwargs["n_tabular_features"]] + kwargs["hidden_shapes"] + [len(kwargs["class_weights"])]

    model = globals()[args.model](**kwargs)
    kwargs["model"]  = model

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
        resume_from_checkpoint=args.continue_from_checkpoint
        )

    if not args.continue_from_checkpoint:
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    else:  # TODO - resume from ckpt not ready
        PATH = "checkpoints/cp.ckpt"
        pl_model = PlModelWrap.load_from_checkpoint(PATH)
        trainer.fit(pl_model, ckpt_path=PATH, train_dataloaders=train_loader, val_dataloaders=valid_loader)


    # if args.wandb_sweeping or args.wandb_log:
    #     wandb.finish()


def wandb_interface(args):
    if args.wandb_sweeping or args.wandb_log:
        if args.wandb_sweeping:  # gets the args from wandb
            wandb.init()

            # taking the relevant args from the sweep's parameters
            args_dict = args.__dict__
            for key in wandb.config.keys():
                if wandb.config[key] == "None": # convert the string Nones
                    args_dict[key] = None
                else:
                    args_dict[key] = wandb.config[key]
            if type(args_dict["hidden_shapes"]) == str: # "[16, 16, 3]" also works
                args_dict["hidden_shapes"] = eval(args_dict["hidden_shapes"])
            args = Namespace(**args_dict)

            # the agent's CUDA_VISIBLE_DEVICES is alredy set:
            args.GPU = [0]

        # run_id = datetime.datetime.now().strftime("%d-%m-%Y %Hh%Mm%Ss - ") + args.experiment_name
        logger = WandbLogger(project=args.project_name,
                            name=args.experiment_name,
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
    parser.add_argument("-nw", "--num_workers", type=int, default=4)

    # model
    parser.add_argument("--model", default="ResNet")
    parser.add_argument("--init_features", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bn_momentum",  type=float, default=0.05)
    # parser.add_argument("-cw", "--class_weights",  nargs="+", type=float, default=[0])  # [0] is auto weight calc

    # MLP
    parser.add_argument("--hidden_shapes",  nargs="+", type=int, default=[1])  # MLP hidden layers shapes

    # CNN
    parser.add_argument("--cnn_mlp_shapes",  nargs="+", type=int, default=[3])  # end of cnn hidden layers shapes
    parser.add_argument("--cnn_mlp_dropout",  type=float, default=0.0)
    parser.add_argument("--cnn_dropout",  type=float, default=0.0)

    # data
    parser.add_argument("--metadata_path", default="metadata_by_features_sets/set-1.csv")
    parser.add_argument("--adni_dir", default="/media/rrtammyfs/Users/daniel/adni_class_pred_1x1x1_v1")
    parser.add_argument("--data_fold", type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('-tform', "--transform", default="normalize")
    parser.add_argument('-l2r', '--load2ram', action='store_true')
    parser.add_argument('--overfit_batches', type=float, default=0.0)
    parser.add_argument("--class_names",  nargs="+", type=str, default=["CN", "MCI", "AD"])  

    # checkpoints save dir
    parser.add_argument("-cp_dir", "--checkpoint_dir", default="checkpoints")
    parser.add_argument("-cp_en", "--enable_checkpointing",  action='store_true')

    # resume learning from checkpoint option
    parser.add_argument("-cp_cont", "--continue_from_checkpoint", action='store_true')
    parser.add_argument("--cp_cont_epoch", type=int, default=1)

    # project name dont change!
    parser.add_argument("--project_name", default="HyperNets_imgNtabular")

    # running from here or from shell :
    parser.add_argument('-rfs', '--runfromshell', action='store_true')

    args = parser.parse_args()

    if not args.runfromshell:
        print("Running from IDE")
        # env
        args.GPU = [1]
        args.num_workers = 0

        # logging and checkpointing
        args.wandb_log = False
        args.experiment_name = "tst"
        args.enable_checkpointing = False

        # model
        args.model = "age_noembd_lastblockHyp_FFT_fcHyp_2"  #"PreactivResNet_instN"  'ResNet' 'MLP4Tabular' 'ResNetHyperEnd'
        args.init_features = 32
        args.hidden_shapes = [24, 32, 5]
        # args.dropout = 0.2
        # args.bn_momentum = 0.05
        # args.cnn_mlp_shapes = [8, 3]
        # args.cnn_mlp_dropout = 0.1
        args.cnn_dropout = 0.1

        # training
        args.epochs = 30
        args.lr = 0.0001
        args.batch_size = 4
        args.L2 = 0.00001
        args.overfit_batches = 0.05  # 0.0 for regular training or comment this line

        # data
        # args.adni_dir = "/usr/local/faststorage/adni_class_pred_1x1x1_v1"
        # args.metadata_path = "metadata_by_features_sets/set-1.csv"
        args.transform = "hippo_crop_lNr"  # basic_aug  normalize  hippo_crop  hippo_crop_lNr
        args.data_fold = 0
        args.class_names = ["CN", "MCI", "AD"]


    else:
        print("Running from Shell")

    # print(args)
    # raise ValueError("------------ end program ----------------")
    main(args)

