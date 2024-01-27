from train import *
import re

class ModelsEnsemble(nn.Module):
    """ ensemble model class"""
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList()

    def append(self, model):
        self.models.append(model)

    def forward(self, x):
        # out = self.average_softmax_prediction(x)
        out = self.confidence_weighted_average_softmax_prediction(x)
        # out = self.confidence_weighted_majority_voting_prediction(x)
        # out = self.most_confidence_prediction(x)
        return out

    def average_softmax_prediction(self, x):
        # self.models.train()
        out = self.models[0](x).softmax(dim=-1)
        # out = self.models[0](x)
        for model in self.models[1:]:
            out += model(x).softmax(dim=-1)
            # out += model(x)
        # entropy(model(x).softmax(dim=-1).cpu().detach(), axis=1)
        out = out / len(self.models)
        return out

    def confidence_weighted_average_softmax_prediction(self, x):  # weighted (by entropy) average softmax
        # self.models.train()
        probs = []
        for model in self.models:
            probs.append(model(x).softmax(dim=-1))

        entropies = []
        for prob in probs:
            entropies.append(entropy(prob).view(-1, 1))
        inverse_entropies = 1 / torch.cat(entropies, dim=-1)
        weights = inverse_entropies / inverse_entropies.sum(dim=1, keepdim=True)

        out = probs[0] * weights[:, 0].view(-1, 1)
        for i in range(len(probs)-1):
            out += probs[i + 1] * weights[:, i + 1].view(-1, 1)

        return out


    def confidence_weighted_majority_voting_prediction(self, x):  # weighted (by entropy) voting
        # self.models.train()
        probs = []
        for model in self.models:
            probs.append(model(x).softmax(dim=-1))

        entropies = []
        for prob in probs:
            entropies.append(entropy(prob).view(-1, 1))
        inverse_entropies = 1 / torch.cat(entropies, dim=-1)
        weights = inverse_entropies / inverse_entropies.sum(dim=1, keepdim=True)

        for i in range(len(probs)):
            probs[i] = (probs[i] == torch.max(probs[i], dim=1, keepdim=True).values).float()

        out = probs[0] * weights[:, 0].view(-1, 1)
        for i in range(len(probs)-1):
            out += probs[i + 1] * weights[:, i + 1].view(-1, 1)
        return out


    def most_confidence_prediction(self, x): # most confident decides
        # self.models.train()
        probs = []
        for model in self.models:
            probs.append(model(x).softmax(dim=-1))

        entropies = []
        for prob in probs:
            entropies.append(entropy(prob).view(-1, 1))
        entropies = torch.cat(entropies, dim=-1)

        # gives weight of 1 to the most confident model and 0 to the rest
        weights = (entropies == torch.min(entropies, dim=1, keepdim=True).values).float()
        out = probs[0] * weights[:, 0].view(-1, 1)
        for i in range(len(probs)-1):
            out += probs[i + 1] * weights[:, i + 1].view(-1, 1)
        return out


def entropy(probabilities):
    # Calculate the negative log probabilities (cross-entropy) and then multiply by probabilities
    negative_log_probs = -torch.log(probabilities)
    entropy_values = torch.sum(probabilities * negative_log_probs, dim=-1)
    return entropy_values


def main(args):
    # torch.manual_seed(0)
    logger, args = wandb_interface(args)  # this line must be first

    # Create the data loaders:
    test_loader = get_test_loader(batch_size=args.batch_size, features_set=args.features_set, adni_dir=args.adni_dir,
                                  fold=args.data_fold, num_workers=args.num_workers,
                                  transform=tform_dict[args.transform_valid], load2ram=args.load2ram,
                                  only_tabular=args.only_tabular, num_classes=args.num_classes,
                                  with_skull=args.with_skull, no_bias_field_correct=args.no_bias_field_correct,
                                  split_seed=args.split_seed)


    kwargs = args.__dict__

    model = ModelsEnsemble()

    if args.versions is None:
        print(f"loading experiment: {args.experiment_name}")
        for fold_directory in os.listdir(os.path.join(args.checkpoint_dir, args.experiment_name)):
            model_path = os.path.join(args.checkpoint_dir, args.experiment_name, fold_directory, "best_val.ckpt")
            m = PlModelWrap.load_from_checkpoint(model_path).model
            model.append(m)
    else:
        versions = args.versions.split(",")
        experiment_base_name = re.sub(r"_v\d-", "{}-", args.experiment_name)
        for v in versions:
            experiment_name = experiment_base_name.format(v)
            print(f"loading experiment: {experiment_name}")
            experiment_dir = os.path.join(args.checkpoint_dir, experiment_name)
            for fold_directory in os.listdir(experiment_dir):
                model_path = os.path.join(experiment_dir, fold_directory, "best_val.ckpt")
                m = PlModelWrap.load_from_checkpoint(model_path).model
                model.append(m)


    kwargs["class_weights"] = torch.Tensor([1 for _ in list(model.models[0].parameters())[-1]])
    kwargs["model"] = model
    kwargs["class_names"] = list(test_loader.dataset.labels_dict.keys())

    pl_model = PlModelWrap4test(**kwargs)

    # Callbacks:
    callbacks = [TimeEstimatorCallback(**kwargs)]

    if len(args.GPU) > 1:
        strategy = "dp"
    else:
        strategy = None

    # Create the trainer:
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.GPU,
        strategy=strategy,

        logger=logger,
        callbacks=callbacks,

        max_epochs=args.epochs,
        fast_dev_run=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        overfit_batches=args.overfit_batches,

        enable_checkpointing=False,
    )

    # trainer.test(pl_model, dat)
    trainer.test(pl_model, test_dataloaders=test_loader)



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
                             name=args.experiment_name + f"-{args.versions}_W_EvalTest-fset_{args.features_set}-classes{args.num_classes}",
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
    parser.add_argument("--versions", default=None)
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
        GPU = "1"

        exname = "T_AsHyper_R_R_FFT_FFF_FF_embd_trainedTabular8_cw085_v2-seed0-fs12"  # the experiment name is the file name
        features_set = "12"  # set 4 is norm minmax (0 to 1), set 5 is std-mean
        num_classes = "3"
        split_seed = "0"

        model = "MLP4Tabular"
        cnn_dropout = "0.1"
        init_features = "4"
        lr = "0.0001"
        L2 = "0.00001"
        epochs = "180"
        batch_size = "16"
        tform = "hippo_crop_lNr"  # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
        tform_valid = "hippo_crop_2sides"  # hippo_crop_2sides hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scaletform_valid="hippo_crop_2sides"   # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
        num_workers = "8"

        # flags:
        with_skull = ""  # "--with_skull"  or ""
        no_bias_field_correct = "--no_bias_field_correct"  # "--no_bias_field_correct" or ""
        load2ram = ""  # "-l2r" or ""
        wandb_logging = ""  # "-wandb" or ""
        ckpt_en = "-ckpt_en"

        adni_dir = "/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
        # --------------------------------------------------------------------------------------------------
        # only for running from here:
        overfit_batches = ""  # "--overfit_batches 0.05"
        data_fold = "0"
        # --------------------------------------------------------------------------------------------------

        args_string = f"-exname {exname} --model {model}  --cnn_dropout {cnn_dropout} --init_features {init_features}  -lr {lr} --L2 {L2}  --epochs {epochs} --batch_size {batch_size} --data_fold {data_fold} -tform {tform} --features_set {features_set} --GPU {GPU} {wandb_logging} --adni_dir {adni_dir} -nw {num_workers} {with_skull} {no_bias_field_correct} {load2ram} {overfit_batches} {ckpt_en} --num_classes {num_classes} --split_seed {split_seed}"
        args = parser.parse_args(args_string.split())

    else:
        print("Running from Shell")

    main(args)

    # w = torch.Tensor([[0.9, 0.9, 1]])
    # w = w.to(out.device)
    # out = out * w


