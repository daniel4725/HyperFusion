from model_trainer import *
from sklearn.manifold import TSNE
import plotly.graph_objects as go


class ModelsEnsemble(nn.Module):
    """ ensemble model class"""
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList()

    def append(self, model):
        self.models.append(model)


    # def forward(self, x):
    #     return [model(x).softmax(dim=-1) for model in self.models]

    def forward(self, x):
        # self.models.train()
        out = self.models[0](x).softmax(dim=-1)
        # out = self.models[0](x)
        for model in self.models[1:]:
            out += model(x).softmax(dim=-1)
            # out += model(x)

        return out / len(self.models)
        # return out


# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

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


    # test_loader.dataset.metadata = test_loader.dataset.metadata.iloc[:40]
    kwargs = args.__dict__

    models = []
    for fold_directory in os.listdir(os.path.join(args.checkpoint_dir, args.experiment_name)):
        model_path = os.path.join(args.checkpoint_dir, args.experiment_name, fold_directory, "best_val.ckpt")
        # model_path = "/home/duenias/PycharmProjects/HyperNetworks/tmp_cpdir/TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw1_va-seed0-fs12/fold_0/best_val.ckpt"
        m = PlModelWrap.load_from_checkpoint(model_path).model
        models.append(m)

    labels_dict = {0: "CN", 1: "MCI", 2:"AD"}
    test_iter = iter(test_loader)
    embeddings = [[], [], [], []]
    labels = np.array([], dtype='<U3')
    for batch in test_iter:
        imgs, tabular, y = batch
        labels = np.concatenate([labels, np.array([labels_dict[l.item()] for l in y])])
        for i, model in enumerate(models):
            embd = model.block4.downsample[0].layer.hyper_net.embedding_model(tabular).detach().numpy()
            # embd = tabular.detach().numpy()
            # embd = model((imgs, tabular)).detach().numpy()
            # embd = model.mlp[2](model.mlp[1](model.mlp[0](tabular))).detach().numpy()
            embeddings[i].append(embd)
            # weights, biases = model.block4.downsample[0].layer.hyper_net(tabular)
            # embeddings[i].append(weights.detach().numpy())

    for i in range(4):
        embedding_vectors = np.concatenate(embeddings[i])
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=0)
        embedded_vectors = tsne.fit_transform(embedding_vectors)

        # Plot the embedded vectors with color-coded labels
        label_colors = {"AD": "red", "MCI": "green", "CN": "blue"}
        for label in label_colors.keys():
            mask = labels == label
            plt.scatter(embedded_vectors[mask, 0], embedded_vectors[mask, 1], label=label, color=label_colors[label])

        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Plot of Embedding Vectors with Labels')
        plt.legend()
        plt.show()

    for i in range(4):
        data = np.concatenate(embeddings[i])

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=3, random_state=0)
        tsne_data = tsne.fit_transform(data)

        # Create a scatter plot with 3D coordinates
        color = []
        for l in labels:
            if l == "CN":
                color.append("blue")
            if l == "MCI":
                color.append("green")
            if l == "AD":
                color.append("red")

        fig = go.Figure(data=go.Scatter3d(
            x=tsne_data[:, 0],
            y=tsne_data[:, 1],
            z=tsne_data[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                colorscale='Viridis',
                opacity=0.8
            )
        ))

        # Set axis labels and title
        fig.update_layout(scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
        ), title='Interactive 3D t-SNE Visualization')

        # Show the plot
        fig.show()


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
                             name=args.experiment_name + f"-EvalTest-fset_{args.features_set}-classes{args.num_classes}",
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

        # exname = "baseline-tabular_embd8_v2-seed0-fs12"
        # exname = "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed0-fs12"
        # exname = "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v1-seed0-fs12"
        exname = "TabularAsHyper_R_R_R_FFT_FF_embd8_2losses_cw085_v1-seed0-fs13"
        exname = "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v1-seed0-fs13"
        features_set = "13"
        num_classes = "3"
        split_seed = "0"



        model = "MLP4Tabular"
        cnn_dropout = "0.1"
        init_features = "4"
        lr = "0.0001"
        L2 = "0.00001"
        epochs = "180"
        batch_size = "32"
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

    # print(args)
    # raise ValueError("------------ end program ----------------")
    main(args)

