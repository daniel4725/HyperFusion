from exp_utils import *
change_cwd()
cfg = get_default_config()
args = sys.argv
gpu, fold, vers, seed, f_set = args[1:]
gpu, fold, seed, f_set = int(gpu), int(fold), int(seed), int(f_set)
# -------------------------------------------------
# ------ update default config below --------------


cfg.trainer.gpu = [gpu]
cfg.data_module.dataset_cfg.fold = fold
cfg.data_module.dataset_cfg.split_seed = seed

cfg.wandb.project_name = "HyperFusion_revision"
# cfg.wandb.project_name = "HyperNets_imgNtabular"
# cfg.wandb.project_name = "testing"

cfg.experiment_name = f"HyperFusion_resize_brain{vers}-seed{seed}-fs{f_set}"
cfg.model.model_name = "HyperFusion_AD_untraind_tabular"
cfg.data_module.dataset_cfg.features_set = f_set
cfg.trainer.epochs = 250
cfg.lightning_wrapper.loss.class_weights = [1.1, 0.6962, 1.4]

cfg.data_module.dataset_cfg.transform_train = "crop_lNr_after_l2r"
cfg.data_module.dataset_cfg.l2r_tform_train = "brain_resize_2sides_for_load_2_ram_func_train"
cfg.data_module.dataset_cfg.transform_valid = "None"
cfg.data_module.dataset_cfg.l2r_tform_valid = "brain_resize_2sides_for_load_2_ram_func_val"


cfg.checkpointing.ckpt_dir = "/home/duenias/PycharmProjects/tmp_ckpts"


# flags:
cfg.data_module.dataset_cfg.load2ram = True
cfg.checkpointing.enable = True
# cfg.data_module.dataset_cfg.only_tabular = True



# -----------------------------------------------------------------
# ------- saving the temp config and executing the training -------
config_path = save_config(cfg)
command = f"python3 /home/duenias/PycharmProjects/HyperFusion/train.py --config_path {config_path}"
print(f"executing experiment with config path: {config_path}")
os.system(command)
os.remove(config_path)