from exp_utils import *
change_cwd()
cfg = get_default_config()
args = sys.argv
gpu, vers = args[1:]
gpu = int(gpu)
# gpu, vers = 2, "_v1"
# -------------------------------------------------
# ------ update default config below --------------


cfg.trainer.gpu = [gpu]

cfg.wandb.project_name = "brainage_revision"

cfg.experiment_name = f"B_fill__FT_FFF{vers}"
cfg.model.model_name = "B_fill__FT_FFF"
cfg.trainer.epochs = 60

cfg.data_module.dataset_cfg.partial_data = None

# flags:
cfg.checkpointing.enable = True
cfg.wandb.enable = True


# -----------------------------------------------------------------
# ------- saving the temp config and executing the training -------
config_path = save_config(cfg)
command = f"python3 /home/duenias/PycharmProjects/HyperFusion/train.py --config_path {config_path}"
print(f"executing experiment with config path: {config_path}")
os.system(command)
os.remove(config_path)