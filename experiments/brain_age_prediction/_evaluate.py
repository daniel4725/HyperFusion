from exp_utils import *
change_cwd()
cfg = get_default_config(eval=True)
args = sys.argv
gpu, vers, seed, f_set, experiment = args[1:]
gpu, seed, f_set = int(gpu), int(seed), int(f_set)
# -------------------------------------------------
# ------ update default config below --------------


cfg.trainer.gpu = [gpu]
cfg.data_module.dataset_cfg.split_seed = seed

cfg.wandb.project_name = "HyperFusion_revision_test"

cfg.experiment_name = f"{experiment}{vers}-seed{seed}-fs{f_set}"
cfg.data_module.dataset_cfg.features_set = f_set


cfg.checkpointing.ckpt_dir = "/home/duenias/PycharmProjects/tmp_ckpts"

# flags:
cfg.data_module.dataset_cfg.load2ram = False
cfg.checkpointing.enable = False
# cfg.data_module.dataset_cfg.only_tabular = True



# -----------------------------------------------------------------
# ------- saving the temp config and executing the training -------
config_path = save_config(cfg)
command = f"python3 /home/duenias/PycharmProjects/HyperFusion/eval.py --config_path {config_path}"
print(f"executing evaluation with config path: {config_path}")
os.system(command)
os.remove(config_path)