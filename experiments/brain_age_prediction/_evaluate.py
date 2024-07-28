from exp_utils import *
change_cwd()
cfg = get_default_config(eval=True)
args = sys.argv
gpu, vers, experiment = args[1:]
gpu = int(gpu)
# -------------------------------------------------
# ------ update default config below --------------

cfg.trainer.gpu = [gpu]

cfg.wandb.project_name = "HyperFusion_revision_brainage_test"

cfg.experiment_name = f"{experiment}{vers.split(',')[0]}"

cfg.versions = vers

cfg.data_module.dataset_cfg.partial_data = None

# flags:
cfg.checkpointing.enable = False
# cfg.data_module.dataset_cfg.only_tabular = True



# -----------------------------------------------------------------
# ------- saving the temp config and executing the training -------
config_path = save_config(cfg)
command = f"python3 /home/duenias/PycharmProjects/HyperFusion/eval.py --config_path {config_path}"
print(f"executing evaluation with config path: {config_path}")
os.system(command)
os.remove(config_path)