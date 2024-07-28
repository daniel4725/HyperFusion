import yaml
from easydict import EasyDict
import os
from datetime import datetime
import sys

def easydict_to_dict(edict):
    if isinstance(edict, EasyDict):
        edict = dict(edict)
        for key, value in edict.items():
            edict[key] = easydict_to_dict(value)
    return edict

def change_cwd():
    # change to the current script's directory path
    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)


def get_default_config(eval=False):
    type = "eval" if eval else "train"
    default_cfg_path = f"/home/duenias/PycharmProjects/HyperFusion/experiments/brain_age_prediction/default_{type}_config.yml"
    with open(default_cfg_path, 'r') as file:
        config = EasyDict(yaml.safe_load(file))
    return config

def save_config(config):
    # Save the YAML data to a file
    config = easydict_to_dict(config)
    config_str = yaml.dump(config)
    time_suffix = datetime.now().strftime(r"%y%m%m_%H%M%S_%f")
    config_dir = os.path.join(os.getcwd(), "temp_configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{time_suffix}.yaml")
    with open(config_path, 'w') as file:
        file.write(config_str)
    return config_path