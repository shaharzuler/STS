import json
import os

def get_default_config(machine:int=0) -> dict:
    if machine==0:
        config_path = "/home/shahar/cardio_corr/my_packages/sts_project/STS/STS/src/configs/default_config.json"
    elif machine==3:
        config_path = "/home/shahar/cardio_corr/sts_project/STS/STS/src/configs/default_config.json"
    with open(config_path) as file:
        args = json.load(file)
    return args

def save_config(config:dict, save_dir:str):
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        f.write(json.dumps(config))



