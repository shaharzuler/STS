import json
import os

def get_default_config() -> dict:
    with open("/home/shahar/cardio_corr/my_packages/sts_project/STS/STS/src/configs/default_config.json") as file:
        args = json.load(file)
    return args

def save_config(config:dict, save_dir:str):
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        f.write(json.dumps(config))



