import json

def get_default_config() -> dict:
    with open("/home/shahar/cardio_corr/my_packages/sts_project/STS/STS/src/configs/default_config.json") as file:
        args = json.load(file)
    return args


