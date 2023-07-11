from easydict import EasyDict

from STS import train_sts_dpc, infer_sts_dpc, get_default_config

config = EasyDict(get_default_config())

config.max_epochs=5000
config.log_every_n_steps=10
config.save_checkpoints_every_n_epochs=10

best_model_path, config = train_sts_dpc(config)

infer_output_path = infer_sts_dpc(config, best_model_path)

print(1)