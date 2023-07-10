from easydict import EasyDict

from STS import train_sts_dpc, infer_sts_dpc, get_default_config

config = EasyDict(get_default_config())

config.max_epochs=5
config.log_every_n_steps=1
config.save_checkpoints_every_n_epochs=1
config.do_train = True
best_model_path, config = train_sts_dpc(config)
config.do_train = False
infer_sts_dpc(config, best_model_path)