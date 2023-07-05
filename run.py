from easydict import EasyDict

from STS import train_sts_dpc, infer_sts_dpc, get_default_config

config = EasyDict(get_default_config())
train_sts_dpc(config)
infer_sts_dpc(config)