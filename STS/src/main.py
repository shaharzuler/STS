
import os
import time
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from .DPC.utils import pytorch_lightning_utils 
from .DPC.models.DeepPointCorr.DeepPointCorr import DeepPointCorr

def init(config):
    seed = 42
    pl.seed_everything(seed=seed)
    config.log_to_dir += time.strftime("_%Y%m%d_%H%M%S")
    config.visualization_save_dir = os.path.join(config.log_to_dir, "visualization")
    config.wandb_save_dir = os.path.join(config.log_to_dir, "wandb")
    config.checkpoints_save_dir = os.path.join(config.log_to_dir, "checkpoints")
    Path(config.visualization_save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.wandb_save_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoints_save_dir).mkdir(parents=True, exist_ok=True)
    return config


def get_model_module(config):
    if(config.resume_from_checkpoint is not None):
        config = pytorch_lightning_utils.load_params_from_checkpoint(config)
    model_module = DeepPointCorr
    return model_module

def get_logger(config):
    logger = WandbLogger(save_dir=config.wandb_save_dir, project=config.wandb_project_name, name=config.exp_name)
    logger.log_hyperparams(config)
    return logger

def get_trainer(config):
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoints_save_dir,
        monitor="train/train_tot_loss/epoch",
        mode="min",
        filename="{epoch:02d}",
        verbose=True,
        save_top_k=3,
        period=config.save_checkpoints_every_n_epochs)
    
    logger = get_logger(config)

    trainer = pl.Trainer(
        # default_root_dir=config.default_root_dir,
        callbacks=[checkpoint_callback],
        log_gpu_memory="all",
        weights_summary="top",
        logger=logger,
        max_epochs=config.max_epochs,
        precision=config.precision,
        auto_lr_find=False,  
        gradient_clip_val=config.gradient_clip_val,
        benchmark=True,  
        gpus=1,
        distributed_backend=None,
        # num_sanity_val_steps=config.num_sanity_val_steps,
        # val_check_interval=config.val_check_interval,  # how many times(0.25=4) to run validation each training loop
        limit_train_batches=config.limit_train_batches ,  # how much of the training data to train on
        # limit_val_batches=config.limit_val_batches,  # how much of the validation data to train on
        # limit_test_batches=config.limit_test_batches,  # how much of the validation data to train on
        terminate_on_nan=True,

        resume_from_checkpoint=config.resume_from_checkpoint,
        replace_sampler_ddp=False,
        accumulate_grad_batches=config.accumulate_grad_batches,
        flush_logs_every_n_steps=config.flush_logs_every_n_steps,
        log_every_n_steps=config.log_every_n_steps,
        reload_dataloaders_every_epoch=False,
        move_metrics_to_cpu =True 
    )
    return trainer, checkpoint_callback
    

def train_sts_dpc(config):
    config.mode='train'
    config.inference = False
    config = init(config)
    model = get_model_module(config)(hparams=config)
    trainer, checkpoint_callback = get_trainer(config)
    trainer.fit(model)
    return checkpoint_callback.best_model_path, config

def infer_sts_dpc(config, ckpts_path): 
    model_module = get_model_module(config)
    config.max_epochs = 1
    model = model_module.load_from_checkpoint(checkpoint_path=ckpts_path, hparams=config)
    trainer, checkpoint_callback = get_trainer(config)
    trainer.fit(model) ### pl "predict" and "test" of this pl version are buggy. I'm gonna use the fit and let it train on neutral for 1 epoch. instead of return, batch is stored in model as model.batch. ¯\_(ツ)_/¯
    pred = model.batch
    inference_path = trainer.save_inference(pred)
    return inference_path



    print(1)


