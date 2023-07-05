
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .DPC.utils import pytorch_lightning_utils 
from .DPC.models.DeepPointCorr.DeepPointCorr import DeepPointCorr

def init():
    seed = 42
    pl.seed_everything(seed=seed)

def get_model(config):
    if(config.resume_from_checkpoint is not None):
        hparams = pytorch_lightning_utils.load_params_from_checkpoint(config)
    model = DeepPointCorr(hparams=config)
    return model

def get_logger(config):
    logger = WandbLogger(save_dir=config.wandb_save_dir, project=config.wandb_project_name, name=config.exp_name)
    logger.log_hyperparams(config)

def get_trainer(config):
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.log_to_dir,
        filename="{epoch:02d}",
        verbose=True,
        save_top_k=-1) # saves checkpoints every epoch,
    
    logger = get_logger(config)

    trainer = pl.Trainer(
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
        num_sanity_val_steps=config.num_sanity_val_steps,
        val_check_interval=config.val_check_interval,  # how many times(0.25=4) to run validation each training loop
        limit_train_batches=config.limit_train_batches ,  # how much of the training data to train on
        limit_val_batches=config.limit_val_batches,  # how much of the validation data to train on
        limit_test_batches=config.limit_test_batches,  # how much of the validation data to train on
        terminate_on_nan=True,
        check_val_every_n_epoch=1,

        # load
        resume_from_checkpoint=config.resume_from_checkpoint,
        replace_sampler_ddp=False,
        accumulate_grad_batches=config.accumulate_grad_batches,
        flush_logs_every_n_steps=config.flush_logs_every_n_steps,
        log_every_n_steps=config.log_every_n_steps,
        reload_dataloaders_every_epoch=False,
        move_metrics_to_cpu =True 
    )
    return trainer
    

def train_sts_dpc(config):
    init()
    model = get_model(config)
    trainer = get_trainer(config)
    trainer.fit(model)

def infer_sts_dpc(config):
    init()
    model = get_model(config)
    trainer = get_trainer(config)
    model = model.load_from_checkpoint("")
    model.output_inference_dir = ""
    os.makedirs(model.output_inference_dir, exist_ok=True)
    model.inference_only = True
    with open(os.path.join(model.output_inference_dir, "orig_model_info.txt"), "w") as f:
        f.write(f"orig checkpoints: \n{config.resume_from_checkpoint}")
    trainer.test(model)