import os
import logging
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium') # for speed up GPU with Tensor cores

from network.trainer import TrainerLitModule
from config import Paths, DataLoaderConfig, NetConfig, TrainConfig
from data_loader import InstanceDataModule

def setup_text_logger(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")
    
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate logs if run multiple times in same session
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

class EpochTextLoggerCallback(Callback):
    """
    Grabs the aggregated training and validation losses at the end of 
    every epoch and writes them to the custom text logger.
    """
    def __init__(self, text_logger: logging.Logger):
        self.text_logger = text_logger

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Skip logging during the initial sanity check step before training starts
        if trainer.sanity_checking:
            return

        # Lightning stores all self.log() outputs in trainer.callback_metrics
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Safely extract the metrics (they are stored as single-value tensors)
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        log_str = f"Epoch {epoch:03d} Summary | "
        if train_loss is not None:
            log_str += f"Train Loss: {train_loss.item():.4f} | "
        if val_loss is not None:
            log_str += f"Val Loss: {val_loss.item():.4f}"

        self.text_logger.info(log_str)

class EarlyStopOnMinLR(Callback):
    """
    Stops training if the optimizer's learning rate falls to/below `min_lr` 
    and stays there for `patience` consecutive epochs.
    """
    def __init__(self, min_lr: float = 1e-5, patience: int = 3, text_logger: logging.Logger = None):
        self.min_lr = min_lr
        self.patience = patience
        self.wait_count = 0
        self.text_logger = text_logger

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        opt = trainer.optimizers[0]
        current_lr = opt.param_groups[0]['lr']
        
        if current_lr <= self.min_lr:
            self.wait_count += 1
            if self.text_logger:
                self.text_logger.info(f"Epoch {trainer.current_epoch}: LR is {current_lr:.1e}. Wait count: {self.wait_count}/{self.patience}")
        else:
            self.wait_count = 0
            
        if self.wait_count >= self.patience:
            msg = f"\nEarly stopping triggered: Learning rate reached threshold ({self.min_lr}) for {self.patience} consecutive epochs."
            if self.text_logger:
                self.text_logger.info(msg)
            else:
                print(msg)
            trainer.should_stop = True

if __name__ == "__main__":
    model_name = "base"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}"
    
    # Setup custom logging directory specifically for this run
    run_log_dir = os.path.join(Paths.logs, run_name)
    text_logger = setup_text_logger(run_log_dir, run_name)
    text_logger.info(f"Initialized Configuration for run: {run_name}")
    
    # Initialize all configurations
    dl_cfg = DataLoaderConfig()
    net_cfg = NetConfig()
    train_cfg = TrainConfig()
    
    # Initialize the DataModule
    data_module = InstanceDataModule(
        parsed_data_path=Paths.parsed_data, 
        data_loader_config=dl_cfg
    )
    
    # Initialize the Lightning Module
    model = TrainerLitModule(net_cfg, train_cfg)
    
    # Configure TensorBoard Logger 
    tb_logger = TensorBoardLogger(
        save_dir=Paths.logs,
        name=run_name,
        version="" 
    )
    
    # Configure Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(Paths.ckpts, run_name),
        filename=f"{run_name}-{{epoch:02d}}-{{val_loss:.4f}}", 
        monitor="val_loss",  
        mode="min",         
        save_top_k=2,        
        save_weights_only=False 
    )
    
    # Initialize Custom LR Early Stopping
    lr_early_stop_callback = EarlyStopOnMinLR(
        min_lr=train_cfg.min_lr, 
        patience=3, 
        text_logger=text_logger
    )
    
    # Initialize Epoch Text Logger 
    epoch_text_logger_callback = EpochTextLoggerCallback(text_logger=text_logger)
    
    # Initialize Trainer and start training
    trainer = Trainer(
        max_epochs=train_cfg.max_epochs, 
        accelerator="auto",
        log_every_n_steps=10,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_early_stop_callback, epoch_text_logger_callback] 
    )
    
    text_logger.info(f"Starting training run: {run_name}...")
    trainer.fit(model, datamodule=data_module)
    text_logger.info("Training Finished.")