import os
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium') # for speed up GPU with Tensor cores

from network.trainer import TrainerLitModule
from config import Paths, DataLoaderConfig, NetConfig, TrainConfig
from data_loader import InstanceDataModule

if __name__ == "__main__":
    model_name = "base"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_name}"
    
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
    logger = TensorBoardLogger(
        save_dir=Paths.logs,
        name=run_name,
        version="" 
    )
    
    # Configure Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(Paths.ckpts, run_name),
        filename=f"{run_name}-{{epoch:02d}}-{{val_loss:.4f}}", # Appends epoch and loss to filename
        monitor="val_loss",  # Make sure you are logging "val_loss" in your validation_step!
        mode="min",          # "min" because we want the lowest validation loss
        save_top_k=2,        # Keep only the best 2
        save_weights_only=False # Set to True if you want to save space by dropping optimizer states
    )
    
    # Initialize Trainer and start training
    trainer = Trainer(
        max_epochs=train_cfg.max_epochs, 
        accelerator="auto",
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    print(f"Starting training run: {run_name}...")
    trainer.fit(model, datamodule=data_module)