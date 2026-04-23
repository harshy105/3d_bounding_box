from network.trainer import TrainerLitModule
from pytorch_lightning import Trainer
from config import Paths, DataLoaderConfig, NetConfig, TrainConfig
from data_loader import InstanceDataModule

if __name__ == "__main__":
    # 1. Initialize all configurations
    dl_cfg = DataLoaderConfig()
    net_cfg = NetConfig()
    train_cfg = TrainConfig()
    
    # 2. Initialize the DataModule
    data_module = InstanceDataModule(
        parsed_data_path=Paths.parsed_data, 
        data_loader_config=dl_cfg
    )
    
    # 3. Initialize the Lightning Module
    model = TrainerLitModule(net_cfg, train_cfg)
    
    # 4. Initialize Trainer and start training
    trainer = Trainer(
        max_epochs=train_cfg.max_epochs, 
        accelerator="auto",
        log_every_n_steps=10
    )
    
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)