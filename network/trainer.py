from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Dict, Any
from torch import Tensor
if TYPE_CHECKING:
    from config import NetConfig, TrainConfig

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from network.votenet.votenet import VoteNet
from network.loss_helper import InstanceBoxLoss

class TrainerLitModule(pl.LightningModule):
    def __init__(self, net_config: NetConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = VoteNet(net_config)
        self.criterion = InstanceBoxLoss(train_config)
        self.train_cfg = train_config

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        end_points = self.model(x)
        return end_points["center"], end_points["size"], end_points["rot_6d"]

    def shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tuple[Tensor, Dict[str, int]]:
        pc_pts = batch["pc_pts"]            # (B, N, 3) or (B, N, 6)
        targ_c = batch["bbox_center"]       # (B, 3)
        targ_s = batch["bbox_dims"]         # (B, 3)
        targ_rot6d = batch["bbox_rot_6d"]   # (B, 6)
        targ_corners = batch["bbox_3d"]     # (B, 8, 3)

        pred_c, pred_s, pred_rot6d = self(pc_pts)

        # Compute Loss
        loss, loss_dict = self.criterion(
            targ_c, targ_s, targ_rot6d, targ_corners,
            pred_c, pred_s, pred_rot6d, 
        )
        return loss, loss_dict

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, loss_dict = self.shared_step(batch, batch_idx)
        current_batch_size = batch["pc_pts"].shape[0]
        
        # Log overall loss (Epoch only)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
        
        # Log individual loss components
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=False, on_epoch=True, batch_size=current_batch_size)
            
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, loss_dict = self.shared_step(batch, batch_idx)
        current_batch_size = batch["pc_pts"].shape[0]
        
        # Log overall loss (Epoch only)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
        
        # Log individual loss components
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v, on_step=False, on_epoch=True, batch_size=current_batch_size)
            
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.parameters(), 
            lr=self.train_cfg.lr, 
            weight_decay=self.train_cfg.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3, 
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }