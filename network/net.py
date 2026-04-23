from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import NetConfig, TrainConfig

import pytorch_lightning as pl
from torch.optim import AdamW

from network.modules import InstanceVoteNet
from network.loss import InstanceBoxLoss

class Net(pl.LightningModule):
    def __init__(self, net_config: NetConfig, train_config: TrainConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model = InstanceVoteNet(net_config)
        self.criterion = InstanceBoxLoss(train_config)
        self.train_cfg = train_config

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, batch_idx):
        # Note: Adjust these key names based on your LMDB dataset output
        pc_pts = batch["pc_pts"]            # (B, N, 3)
        targ_c = batch["bbox_center"]       # (B, 3)
        targ_s = batch["bbox_dims"]         # (B, 3)
        targ_rot6d = batch["bbox_rot_6d"]   # (B, 6)
        targ_corners = batch["bbox_3d"]     # (B, 8, 3)

        # Forward pass
        pred_c, pred_s, pred_rot6d = self(pc_pts)

        # Compute Loss
        loss, loss_dict = self.criterion(
            pred_c, pred_s, pred_rot6d, 
            targ_c, targ_s, targ_rot6d, targ_corners
        )
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.train_cfg.lr, 
            weight_decay=self.train_cfg.weight_decay
        )
        return optimizer