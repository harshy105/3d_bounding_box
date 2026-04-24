from __future__ import annotations
from typing import TYPE_CHECKING, Dict
from torch import Tensor
if TYPE_CHECKING:
    from config import TrainConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.utils import reconstruct_unique_box

class InstanceBoxLoss(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.w_c = config.center_loss_weight
        self.w_s = config.dim_loss_weight
        self.w_r = config.rot_loss_weight
        self.w_corner = config.corner_loss_weight

    def forward(self, pred_c: Tensor, pred_s: Tensor, pred_rot6d: Tensor, 
                targ_c: Tensor, targ_s: Tensor, targ_rot6d: Tensor, 
                targ_corners: Tensor) -> Dict[str, int]:
        # Huber Loss for Center and Dimensions
        loss_c = F.huber_loss(pred_c, targ_c, reduction='mean')
        loss_s = F.huber_loss(pred_s, targ_s, reduction='mean')
        
        # L2 Loss for 6D Rotation
        loss_r = F.mse_loss(pred_rot6d, targ_rot6d, reduction='mean')
        
        # Corner Loss (Differentiable Geometry)
        pred_corners = reconstruct_unique_box(pred_c, pred_s, pred_rot6d)
        loss_corner = F.huber_loss(pred_corners, targ_corners, reduction='mean')
        
        total_loss = (self.w_c * loss_c) + (self.w_s * loss_s) + \
                     (self.w_r * loss_r) + (self.w_corner * loss_corner)
                     
        return total_loss, {
            "loss_center": loss_c.item(), "loss_dims": loss_s.item(), 
            "loss_rot": loss_r.item(), "loss_corner": loss_corner.item()
        }