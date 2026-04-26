from __future__ import annotations
from typing import TYPE_CHECKING, Dict
from torch import Tensor
if TYPE_CHECKING:
    from config import TrainConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.utils import reconstruct_unique_box

def geodesic_loss(R_pred: Tensor, R_targ: Tensor) -> Tensor:
    """
    Based on the paper Zhou et. al. On the Continuity of Rotation Representations in Neural Networks
    """
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_targ)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    # Clamp to prevent NaN gradients (as we discussed!)
    cos_ang = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_ang).mean()


class InstanceBoxLoss(nn.Module):
    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.w_c = config.center_loss_weight
        self.w_s = config.dim_loss_weight
        self.w_r = config.rot_loss_weight
        self.w_corner = config.corner_loss_weight
        self.rotation_loss_type = config.rotation_loss_type

    def forward(self, targ_c: Tensor, targ_s: Tensor, 
                targ_rot6d: Tensor, targ_corners: Tensor,
                pred_c: Tensor, pred_s: Tensor = None, 
                pred_rot6d: Tensor = None,) -> Dict[str, int]:
        # Huber Loss for Center and Dimensions
        loss_c = F.huber_loss(pred_c, targ_c, reduction='mean')
        if pred_s is not None:
            loss_s = F.huber_loss(pred_s, targ_s, reduction='mean')
        else:
            loss_s = torch.zeros_like(loss_c)
        
        # Rotation and corner losses
        if pred_s is not None and pred_rot6d is not None:
            pred_corners, R_pred = reconstruct_unique_box(pred_c, pred_s, pred_rot6d, output_rot_mat=True)
            targ_corners_reconstrcted, R_targ = reconstruct_unique_box(targ_c, targ_s, targ_rot6d, output_rot_mat=True)
            assert (targ_corners - targ_corners_reconstrcted).sum() < 1e-3
            
            if self.rotation_loss_type.value == 1:
                # Based on Zhou et. al. 2019
                loss_r = geodesic_loss(R_pred, R_targ)
            if self.rotation_loss_type.value == 2:
                # values are small thus no need for Huber
                loss_r = F.mse_loss(R_pred, R_targ, reduce="mean") 
            if self.rotation_loss_type.value == 3:
                # values are small thus no need for Huber
                loss_r = F.mse_loss(pred_rot6d, targ_rot6d, reduction='mean') 
                
            # Corner loss based on Frustum-Pointnet
            loss_corner = F.huber_loss(pred_corners, targ_corners, reduction='mean')
        else:
            loss_r = torch.zeros_like(loss_c)
            loss_corner = torch.zeros_like(loss_c)
            
        total_loss = (self.w_c * loss_c) + (self.w_s * loss_s) + \
                     (self.w_r * loss_r) + (self.w_corner * loss_corner)
                     
        return total_loss, {
            "loss_center": loss_c.item(), "loss_dims": loss_s.item(), 
            "loss_rot": loss_r.item(), "loss_corner": loss_corner.item()
        }