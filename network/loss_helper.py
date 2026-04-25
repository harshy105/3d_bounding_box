from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple
from torch import Tensor
if TYPE_CHECKING:
    from config import TrainConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.utils import reconstruct_unique_box


def _geodesic_loss(R_pred: Tensor, R_targ: Tensor) -> Tensor:
    """
    Geodesic (angular) distance between two batches of rotation matrices.

    L = mean( arccos( (trace(R_pred^T R_targ) - 1) / 2 ) )

    This is the true SO(3) distance — invariant to representation choice
    and well-behaved near the identity (unlike MSE on 6D vectors).

    Args:
        R_pred : (B, 3, 3)
        R_targ : (B, 3, 3)

    Returns:
        scalar loss (mean angle in radians across the batch)
    """
    # Relative rotation: if perfect, R_rel = I
    R_rel   = torch.bmm(R_pred.transpose(1, 2), R_targ)        # (B, 3, 3)

    # trace ∈ [-1, 3];  (trace - 1) / 2 ∈ [-1, 1]
    trace   = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_ang = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    # arccos → angle in [0, π]
    return torch.acos(cos_ang).mean()


def _rotation_regularization(R_pred: Tensor) -> Tensor:
    """
    Penalise deviation of R_pred from SO(3): || R^T R - I ||_F

    Keeps the predicted rotation matrix well-conditioned during early
    training before Gram-Schmidt has stabilised the gradients.

    Args:
        R_pred : (B, 3, 3)

    Returns:
        scalar regularization loss
    """
    B   = R_pred.shape[0]
    I   = torch.eye(3, device=R_pred.device).unsqueeze(0).expand(B, -1, -1)
    RtR = torch.bmm(R_pred.transpose(1, 2), R_pred)
    return F.mse_loss(RtR, I)


class InstanceBoxLoss(nn.Module):
    """
    Combined loss for single-object 3D bounding box regression.

    Component breakdown
    -------------------
    loss_center : Huber loss on the *residual* between predicted center and
                  the point-cloud centroid.  Supervising the residual (rather
                  than the absolute coordinate) removes scene-scale variance
                  and makes the target ~zero-mean, which is much easier to
                  regress.

    loss_dims   : Huber loss in log-size space.  Both prediction and target
                  are log-transformed before comparison so the loss is scale-
                  invariant (a 10 % error on a small dimension costs the same
                  as a 10 % error on a large one).

    loss_rot    : Geodesic (angular) distance in SO(3) — the true rotation
                  metric.  Unlike MSE on 6D vectors, this is invariant to the
                  specific parameterisation and gives meaningful gradients
                  even far from the target.

    loss_reg    : Orthogonality regularisation || R^T R - I ||_F.  Stabilises
                  early training before the rotation head has converged.

    loss_corner : Huber loss on all 8 box corners.  Acts as a coupling term
                  that jointly penalises misaligned center + rotation, and
                  provides a geometry-aware gradient that the individual
                  component losses cannot.

    Config keys expected
    --------------------
    center_loss_weight, dim_loss_weight, rot_loss_weight,
    corner_loss_weight, rot_reg_weight  (new — suggest 0.1)
    """

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.w_c      = config.center_loss_weight
        self.w_s      = config.dim_loss_weight
        self.w_r      = config.rot_loss_weight
        self.w_corner = config.corner_loss_weight
        self.w_reg    = getattr(config, 'rot_reg_weight', 0.1)

    def forward(
        self,
        pred_c:      Tensor,   # (B, 3)  predicted center
        pred_s:      Tensor,   # (B, 3)  predicted size  (linear, not log)
        pred_rot6d:  Tensor,   # (B, 6)  predicted 6D rotation
        targ_c:      Tensor,   # (B, 3)  GT center
        targ_s:      Tensor,   # (B, 3)  GT size         (linear, not log)
        targ_rot6d:  Tensor,   # (B, 6)  GT 6D rotation
        targ_corners: Tensor,  # (B, 8, 3) GT corners
        pc_centroid: Tensor,   # (B, 3)  mean of input point cloud  ← new arg
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Args:
            pc_centroid : mean xyz of the input point cloud for this sample.
                          Compute once before the forward pass:
                              pc_centroid = point_clouds[..., :3].mean(dim=1)
                          and pass it through so the center loss supervises
                          the residual offset rather than the raw coordinate.
        """

        # ------------------------------------------------------------------ #
        # 1. Center — supervise the residual from the point-cloud centroid    #
        #    Both pred and target offsets are small numbers near zero,        #
        #    removing scene-scale from the loss.                              #
        # ------------------------------------------------------------------ #
        pred_c_res = pred_c  - pc_centroid                     # (B, 3)
        targ_c_res = targ_c  - pc_centroid                     # (B, 3)
        loss_c = F.huber_loss(pred_c_res, targ_c_res, reduction='mean')

        # ------------------------------------------------------------------ #
        # 2. Dimensions — log-space so the loss is scale-invariant           #
        # ------------------------------------------------------------------ #
        loss_s = F.huber_loss(
            torch.log(pred_s.clamp(min=1e-6)),
            torch.log(targ_s.clamp(min=1e-6)),
            reduction='mean',
        )

        # ------------------------------------------------------------------ #
        # 3. Rotation — geodesic distance on SO(3)                           #
        # ------------------------------------------------------------------ #
        # output_rot_mat=True reuses the Gram-Schmidt already computed for   #
        # the corners — no duplicate work, single source of truth.           #
        pred_corners, R_pred = reconstruct_unique_box(pred_c, pred_s, pred_rot6d, output_rot_mat=True)
        _,            R_targ = reconstruct_unique_box(targ_c, targ_s, targ_rot6d, output_rot_mat=True)
        loss_r   = _geodesic_loss(R_pred, R_targ)

        # ------------------------------------------------------------------ #
        # 4. Orthogonality regularisation                                    #
        # ------------------------------------------------------------------ #
        loss_reg = _rotation_regularization(R_pred)

        # ------------------------------------------------------------------ #
        # 5. Corner loss — geometry-aware coupling term                      #
        # ------------------------------------------------------------------ #
        loss_corner = F.huber_loss(pred_corners, targ_corners, reduction='mean')

        # ------------------------------------------------------------------ #
        # Total                                                               #
        # ------------------------------------------------------------------ #
        total = (
            self.w_c      * loss_c      +
            self.w_s      * loss_s      +
            self.w_r      * loss_r      +
            self.w_reg    * loss_reg    +
            self.w_corner * loss_corner
        )

        return total, {
            "loss_center": loss_c.item(),
            "loss_dims":   loss_s.item(),
            "loss_rot":    loss_r.item(),
            "loss_rot_reg": loss_reg.item(),
            "loss_corner": loss_corner.item(),
        }