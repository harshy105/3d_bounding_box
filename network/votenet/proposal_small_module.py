# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" BboxRegressionHead: direct single-object 3D bbox regression from point features.

    Replaces the original ProposalModule entirely. No FPS clustering, no vote
    aggregation, no objectness/semantic heads. Works directly on the (B, C, K)
    feature tensor from the PointNet++ backbone.

    6D rotation follows Zhou et al. "On the Continuity of Rotation
    Representations in Neural Networks" — decoded via Gram-Schmidt into a
    proper SO(3) rotation matrix with no gimbal lock or discontinuities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.utils import reconstruct_unique_box

def _decode(net: torch.Tensor, xyz: torch.Tensor, end_points: dict) -> dict:
    """
    Decode raw 12-dim MLP output into named bbox quantities.

    Args:
        net        : (B, 12)
        xyz        : (B, K, 3)  backbone subsampled points, used as center anchor
        end_points : dict

    Channel layout:
        0:3  — center residual  (added to point-cloud centroid)
        3:6  — log-size         (exp → w, h, l > 0)
        6:12 — 6D rotation      (Gram-Schmidt → SO(3))

    Returns:
        end_points updated with 'center', 'size', 'rot_6d', 'rot_mat'
    """
    centroid = xyz.mean(dim=1)                    

    center  = centroid + net[:, 0:3]                
    size    = torch.exp(net[:, 3:6])                
    rot_6d  = net[:, 6:12]                           

    end_points['center']  = center
    end_points['size']    = size
    end_points['rot_6d']  = rot_6d

    box, rot_matrix = reconstruct_unique_box(center, size, rot_6d, output_rot_mat=True) 
    end_points['box']     = box

    end_points['rot_mat'] = rot_matrix

    return end_points


class BboxRegressionHead(nn.Module):
    """
    Lightweight bbox regression head for single-object, pre-cropped point clouds.

    Parameters
    ----------
    feat_dim : int
        Channel dimension of incoming features (matches backbone output = 256).
    dropout : float
        Dropout rate on FC layers. Use 0.4–0.5 for small datasets (~2k samples).
    """

    def __init__(self, feat_dim: int = 256, dropout: float = 0.4):
        super().__init__()

        self.conv1 = nn.Conv1d(feat_dim, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.bn1   = nn.BatchNorm1d(256)
        self.bn2   = nn.BatchNorm1d(256)

        self.fc1     = nn.Linear(256, 512)
        self.fc2     = nn.Linear(512, 256)
        self.bn_fc1  = nn.BatchNorm1d(512)
        self.bn_fc2  = nn.BatchNorm1d(256)
        self.drop1   = nn.Dropout(p=dropout)
        self.drop2   = nn.Dropout(p=dropout)

        self.fc_out  = nn.Linear(256, 12)

    def forward(self, features: torch.Tensor, xyz: torch.Tensor, end_points: dict) -> dict:
        """
        Args:
            features   : (B, feat_dim, K)  per-point features from backbone
            xyz        : (B, K, 3)         corresponding point positions
            end_points : dict

        Returns:
            end_points with 'center' (B,3), 'size' (B,3),
                             'rot_6d' (B,6), 'rot_mat' (B,3,3)
        """
        # -- Per-point refinement --------------------------------------------
        x = F.relu(self.bn1(self.conv1(features))) 
        x = F.relu(self.bn2(self.conv2(x)))         

        # -- Global max-pool -------------------------------------------------
        x = x.max(dim=2)[0]                       

        # -- FC regression ---------------------------------------------------
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))  
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))  
        x = self.fc_out(x)                                

        # -- Decode ----------------------------------------------------------
        end_points = _decode(x, xyz, end_points)
        return end_points


if __name__ == '__main__':
    B, K = 8, 32
    features   = torch.rand(B, 256, K).cuda()
    xyz        = torch.rand(B, K, 3).cuda()
    end_points = {}

    head = BboxRegressionHead(feat_dim=256, dropout=0.4).cuda()
    out  = head(features, xyz, end_points)

    print('center  ', out['center'].shape)   
    print('size    ', out['size'].shape)     
    print('rot_6d  ', out['rot_6d'].shape)  
    print('rot_mat ', out['rot_mat'].shape) 
    print('box     ', out['box'].shape)     

    I   = torch.bmm(out['rot_mat'].transpose(1, 2), out['rot_mat'])
    eye = torch.eye(3).cuda().unsqueeze(0).expand(B, -1, -1)
    print('R^T R ≈ I:', torch.allclose(I, eye, atol=1e-5))