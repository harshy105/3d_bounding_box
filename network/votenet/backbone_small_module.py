# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Pointnet2Backbone — slimmed for single-object, ~2k instance dataset.

    Changes vs. original:
      - SA layers:    4 → 3  (SA4 removed — redundant for pre-cropped objects)
      - npoint:       2048/1024/512/256 → 512/128/32
                      Object is already isolated; dense sampling wastes compute
                      and adds parameters that overfit on small datasets.
      - MLP widths:   narrowed ~50% throughout to reduce parameter count
      - FP layers:    both removed — feature propagation back to all points is
                      only needed for per-point tasks (voting, segmentation).
                      BboxRegressionHead global-pools the SA3 output directly,
                      so FP adds zero useful signal and significant parameters.
      - Output key:   'sa3_features'/'sa3_xyz' 

    Parameter count (approximate):
        Original backbone : ~3.1 M
        This backbone     : ~0.5 M   ← much better fit for 2k samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from network.votenet.pointnet2_modules import PointnetSAModuleVotes


class Pointnet2Backbone(nn.Module):
    """
    3-layer PointNet++ backbone for pre-cropped single-object point clouds.

    Parameters
    ----------
    input_feature_dim : int, default 0
        Extra per-point channels beyond xyz (e.g. 3 for RGB or normals).
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.2,
            nsample=32,  
            mlp=[input_feature_dim, 32, 32, 64], 
            use_xyz=True,
            normalize_xyz=True,
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=128,
            radius=0.4,
            nsample=32,
            mlp=[64, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True,
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=32,
            radius=0.8,
            nsample=16,
            mlp=[128, 128, 128, 256], 
            use_xyz=True,
            normalize_xyz=True,
        )


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        """
        Args:
            pointcloud : (B, N, 3 + input_feature_dim)

        Returns:
            end_points with keys:
                'sa1_xyz'      (B, 512, 3)
                'sa1_features' (B, 64,  512)
                'sa2_xyz'      (B, 128, 3)
                'sa2_features' (B, 128, 128)
                'sa3_xyz'      (B, 32,  3)    ← output for BboxRegressionHead
                'sa3_features' (B, 256, 32)   ← output for BboxRegressionHead
        """
        if not end_points:
            end_points = {}

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 3 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds']     = fps_inds
        end_points['sa1_xyz']      = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features)
        end_points['sa2_inds']     = fps_inds
        end_points['sa2_xyz']      = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features)
        end_points['sa3_inds']     = fps_inds
        end_points['sa3_xyz']      = xyz
        end_points['sa3_features'] = features

        return end_points


if __name__ == '__main__':
    backbone = Pointnet2Backbone(input_feature_dim=0).cuda()
    backbone.eval()

    # Count parameters
    n_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    out = backbone(torch.rand(8, 2048, 3).cuda())
    for key in sorted(out.keys()):
        print(f'{key:20s}  {tuple(out[key].shape)}')