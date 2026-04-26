# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" 
This is a slimmed-down version of the PointNet++ backbone, optimized for a smaller, 
single-object dataset. The original architecture was overkill, so I've made a few tweaks:

- Reduced from 4 to 3 Set Abstraction (SA) layers since we're working with pre-cropped objects.
- Lowered the number of points sampled and Narrowed the MLPs.

This brings the parameter count down to ~0.5M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from network.votenet.pointnet2_modules import PointnetSAModuleVotes
from network.votenet.pointnet2_modules import PointnetFPModule


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim, num_seeds):
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
            npoint=num_seeds,
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
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # Pass through three Set Abstraction layers
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
    
if __name__=='__main__':
    backbone_net = Pointnet2Backbone(
        input_feature_dim=3,
        num_seeds=32,
        ).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)