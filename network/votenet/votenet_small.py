# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Single-object 3D bounding box regressor built on a PointNet++ backbone.

    Removed from original VoteNet:
      - VotingModule   : unnecessary when point cloud is a pre-cropped single instance
      - ProposalModule : replaced by a lightweight BboxRegressionHead
      - num_class, num_heading_bin, num_size_cluster : not needed for 1-object regression

    Output per sample: center (3) + log-size (3) + 6D rotation (6) = 12 values,
    decoded into end_points keys: 'center', 'size', 'rot_6d', 'rot_mat'.

Author: Charles R. Qi and Or Litany (original)
Modified for single-instance bbox regression.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import NetConfig
    
import torch
import torch.nn as nn
import numpy as np

from network.votenet.backbone_small_module import Pointnet2Backbone
from network.votenet.proposal_small_module import BboxRegressionHead


class VoteNet(nn.Module):
    """
    PointNet++ backbone + direct 12-DoF bounding box regression head.

    Designed for pre-cropped single-object point clouds with small datasets
    (~2k samples). No voting, no multi-proposal clustering, no NMS.

    Parameters
    ----------
    input_feature_dim : int, default 0
        Extra per-point feature channels beyond xyz.
        E.g. for an (N, 6) cloud (xyz + RGB) set this to 3.
    dropout : float, default 0.4
        Dropout probability applied in the regression head FC layers.
        Higher values help regularise small datasets.
    """

    def __init__(self, config: NetConfig):
        super().__init__()

        self.input_feature_dim = config.input_feature_dim

        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Direct regression head → 12 box params
        self.bbox_head = BboxRegressionHead(feat_dim=config.proposal_hid_dim, 
                                            dropout=config.dropout)

    def forward(self, inputs):
        """
        Args:
            inputs : dict with key 'point_clouds'
                point_clouds : (B, N, 3 + input_feature_dim)

        Returns:
            end_points : dict
                'center'   (B, 3)     absolute box center
                'size'     (B, 3)     box dimensions (w, h, l), always positive
                'rot_6d'   (B, 6)     raw 6D rotation (for loss computation)
                'rot_mat'  (B, 3, 3)  rotation matrix decoded via Gram-Schmidt
        """
        end_points = {}

        end_points = self.backbone_net(inputs['point_clouds'], end_points)

        features = end_points['sa3_features']   
        xyz      = end_points['sa3_xyz']   

        end_points = self.bbox_head(features, xyz, end_points)

        return end_points


if __name__ == '__main__':
    B = 8
    inputs = {'point_clouds': torch.rand((B, 2048, 3)).cuda()}
    model = VoteNet(input_feature_dim=0, dropout=0.4).cuda()
    end_points = model(inputs)
    print('center  ', end_points['center'].shape)   
    print('size    ', end_points['size'].shape)     
    print('rot_6d  ', end_points['rot_6d'].shape)  
    print('rot_mat ', end_points['rot_mat'].shape) 