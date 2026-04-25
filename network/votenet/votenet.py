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

from network.votenet.backbone_module import Pointnet2Backbone
from network.votenet.proposal_module import ProposalModule
from network.votenet.voting_module import VotingModule


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
        
        self.vgen = VotingModule(config.voting_factor, 
                            seed_feature_dim=config.seed_feature_dim)

        self.pnet = ProposalModule(num_proposal=config.num_proposal, 
                            seed_feat_dim=config.seed_feature_dim)

    def forward(self, pc_pts):
        """
        Args:
            inputs : 
                point_clouds : (B, N, 3 + input_feature_dim)

        Returns:
            end_points : dict
                'center'   (B, 3)     absolute box center
                'size'     (B, 3)     box dimensions (w, h, l), always positive
                'rot_6d'   (B, 6)     raw 6D rotation (for loss computation)
                'rot_mat'  (B, 3, 3)  rotation matrix decoded via Gram-Schmidt
        """
        end_points = {}

        end_points = self.backbone_net(pc_pts, end_points)

        xyz = end_points['sa3_xyz']
        features = end_points['sa3_features']
        end_points['seed_inds'] = end_points['sa3_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)

        return end_points