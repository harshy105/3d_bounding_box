# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from network.votenet.pointnet2_modules import PointnetSAModuleVotes

def decode_single_scores(net, end_points):
    base_xyz = end_points['aggregated_vote_xyz'][:, 0, :] # (batch_size, 3)
    center = base_xyz + net[:, :, 0] # (batch_size, 3)
    end_points['center'] = center

    end_points["size"] = None
    end_points["rot_6d"] = None
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_proposal, seed_feat_dim):
        super().__init__() 
        self.num_proposal = num_proposal

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.conv3 = torch.nn.Conv1d(128,3,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)
        sample_inds = fps_inds
        
        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 3, num_proposal)

        assert self.num_proposal == 1, "Decoding only based on one bounding box"
        end_points = decode_single_scores(net, end_points)
        return end_points

if __name__ == "__main__":
    xyz = torch.zeros(8, 32, 3).float().cuda()
    features = torch.zeros(8, 256, 32).float().cuda()
    pnet = ProposalModule(1, 256).cuda()
    pnet.eval()
    end_points = {}
    end_points = pnet(xyz, features, end_points)
    print(end_points["center"].shape)