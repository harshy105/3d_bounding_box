import torch
import torch.nn as nn
import torch.nn.functional as F

from network.votenet.pointnet2_modules import PointnetSAModuleVotes

def decode_single_scores(net, end_points):
    """
    Decodes the 12-channel output into physical bounding box parameters.
    """
    base_xyz = end_points['aggregated_vote_xyz'][:, 0, :] # (B, 3)
    
    # 1. Center Offset (Channels 0:3)
    center = base_xyz + net[:, 0:3, 0] 
    end_points['center'] = center

    # 2. Dimensions (Channels 3:6)
    end_points["size"] = net[:, 3:6, 0]

    # 3. 6D Rotation (Channels 6:12)
    end_points["rot_6d"] = net[:, 6:12, 0]
    
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_proposal, seed_feat_dim, num_votes):
        super().__init__() 
        self.num_proposal = num_proposal
        
        # --------- VOTE CLUSTERING ---------
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=num_proposal,  
                radius=1.0,  # Huge radius to capture the whole instance
                nsample=num_votes,  # Grab ALL votes
                mlp=[seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # --------- PROPOSAL GENERATION ---------
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        
        # Output 12 channels (3 center + 3 size + 6 rotation)
        self.conv3 = nn.Conv1d(128, 12, 1) 
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, xyz, features, end_points):
        xyz, features, fps_inds = self.vote_aggregation(xyz, features)
        # sample_inds = fps_inds
        
        end_points['aggregated_vote_xyz'] = xyz 
        # end_points['aggregated_vote_inds'] = sample_inds 

        # Regression Head
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # Shape: (B, 12, 1)

        assert self.num_proposal == 1, "Decoding only based on one bounding box"
        end_points = decode_single_scores(net, end_points)
        
        return end_points