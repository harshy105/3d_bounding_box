import torch
import torch.nn as nn
import torch.nn.functional as F

from network.votenet.pointnet2_modules import PointnetSAModuleVotes

def decode_single_scores(net, end_points):
    """
    Decodes the 12-channel output into physical bounding box parameters.
    """    
    # 1. Center Offset (Channels 0:3)
    center = end_points["centroid"] + net[:, 0:3] 
    end_points['center'] = center

    # 2. Dimensions (Channels 3:6)
    end_points["size"] = torch.exp(net[:, 3:6]) # always +ve

    # 3. 6D Rotation (Channels 6:12)
    end_points["rot_6d"] = net[:, 6:12]
    
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_proposal, seed_feat_dim, 
                 num_votes, use_pointnet_agg,
                 dropout):
        super().__init__() 
        self.num_proposal = num_proposal
        self.use_pointnet_agg = use_pointnet_agg
        
        # --------- VOTE CLUSTERING ---------
        if use_pointnet_agg:
            self.vote_aggregation = PointnetSAModuleVotes( 
                    npoint=num_proposal,  
                    radius=0.6,  # Huge radius to capture the whole instance
                    nsample=num_votes,  # Grab ALL votes
                    mlp=[seed_feat_dim, 64, 128, 256],
                    use_xyz=True,
                    normalize_xyz=True
                )

        self.conv1 = nn.Conv1d(seed_feat_dim, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

        self.fc_out  = nn.Linear(256, 12)

    def forward(self, xyz, features, end_points):
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net)))         

        if self.use_pointnet_agg:
            xyz, net, _ = self.vote_aggregation(xyz, net)
            xyz = xyz[:, 0, :]
            net = net[:, :, 0]
        else:
            xyz = xyz.mean(dim=1)
            net = net.max(dim=2)[0]
        
        end_points["centroid"] = xyz

        # Regression Head
        net = self.drop1(F.relu(self.bn_fc1(self.fc1(net))))  
        net = self.drop2(F.relu(self.bn_fc2(self.fc2(net))))  
        net = self.fc_out(net)  

        assert self.num_proposal == 1, "Decoding only based on one bounding box"
        end_points = decode_single_scores(net, end_points)
        
        return end_points