import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetBackbone(nn.Module):
    """
    Placeholder for PointNet++. 
    Extracts pointwise features F \in R^{N x D} from P \in R^{N x 3}.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, p):
        # p: (B, N, 3) -> requires (B, 3, N) for Conv1D
        x = p.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        F_feat = F.relu(self.bn3(self.conv3(x)))
        # Output: (B, N, D)
        return F_feat.transpose(1, 2)

class VotingModule(nn.Module):
    """ MLP_vote to generate offset vectors \Delta p_i """
    def __init__(self, feature_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 3) # Outputs 3D offsets (\Delta x, \Delta y, \Delta z)
        )

    def forward(self, F_feat):
        # F_feat: (B, N, D)
        B, N, D = F_feat.shape
        x = F_feat.reshape(B * N, D)
        offsets = self.mlp(x)
        return offsets.reshape(B, N, 3)

class BBoxRegressionModule(nn.Module):
    """ MLP_box to predict \Delta s and q_raw (6D Rotation) """
    def __init__(self, global_feat_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_feat_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 9) # 3 for \Delta s, 6 for 6D rotation
        )

    def forward(self, f_global):
        out = self.mlp(f_global)
        delta_s = out[:, 0:3]
        rot_6d = out[:, 3:9]
        return delta_s, rot_6d