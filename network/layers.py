import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class PointNetBackbone(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(out_dim)

    def forward(self, p: Tensor) -> Tensor:
        x = p.transpose(1, 2) # p: (B, N, 3) -> requires (B, 3, N) for Conv1D
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        F_feat = F.relu(self.bn3(self.conv3(x)))
        return F_feat.transpose(1, 2) # Output: (B, N, D)

class VotingModule(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 3) # Outputs 3D offsets (dx, dy, dz)
        )

    def forward(self, F_feat: Tensor) -> Tensor:
        # F_feat: (B, N, D)
        B, N, D = F_feat.shape
        x = F_feat.reshape(B * N, D)
        offsets = self.mlp(x)
        return offsets.reshape(B, N, 3)

class BBoxRegressionModule(nn.Module):
    def __init__(self, global_feat_dim: int) -> None:
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

    def forward(self, f_global: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.mlp(f_global)
        delta_s = out[:, 0:3]
        rot_6d = out[:, 3:9]
        return delta_s, rot_6d