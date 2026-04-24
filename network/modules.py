import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from utilities.utils import apply_weights
from network.layers import PointNetBackbone, VotingModule, BBoxRegressionModule
from config import NetConfig

class InstanceVoteNet(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.template_dims = torch.tensor(config.template_dims, dtype=torch.float32)
        
        # 1. Feature Extraction
        self.backbone = PointNetBackbone(out_dim=config.point_feature_dim)
        self.vote_mlp = VotingModule(feature_dim=config.point_feature_dim)
        
        # 2. Aggregation MLP_feat
        self.feat_mlp = nn.Sequential(
            nn.Linear(config.point_feature_dim, config.global_feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.global_feature_dim)
        )
        
        # 3. Regression
        self.box_mlp = BBoxRegressionModule(global_feat_dim=config.global_feature_dim)
        self.apply(apply_weights)

    def forward(self, P: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, N, _ = P.shape # (B, N, 3)
        
        # Stage 1: Point Features & Voting
        F_feat = self.backbone(P)  # (B, N, D)
        delta_P = self.vote_mlp(F_feat) # Delta centers (B, N, 3)
        c_i = P + delta_P  # Centers based on each point (B, N, 3)
        
        # Stage 2: Aggregation
        c_pred = torch.mean(c_i, dim=1) # Consensus center (B, 3)
        
        F_trans = F_feat.reshape(B * N, -1) # (B, N, D)
        F_agg = self.feat_mlp(F_trans).reshape(B, N, -1) # (B, N, D_global)
        f_global, _ = torch.max(F_agg, dim=1) # Maxpool (B, D_global)
        
        # Stage 3: Bounding Box Regression
        delta_s, rot_6d = self.box_mlp(f_global)
        
        # Add delta_s to template dimensions
        template = self.template_dims.to(delta_s.device)
        s_pred = template + delta_s # Final dimensions
        
        return c_pred, s_pred, rot_6d