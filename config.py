from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Paths:
    data: str = "/mnt/c/d/3d_bb_data/raw"
    parsed_data: str = "/mnt/c/d/3d_bb_data/processed"
    ckpts: str = "/mnt/c/d/3d_bb_ckpts_n_logs/ckpts"
    logs: str = "/mnt/c/d/3d_bb_ckpts_n_logs/logs"

@dataclass
class DataLoaderConfig:
    max_number_pc_pts: int = 2048
    batch_size: int = 32
    num_workers: int = 8
    apply_aug: bool = True
    shuffle: bool = True
    persistent_workers: bool = True

@dataclass
class NetConfig:
    input_feature_dim : int = 3 # for RGB
    dropout: float = 0.4
    seed_feature_dim: int = 256
    voting_factor: int = 1
    num_proposal: int = 1
    num_proposal_seeds: int = 32

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 50
    center_loss_weight: float = 1.0
    dim_loss_weight: float = 0.0
    rot_loss_weight: float = 0.0
    corner_loss_weight: float = 0.0