from dataclasses import dataclass
from typing import Tuple

@dataclass
class Paths:
    data = "/mnt/c/d/3d_bb_data/raw"
    parsed_data = "/mnt/c/d/3d_bb_data/processed"
    
@dataclass
class DataPreprocessingConfig:
    crop_img_size: Tuple = (64, 64)
    
@dataclass
class DataLoaderConfig:
    max_number_pc_pts: int = int(4e4)
    batch_size: int = 32
    num_workers: int = 8
    apply_aug: bool = True
    shuffle: bool = True
    persistent_workers: bool = True