from dataclasses import dataclass
from typing import Tuple

@dataclass
class Paths:
    data = "/mnt/c/d/3d_bb_data/raw"
    parsed_data = "/mnt/c/d/3d_bb_data/processed"
    
@dataclass
class data_preprocessing:
    crop_img_size: Tuple = (64, 64)
    
@dataclass
class data_loader_config:
    max_number_pc_pts: int = int(4e4)