from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from config import DataLoaderConfig

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import lmdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm 
import os

from utilities.utils import (extract_3d_bbox_params, augment_instance, 
                             reconstruct_unique_box, reorder_original_box)
from utilities.plotting import plot_instance
from unit_test.box_preprocessing_test import (have_identical_corner_sets, 
                                              are_corners_close)

class LMDBInstanceDataset(Dataset):
    def __init__(self, lmdb_path: str, data_loader_config: DataLoaderConfig, 
                 apply_aug: bool = False, vis_sample: bool = False) -> None:
        self.lmdb_path = lmdb_path
        self.apply_aug = apply_aug
        self.num_points = data_loader_config.max_number_pc_pts
        self.env = None
        self.vis_sample = vis_sample
        
        # We open the environment once to get the keys, then close it.
        # This helps prevent multiprocessing issues with PyTorch workers.
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]
        env.close()

    def _init_env(self) -> None:
        # Lazy-initialize the LMDB environment for each DataLoader worker.
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Fetches and processes a single sample from the dataset."""
        self._init_env()
        
        with self.env.begin() as txn:
            byteflow = txn.get(self.keys[idx])
            sample = pickle.loads(byteflow)
            
        return self.process_sample(idx, sample)
            
    def process_sample(self, idx: int, sample: Dict[str, np.ndarray]) -> Dict[str, Tensor]:
        pc_pts = sample["pc_pts"] # shape (N, 6) position + RGB
        bbox_3d = sample["bbox_3d"]
        
        if self.apply_aug:
            pc_pts, bbox_3d = augment_instance(pc_pts, bbox_3d)
            
        # Subsample or zero-pad to ensure all point clouds have the same number of points for batching.
        num_current_pts = pc_pts.shape[0]
        if num_current_pts >= self.num_points:
            choice = np.random.choice(num_current_pts, self.num_points, replace=False)
            pc_pts = pc_pts[choice, :]
        else:
            pad_size = self.num_points - num_current_pts
            zero_padding = np.zeros((pad_size, *pc_pts.shape[1:]), dtype=pc_pts.dtype)
            pc_pts = np.concatenate((pc_pts, zero_padding), axis=0)
        
        pc_tensor = torch.from_numpy(pc_pts).float()
        
        # Convert the bounding box to a canonical representation (center, dims, 6D rotation)
        # This helps the network learn in a stable, unambiguous way.
        bbox_tensor = torch.from_numpy(bbox_3d).float()
        bbox_center, bbox_dims, bbox_rot_6d = extract_3d_bbox_params(bbox_tensor)
        
        # Reconstruct the box from our canonical parameters to get a consistent corner ordering.
        reconstructed_box = reconstruct_unique_box(bbox_center, bbox_dims, bbox_rot_6d)
        
        # Reorder the original box corners to match the reconstructed box's order.
        reordered_bbox_tensor = reorder_original_box(bbox_tensor, reconstructed_box)
        
        assert have_identical_corner_sets(reordered_bbox_tensor, bbox_tensor), \
            f"Reordered box is not same as original box in Sample {self.keys[idx]}"
            
        assert are_corners_close(reordered_bbox_tensor, reconstructed_box, atol=1e-3), \
            f"Reconstruction of the box is not 1-to-1 map Sample {self.keys[idx]}"
        
        sample_processed = {
            "pc_pts": pc_tensor,
            "bbox_3d": reordered_bbox_tensor,
            "bbox_center": bbox_center,
            "bbox_dims": bbox_dims,
            "bbox_rot_6d": bbox_rot_6d,
            "key": self.keys[idx].decode('ascii')
        }
        
        # Optional visualization for debugging.
        if self.vis_sample:
            self.visualize_sample(sample_processed, reordered_bbox_tensor)
        
        return sample_processed

    def visualize_sample(self, sample_processed: Dict[str, Tensor], reconstructed_box: Optional[Tensor] = None) -> None:
        pc_tensor = sample_processed["pc_pts"]

        fig = plt.figure(figsize=(8, 8))
        ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
        
        if reconstructed_box is not None:
            plot_instance(pc_pts=pc_tensor.numpy(), bbox_3d=reconstructed_box.numpy(), ax=ax_3d)
            ax_3d.set_title("3D Point Cloud (RGB) & Reconstructed Box")
        else:
            original_box = sample_processed["bbox_3d"]
            plot_instance(pc_pts=pc_tensor.numpy(), bbox_3d=original_box.numpy(), ax=ax_3d)
            ax_3d.set_title("3D Point Cloud (RGB) & Original Box")
        
        plt.tight_layout()
        plt.show()
        plt.close("all")


class InstanceDataModule(LightningDataModule):
    def __init__(self, parsed_data_path: str, data_loader_config: DataLoaderConfig) -> None:
        super().__init__()
        self.lmdb_path = parsed_data_path
        self.batch_size = data_loader_config.batch_size
        self.num_workers = data_loader_config.num_workers
        self.apply_aug = data_loader_config.apply_aug
        self.shuffle = data_loader_config.shuffle
        self.use_persistent =  data_loader_config.persistent_workers if self.num_workers > 0 else False
        self.config = data_loader_config

    def setup(self, stage=None) -> None:
        self.train_dataset = LMDBInstanceDataset(os.path.join(self.lmdb_path, "train"), 
                                        data_loader_config=self.config, apply_aug=self.apply_aug)
        self.val_dataset = LMDBInstanceDataset(os.path.join(self.lmdb_path, "val"), 
                                        data_loader_config=self.config, apply_aug=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers,
            drop_last=True, # for Batchnorm issues 
            persistent_workers=self.use_persistent
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=self.use_persistent
        )

if __name__ == "__main__":
    # A simple test to check the dataset and visualization logic.
    from config import Paths
    from config import DataLoaderConfig
    split = "val"
    dataset = LMDBInstanceDataset(os.path.join(Paths.parsed_data, split), data_loader_config=DataLoaderConfig(),
                                  apply_aug=True, vis_sample=True)
    
    # Make sure the dataset isn't empty before we start iterating.
    if len(dataset) == 0:
        print("Dataset is empty. Please check the LMDB path.")
    else:
        # Loop through every sample in the dataset for debugging and reconstruction verification
        for sample in tqdm(dataset):
            pass