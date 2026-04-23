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
from utilities.plotting import draw_bboxes_on_image, plot_instance
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
        
        # Open environment once just to extract the keys, then close it. 
        # This prevents multiprocessing crashes with PyTorch workers.
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]
        env.close()

    def _init_env(self) -> None:
        # Lazy initialization for DataLoader workers
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict:
        self._init_env()
        
        with self.env.begin() as txn:
            byteflow = txn.get(self.keys[idx])
            sample = pickle.loads(byteflow)
            
        return self.process_sample(idx, sample)
            
    def process_sample(self, idx: int, sample: Dict) -> Dict:            
        pc_pts = sample["pc_pts"]
        bbox_3d = sample["bbox_3d"]
        img_crop = sample["img_crop"]
        
        # 1. Apply data augmentation
        if self.apply_aug:
            pc_pts, bbox_3d, img_crop = augment_instance(pc_pts, bbox_3d, img_crop)
            
        # 2. Ensure uniform point cloud size for batching (Subsample or Zero-Pad)
        num_current_pts = pc_pts.shape[0]
        if num_current_pts >= self.num_points:
            choice = np.random.choice(num_current_pts, self.num_points, replace=False)
            pc_pts = pc_pts[choice, :]
        else:
            pad_size = self.num_points - num_current_pts
            zero_padding = np.zeros((pad_size, *pc_pts.shape[1:]), dtype=pc_pts.dtype)
            pc_pts = np.concatenate((pc_pts, zero_padding), axis=0)
        
        pc_tensor = torch.from_numpy(pc_pts).float()

        # 3. Convert image to Torch and permute to (Channels, Height, Width) format
        img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
        
        # 4. Extract canonical features from the 3D bounding box and reorder the bounding
        #  box corner sequence so that the neural network can learn in a stable manner
        bbox_tensor = torch.from_numpy(bbox_3d).float()
        bbox_center, bbox_dims, bbox_rot_6d = extract_3d_bbox_params(bbox_tensor)
        # Generate a unique bounding box given the bouning box parameters
        reconstructed_box = reconstruct_unique_box(bbox_center, bbox_dims, bbox_rot_6d)
        # reorder original box corners based on the reconstructed box
        reordered_bbox_tensor = reorder_original_box(bbox_tensor, reconstructed_box)
        assert have_identical_corner_sets(reordered_bbox_tensor, bbox_tensor), \
            f"Reordered box is not same as original box in Sample {self.keys[idx]}"
        # Test whether the unique reconstructed box is acutally same as the some permuation of original box
        assert are_corners_close(reordered_bbox_tensor, reconstructed_box, atol=1e-3), \
            f"Reconstruction of the box is not 1-to-1 map Sample {self.keys[idx]}"
        
        sample = {
            "img_crop": img_tensor,
            "pc_pts": pc_tensor,
            "bbox_3d": reordered_bbox_tensor,
            "bbox_center": bbox_center,
            "bbox_dims": bbox_dims,
            "bbox_rot_6d": bbox_rot_6d,
            "key": self.keys[idx].decode('ascii')
        }
        
        # 5. Run tests on the sample and visualize
        if self.vis_sample:
            self.visualize_sample(sample, reordered_bbox_tensor)
        
        return sample

    def visualize_sample(self, sample: Dict, reconstructed_box: Optional[Tensor] = None) -> None:
        img_tensor = sample["img_crop"]
        pc_tensor = sample["pc_pts"]

        # Plotting Setup
        fig = plt.figure(figsize=(12, 5))
        
        # Plot RGB Crop (Channels 0:3, permuted back to HxWxC for Matplotlib)
        ax_img = fig.add_subplot(1, 2, 1)
        img_display = img_tensor[:3].permute(1, 2, 0).numpy()
        ax_img.imshow(img_display)
        
        # Overlay the Instance Mask (Channel 3)
        mask_display = img_tensor[3].numpy()
        ax_img.imshow(np.ma.masked_where(mask_display == 0, mask_display), cmap='gray_r', vmin=0, vmax=1, alpha=0.8)
        ax_img.set_title("RGB Crop + Mask Overlay")
        ax_img.axis("off")
        
        # Plot 3D Point Cloud and the RECONSTRUCTED box
        ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
        if reconstructed_box is not None:
            plot_instance(pc_pts=pc_tensor.numpy(), bbox_3d=reconstructed_box.numpy(), ax=ax_3d)
            ax_3d.set_title("3D Point Cloud & Reconstructed Box")
        else:
            original_box = sample["bbox_3d"]
            plot_instance(pc_pts=pc_tensor.numpy(), bbox_3d=original_box.numpy(), ax=ax_3d)
            ax_3d.set_title("3D Point Cloud & Original Box")
        
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
    # Test the Dataset and the Validation logic
    from config import Paths
    from config import DataLoaderConfig
    split = "val"
    dataset = LMDBInstanceDataset(os.path.join(Paths.parsed_data, split), data_loader_config=DataLoaderConfig,
                                  apply_aug=True, vis_sample=True)
    
    # Check if the dataset contains any samples before starting
    if len(dataset) == 0:
        print("Dataset is empty. Please check the LMDB path.")
    else:
        # Loop through every sample in the dataset for debugging and reconstruction verification
        for sample in tqdm(dataset):
            pass