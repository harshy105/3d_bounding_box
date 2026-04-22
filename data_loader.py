import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import lmdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from config import Paths
from utilities.utils import extract_3d_bbox_params, augment_instance, reconstruct_box
from utilities.plotting import draw_bboxes_on_image, plot_instance

class LMDBInstanceDataset(Dataset):
    def __init__(self, lmdb_path: str, apply_aug: bool = False, num_points: int = 1024) -> None:
        self.lmdb_path = lmdb_path
        self.apply_aug = apply_aug
        self.num_points = num_points
        self.env = None
        
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
            
        return self.process_sample(sample)
            
    def process_sample(self, sample: Dict) -> Dict:            
        pc_pts = sample["pc_pts"]
        bbox_3d = sample["bbox_3d"]
        img_crop = sample["img_crop"]
        
        # 1. Apply data augmentation
        if self.apply_aug:
            pc_pts, bbox_3d, img_crop = augment_instance(pc_pts, bbox_3d, img_crop)
            
        # Ensure uniform point cloud size for batching (Subsample or Pad)
        num_current_pts = pc_pts.shape[0]
        if num_current_pts >= self.num_points:
            choice = np.random.choice(num_current_pts, self.num_points, replace=False)
        else:
            choice = np.random.choice(num_current_pts, self.num_points, replace=True)
        pc_pts = pc_pts[choice, :]

        # 3. Convert image to Torch and permute to (Channels, Height, Width) format
        img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
        
        # 4. Convert point cloud to Torch
        pc_tensor = torch.from_numpy(pc_pts).float()
        
        # Convert bounding box to Torch before parameter extraction
        bbox_tensor = torch.from_numpy(bbox_3d).float()
        
        # 2. Extract features from the 3D bounding box
        # Assuming your function returns a tuple of parameters (e.g., center, size, quaternion)
        bbox_params = extract_3d_bbox_params(bbox_tensor)
        
        return {
            "img_crop": img_tensor,
            "pc_pts": pc_tensor,
            "bbox_3d": bbox_tensor,
            "bbox_params": bbox_params, # Varies based on your extraction logic
            "key": self.keys[idx].decode('ascii')
        }

    def visualize_debug_sample(self, idx: int) -> None:
        """
        5. Visualizes a sample to debug the extraction and reconstruction pipeline.
        """
        sample = self[idx]
        
        img_tensor = sample["img_crop"]
        pc_tensor = sample["pc_pts"]
        original_box = sample["bbox_3d"]
        bbox_params = sample["bbox_params"]
        
        # Reconstruct the box from the extracted parameters
        # (Unpack params according to what extract_3d_bbox_params returns)
        reconstructed_box = reconstruct_box(*bbox_params)
        
        # Validation Check
        is_match = torch.allclose(original_box, reconstructed_box, atol=1e-3)
        print(f"Sample: {sample['key']}")
        print(f"Reconstruction Match (atol=1e-3): {is_match}")
        if not is_match:
            print(f"Max Diff: {torch.max(torch.abs(original_box - reconstructed_box))}")

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
        plot_instance(pc_pts=pc_tensor.numpy(), bbox_3d=reconstructed_box.numpy(), ax=ax_3d)
        ax_3d.set_title("3D Point Cloud & Reconstructed Box")
        
        plt.tight_layout()
        plt.show()
        plt.close("all")


class InstanceDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 4, apply_aug: bool = True) -> None:
        super().__init__()
        self.lmdb_path = Paths.parsed_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.apply_aug = apply_aug

    def setup(self, stage=None) -> None:
        # In a real scenario, you'd split your LMDB into train/val here.
        # For now, we wrap the whole parsed data.
        self.train_dataset = LMDBInstanceDataset(self.lmdb_path, apply_aug=self.apply_aug)
        self.val_dataset = LMDBInstanceDataset(self.lmdb_path, apply_aug=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)

if __name__ == "__main__":
    # Test the Dataset and the Validation logic
    dataset = LMDBInstanceDataset(Paths.parsed_data, apply_aug=True)
    
    # Check if the dataset contains any samples before starting
    if len(dataset) == 0:
        print("Dataset is empty. Please check the LMDB path.")
    else:
        # Loop through every sample in the dataset for debugging and reconstruction verification
        for idx in range(len(dataset)):
            dataset.visualize_debug_sample(idx)