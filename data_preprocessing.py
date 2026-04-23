import os
import numpy as np
import shutil
import matplotlib.image as mpimg
import lmdb
import pickle
import warnings
from tqdm import tqdm

from config import Paths
from config import DataPreprocessingConfig
from utilities.utils import get_rgb_crop

if __name__ == "__main__":
    data_path = Paths.data
    parsed_data_path = Paths.parsed_data # store the data here
    
    # Check if the parsed data directory already exists and remove it
    if os.path.exists(parsed_data_path):
        print(f"Removing existing parsed data at: {parsed_data_path}")
        shutil.rmtree(parsed_data_path)
        
    # Ensure the target directory for LMDB exists
    os.makedirs(parsed_data_path, exist_ok=True)
    
    # Get all valid scene directories
    all_scenes = [s for s in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, s))]
    
    # 1. Set a fixed random seed to ensure reproducibility (unique but consistent every run)
    np.random.seed(42)
    
    # 2. Shuffle scenes sparsely (randomized order)
    np.random.shuffle(all_scenes)
    
    # 3. Calculate split indices (80%, 10%, 10%)
    num_scenes = len(all_scenes)
    train_idx = int(num_scenes * 0.8)
    val_idx = int(num_scenes * 0.9)
    
    scene_splits = {
        'train': set(all_scenes[:train_idx]),
        'val': set(all_scenes[train_idx:val_idx]),
        'test': set(all_scenes[val_idx:])
    }
    
    # Define LMDB map size
    map_size = int(1e9)
    
    # 4. Initialize separate LMDB environments and transactions for each split
    envs = {}
    txns = {}
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(parsed_data_path, split_name)
        os.makedirs(split_path, exist_ok=True)
        envs[split_name] = lmdb.open(split_path, map_size=map_size)
        txns[split_name] = envs[split_name].begin(write=True)
        
    for scene_id in tqdm(all_scenes):
        scene_dir = os.path.join(data_path, scene_id)
        
        # Determine which split this scene belongs to
        if scene_id in scene_splits['train']:
            current_txn = txns['train']
        elif scene_id in scene_splits['val']:
            current_txn = txns['val']
        else:
            current_txn = txns['test']
                        
        pc = np.load(os.path.join(scene_dir, "pc.npy"))
        mask = np.load(os.path.join(scene_dir, "mask.npy"))
        bbox = np.load(os.path.join(scene_dir, "bbox3d.npy"))
        img = mpimg.imread(os.path.join(scene_dir, "rgb.jpg"))
        
        num_instances = mask.shape[0]
        
        for i in range(num_instances):
            inst_mask_2d = mask[i]
            
            # Extract valid pixels for this specific instance
            valid_pixels = inst_mask_2d > 0
            
            # Extract point cloud (N, 3)
            pc_pts = pc[:, valid_pixels].T 
            
            # Skip empty instances (completely occluded or out of frame)
            if len(pc_pts) == 0:
                warnings.warn(f"Scene {scene_id}, Instance {i} has no point clouds")
                continue
                
            # Extract bounding box (8, 3)
            bbox_3d = bbox[i]
            
            # Extract image crop (64, 64, 4), 4th is the mask
            img_crop = get_rgb_crop(img, inst_mask_2d, target_size=DataPreprocessingConfig.crop_img_size)
            
            # Package the parsed data
            sample = {
                "bbox_3d": bbox_3d,
                "img_crop": img_crop,
                "pc_pts": pc_pts
            }
            
            # Create a unique key for the database
            key = f"{scene_id}_inst_{i}".encode("ascii")
            
            # Serialize the dictionary to bytes and write to the correct LMDB split
            current_txn.put(key, pickle.dumps(sample))
            
    # 5. Commit transactions and close environments
    for split_name in ['train', 'val', 'test']:
        txns[split_name].commit()
        envs[split_name].close()
    
    print(f"Data successfully parsed and split (80/10/10) into LMDB at: {parsed_data_path}")