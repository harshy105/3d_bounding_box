import os
import numpy as np
import shutil
import matplotlib.image as mpimg
import lmdb
import pickle
import warnings
from tqdm import tqdm

from config import Paths
from config import data_preprocessing
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
    
    # Define LMDB map size (set to 10GB here, adjust if your dataset is larger)
    map_size = int(1e9)
    
    # Initialize the LMDB environment
    env = lmdb.open(parsed_data_path, map_size=map_size)
    
    # Open a single write transaction for efficiency
    with env.begin(write=True) as txn:
        
        for scene_id in tqdm(os.listdir(data_path)):
            scene_dir = os.path.join(data_path, scene_id)
                            
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
                    warnings.warn("This instance has no point clouds")
                    continue
                    
                # Extract bounding box (8, 3)
                bbox_3d = bbox[i]
                
                # Extract image crop (64, 64, 3)
                img_crop = get_rgb_crop(img, inst_mask_2d, target_size=data_preprocessing.crop_img_size)
                
                # Package the parsed data
                sample = {
                    "bbox_3d": bbox_3d,
                    "img_crop": img_crop,
                    "pc_pts": pc_pts
                }
                
                # Create a unique key for the database
                key = f"{scene_id}_inst_{i}".encode("ascii")
                
                # Serialize the dictionary to bytes and write to LMDB
                txn.put(key, pickle.dumps(sample))
                
    # Close the environment after the transaction finishes
    env.close()
    
    print(f"Data successfully parsed and stored in LMDB at: {parsed_data_path}")