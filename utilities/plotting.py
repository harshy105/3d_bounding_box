import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.image as mpimg

from config import Paths
from utilities.utils import augment_instance

def plot_instance(pc_pts: np.ndarray, bbox_3d: np.ndarray, 
                  ax: plt.Axes = None) -> None:
    """Plots a single point cloud instance with its 3D bounding box."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4), # Top
        (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
    ]

    if ax is None: 
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Filter out any zero-padding
    non_zero_mask = np.any(pc_pts[:, :3] != 0, axis=1)
    valid_pts = pc_pts[non_zero_mask]
    
    # Subsample for faster plotting
    if len(valid_pts) > 2000:
        idx = np.random.choice(len(valid_pts), 2000, replace=False)
        pts = valid_pts[idx]
    else:
        pts = valid_pts

    if len(pts) > 0:
        xyz = pts[:, :3]
        colors = pts[:, 3:]
        
        # Matplotlib needs RGB values between 0.0 and 1.0
        if colors.max() > 1.0:
            colors = colors / 255.0
            
        # Plot the points with their colors
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=5, c=colors, alpha=0.8)

    # Draw Bounding Box
    for edge in edges:
        ax.plot(bbox_3d[list(edge), 0], bbox_3d[list(edge), 1], bbox_3d[list(edge), 2], 
                c="red", linewidth=2)

    # Show the origin for reference
    ax.scatter(0, 0, 0, color="black", s=200, marker="*", label="Origin (0,0,0)")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # Set a consistent viewpoint
    ax.view_init(elev=-90, azim=-90)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

def visualize_all_instances_combined(pc: np.ndarray, mask: np.ndarray, bbox: np.ndarray, 
                                     img: np.ndarray, apply_aug: bool = False) -> None:
    """Visualizes all detected instances in a scene, each in its own subplot. Can also show augmented versions."""
    num_instances = mask.shape[0]
    fig = plt.figure(figsize=(5 * num_instances, 6))
    
    for i in range(num_instances):
        ax_3d = fig.add_subplot(1, num_instances, i + 1, projection="3d")
        inst_mask_2d = mask[i]
        
        # Get the mask for the current instance
        valid_pixels = inst_mask_2d > 0 # Shape: (H, W)
        
        # Extract the XYZ points for this instance
        xyz = pc[:, valid_pixels].T  # Shape: (N, 3)
        
        # Extract the corresponding RGB colors
        rgb = img[valid_pixels]      # Shape: (N, 3)
        
        # Combine XYZ and RGB
        pc_pts = np.hstack((xyz, rgb)) 
        
        bbox_3d = bbox[i].copy()
        title_prefix = "Original"

        # Optionally apply augmentations for visualization
        if apply_aug and len(pc_pts) > 0:
            pc_pts, bbox_3d = augment_instance(pc_pts, bbox_3d)
            title_prefix = "Augmented"
        
        plot_instance(pc_pts, bbox_3d, ax=ax_3d)
        ax_3d.set_title(f"Instance {i}:\n{title_prefix} Colored Point Cloud")
        
    plt.tight_layout()
    plt.show()
    plt.close("all")

if __name__ == "__main__":
    data_path = Paths.data
    
    for scene_id in os.listdir(data_path):
        scene_dir = os.path.join(data_path, scene_id)
        pc = np.load(os.path.join(scene_dir, "pc.npy"))
        mask = np.load(os.path.join(scene_dir, "mask.npy"))
        bbox = np.load(os.path.join(scene_dir, "bbox3d.npy"))
        img = mpimg.imread(os.path.join(scene_dir, "rgb.jpg"))
        
        # Set apply_aug=True to check if augmentations are working correctly.
        visualize_all_instances_combined(pc=pc, mask=mask, bbox=bbox, img=img, apply_aug=True)