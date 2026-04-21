import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
import matplotlib.image as mpimg
import cv2 

from config import Paths

def plot_instance(pc: np.array, mask: np.array, bbox: np.array, 
                  instance_idx: Optional[int]=0, ax: Optional[plt.axes] = None):
    # Flatten H and W to align coordinates with mask
    x, y, z = pc[0].flatten(), pc[1].flatten(), pc[2].flatten()
    all_points = np.stack([x, y, z], axis=1)
    
    # Filter points belonging to this specific instance
    inst_mask = mask[instance_idx].flatten()
    inst_points = all_points[inst_mask > 0]
    inst_bbox = bbox[instance_idx]

    # Define connectivity for an 8-corner 3D box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4), # Top
        (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
    ]

    if ax is None: 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Plot object points (subsampled for speed)
    if len(inst_points) > 2000:
        idx = np.random.choice(len(inst_points), 2000, replace=False)
        pts = inst_points[idx]
    else:
        pts = inst_points

    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="blue", alpha=0.5)

    # Draw Bounding Box
    for edge in edges:
        ax.plot(inst_bbox[list(edge), 0], inst_bbox[list(edge), 1], inst_bbox[list(edge), 2], 
                c="red", linewidth=2)

    # Plot the Origin (Camera center) and Axes
    ax.scatter(0, 0, 0, color="black", s=200, marker="*", label="Origin (0,0,0)")

    ax.set_title(f"Instance {instance_idx}: 3D")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # --- Set Viewpoint ---
    # elev=90, azim=-90 points the camera to look along the Z-axis towards -Z.
    ax.view_init(elev=-90, azim=-90)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    return ax

def get_rgb_crop(rgb_image, inst_mask_2d, padding_px=15, target_size=(64, 64)):
    # Uses the 2D pixel mask directly to crop the RGB image
    rows = np.any(inst_mask_2d > 0, axis=1)
    cols = np.any(inst_mask_2d > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
    v_min, v_max = np.where(rows)[0][[0, -1]]
    u_min, u_max = np.where(cols)[0][[0, -1]]
    
    h, w = rgb_image.shape[:2]
    
    v1 = max(0, v_min - padding_px)
    v2 = min(h, v_max + padding_px)
    u1 = max(0, u_min - padding_px)
    u2 = min(w, u_max + padding_px)
    
    crop = rgb_image[v1:v2, u1:u2]
    
    if crop.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
    crop_resized = cv2.resize(crop, target_size)
    return crop_resized

def visualize_all_instances_combined(pc: np.array, mask: np.array, bbox: np.array, rgb_path: str):
    """
    Creates a 2-Row grid. 
    Row 1: RGB Crops. 
    Row 2: 3D Point Cloud + BBox views below their respective crops.
    """
    img = mpimg.imread(rgb_path)
    num_instances = mask.shape[0]
    
    # Adjust width dynamically based on number of instances
    fig = plt.figure(figsize=(4 * num_instances, 8))
    
    for i in range(num_instances):
        # --- Row 1: RGB Crop ---
        # subplot index is 1-based. For top row, it's i + 1
        ax_rgb = fig.add_subplot(2, num_instances, i + 1)
        inst_mask_2d = mask[i]
        crop = get_rgb_crop(img, inst_mask_2d)
        ax_rgb.imshow(crop)
        ax_rgb.set_title(f"Instance {i}: RGB")
        ax_rgb.axis("off")
        
        # --- Row 2: 3D Visualization ---
        # For bottom row, subplot index is (num_instances + i + 1)
        ax_3d = fig.add_subplot(2, num_instances, num_instances + i + 1, projection="3d")
        plot_instance(pc, mask, bbox, instance_idx=i, ax=ax_3d)
        
    plt.tight_layout()
    plt.show()
    plt.close("all")

if __name__ == "__main__":
    data_path = Paths.data

    # Load the data
    scene_id = "8b061a8b-9915-11ee-9103-bbb8eae05561"
    pc = np.load(os.path.join(data_path, scene_id, "pc.npy"))
    mask = np.load(os.path.join(data_path, scene_id, "mask.npy"))
    bbox = np.load(os.path.join(data_path, scene_id, "bbox3d.npy"))
    rgb_path = os.path.join(data_path, scene_id, "rgb.jpg")
    
    # Run the top-down row/column visualization
    visualize_all_instances_combined(pc=pc, mask=mask, bbox=bbox, rgb_path=rgb_path)