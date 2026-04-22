import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.image as mpimg
import cv2 

from config import Paths
from utilities.utils import recover_intrinsics, get_rgb_crop, augment_instance

def draw_bboxes_on_image(rgb_image: np.ndarray, bbox: np.ndarray, pc: np.ndarray,  
                         target_idx: Optional[int] = None) -> np.ndarray:
    """
    Projects 3D bounding boxes onto the 2D image plane using the recovered camera matrix.
    If target_idx is provided, only draws that specific instance's bounding box.
    """
    K = recover_intrinsics(pc)
    if abs(K[0, 0] - 902.91) + abs(K[1, 1] - 902.91) > 1:
        print(f"fx {K[0, 0]}, fy {K[1, 1]}")
        
    img_drawn = rgb_image.copy() # mpimg loads as RGB
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4), # Top
        (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
    ]
    
    num_instances = bbox.shape[0]
    cmap = plt.get_cmap("tab10")
    
    for i in range(num_instances):
        # --- 1: Skip if this isn't the target index ---
        if target_idx is not None and i != target_idx:
            continue
            
        # Extract RGB color from matplotlib colormap for cv2 (0-255 range)
        color = cmap(i % 10)[:3]
        color = tuple(int(c * 255) for c in color)
        
        bbox_3d = bbox[i]
        pts_2d = []
        
        # Project each corner to 2D
        for pt in bbox_3d:
            X, Y, Z = pt
            if Z > 0: # Ensure point is in front of camera
                u = int(K[0,0] * (X/Z) + K[0,2])
                v = int(K[1,1] * (Y/Z) + K[1,2])
                pts_2d.append((u, v))
            else:
                pts_2d.append(None)
                
        # Draw lines between valid projected corners
        for edge in edges:
            pt1 = pts_2d[edge[0]]
            pt2 = pts_2d[edge[1]]
            if pt1 is not None and pt2 is not None:
                # Ensure points are within image bounds to prevent cv2 crashes
                cv2.line(img_drawn, pt1, pt2, color, 2)
                
    return img_drawn

def plot_instance(pc_pts: np.ndarray, bbox_3d: np.ndarray, 
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    pc_pts: (N, 3) 3D points of the instance
    bbox_3d: (8, 3) 3D bounding box corners
    """
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
    if len(pc_pts) > 2000:
        idx = np.random.choice(len(pc_pts), 2000, replace=False)
        pts = pc_pts[idx]
    else:
        pts = pc_pts

    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="blue", alpha=0.5)

    # Draw Bounding Box
    for edge in edges:
        ax.plot(bbox_3d[list(edge), 0], bbox_3d[list(edge), 1], bbox_3d[list(edge), 2], 
                c="red", linewidth=2)

    # Plot the Origin (Camera center) and Axes
    ax.scatter(0, 0, 0, color="black", s=200, marker="*", label="Origin (0,0,0)")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # --- Set Viewpoint ---
    ax.view_init(elev=-90, azim=-90)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    return ax

def visualize_all_instances_combined(pc: np.ndarray, mask: np.ndarray, bbox: np.ndarray, 
                                     img: np.ndarray, apply_aug: bool = False) -> None:
    """
    Creates a 2-Row grid. 
    Row 1: RGB Crops (Now with 3D BBoxes projected onto them!). 
    Row 2: 3D Point Cloud + BBox views below their respective crops.
    
    Input:
    pc.shape = (3, h, w), where 3 store the 3d coordinate of each pixel
    mask.shape = (num_instances, h, w)
    bbox.shape = (num_instances, h, w)
    image.shape = (h, w, 3), where 3 for RGB    
    """    
    num_instances = mask.shape[0]
    fig = plt.figure(figsize=(4 * num_instances, 8))
    
    for i in range(num_instances):
        # --- Draw only the specific box onto a fresh copy of the image inside the loop ---
        img_with_box = draw_bboxes_on_image(img, bbox, pc, target_idx=i)
        
        # --- Row 1: RGB Crop (uses the image that has the single box drawn on it) ---
        ax_rgb = fig.add_subplot(2, num_instances, i + 1)
        inst_mask_2d = mask[i]
        
        # Pass the image with the single bounding box into the crop function
        crop = get_rgb_crop(img_with_box, inst_mask_2d)
        
        # Extract exactly the points belonging to this instance first
        valid_pixels = inst_mask_2d > 0 # Shape: (h, w)
        pc_pts = pc[:, valid_pixels].T  # Shape: (N, 3)
        bbox_3d = bbox[i].copy()
        
        title_prefix = "Original"

        # Apply augmentation if requested and points exist
        if apply_aug and len(pc_pts) > 0:
            pc_pts, bbox_3d, crop = augment_instance(pc_pts, bbox_3d, crop)
            title_prefix = "Augmented"
        
        ax_rgb.imshow(crop)
        ax_rgb.set_title(f"Instance {i}: {title_prefix} RGB")
        ax_rgb.axis("off")
        
        # --- Row 2: 3D Visualization ---
        ax_3d = fig.add_subplot(2, num_instances, num_instances + i + 1, projection="3d")
        
        # Pass the extracted instance arrays directly to plot_instance
        plot_instance(pc_pts, bbox_3d, ax=ax_3d)
        ax_3d.set_title(f"Instance {i}: 3D")
        ax_3d.set_title(f"Instance {i}: {title_prefix} 3D")
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
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
        
        # Run with apply_aug=True to see the integrated augmentations
        visualize_all_instances_combined(pc=pc, mask=mask, bbox=bbox, img=img, apply_aug=True)