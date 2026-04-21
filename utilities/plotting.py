import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple
import matplotlib.image as mpimg
import cv2 

from config import Paths

def recover_intrinsics(pc: np.ndarray) -> np.ndarray:
    """
    Reverse-engineers the Camera Intrinsic Matrix (K) from the organized point cloud.
    Uses u = fx * (X/Z) + cx and v = fy * (Y/Z) + cy.
    """
    _, H, W = pc.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    X, Y, Z = pc[0].flatten(), pc[1].flatten(), pc[2].flatten()
    u, v = u.flatten(), v.flatten()
    
    # Filter valid depth points
    valid = (Z > 0.1) & np.isfinite(Z) & np.isfinite(X) & np.isfinite(Y)
    
    # Linear fit to find focal lengths (slope) and optical centers (intercept)
    fx, cx = np.polyfit(X[valid] / Z[valid], u[valid], 1)
    fy, cy = np.polyfit(Y[valid] / Z[valid], v[valid], 1)
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K

def draw_bboxes_on_image(rgb_image: np.ndarray, bbox: np.ndarray, pc: np.ndarray,  
                         target_idx: Optional[int] = None) -> np.ndarray:
    """
    Projects 3D bounding boxes onto the 2D image plane using the recovered camera matrix.
    If target_idx is provided, only draws that specific instance's bounding box.
    """
    K = recover_intrinsics(pc)
    img_drawn = rgb_image.copy() # mpimg loads as RGB
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4), # Top
        (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
    ]
    
    num_instances = bbox.shape[0]
    cmap = plt.get_cmap("tab10")
    
    for i in range(num_instances):
        # --- MINIMAL CHANGE 1: Skip if this isn't the target index ---
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

def plot_instance(pc: np.ndarray, mask: np.ndarray, bbox: np.ndarray, 
                  instance_idx: Optional[int]=0, 
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
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
    ax.view_init(elev=-90, azim=-90)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    return ax

def get_rgb_crop(rgb_image: np.ndarray, inst_mask_2d: np.ndarray, padding_px: Optional[int] = 15,
                  target_size: Optional[Tuple]=(64, 64)) -> np.ndarray:
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

def visualize_all_instances_combined(pc: np.ndarray, mask: np.ndarray, bbox: np.ndarray, 
                                     rgb_path: str) -> None:
    """
    Creates a 2-Row grid. 
    Row 1: RGB Crops (Now with 3D BBoxes projected onto them!). 
    Row 2: 3D Point Cloud + BBox views below their respective crops.
    """
    img = mpimg.imread(rgb_path)
    
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
        
        ax_rgb.imshow(crop)
        ax_rgb.set_title(f"Instance {i}: RGB + 2D Proj")
        ax_rgb.axis("off")
        
        # --- Row 2: 3D Visualization ---
        ax_3d = fig.add_subplot(2, num_instances, num_instances + i + 1, projection="3d")
        plot_instance(pc, mask, bbox, instance_idx=i, ax=ax_3d)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
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