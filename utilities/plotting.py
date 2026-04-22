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

def get_rgb_crop(rgb_image: np.ndarray, inst_mask_2d: np.ndarray, padding_px: Optional[int] = 0,
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

def augment_instance(pc_pts: np.ndarray, bbox_3d: np.ndarray, img_crop: np.ndarray):
    """
    Applies decoupled and coupled augmentations to a single instance.
    pc_pts: (N, 3) 3D points of the instance
    bbox_3d: (8, 3) 3D bounding box corners
    img_crop: (H, W, 3) RGB crop of the instance
    """
    pts_aug = pc_pts.copy()
    box_aug = bbox_3d.copy()
    img_aug = img_crop.copy()
    
    # Decoupled 3D (bbox and point cloud) Geometric Augmentations
    if np.random.rand() > 0.3:
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([
            [ cos_t, 0, sin_t],
            [     0, 1,     0],
            [-sin_t, 0, cos_t]
        ])
        pts_aug = pts_aug @ R.T
        box_aug = box_aug @ R.T

    if np.random.rand() > 0.3:
        shift = np.random.uniform(-0.05, 0.05, size=(1, 3)) 
        pts_aug += shift
        box_aug += shift
        
    # Decoupled 2D Image Augmentations
    if np.random.rand() > 0.3:
        factor = np.random.uniform(0.7, 1.3)
        img_aug = cv2.convertScaleAbs(img_aug, alpha=factor, beta=0)

    # if np.random.rand() > 0.3:
    #     h, w = img_aug.shape[:2]
    #     crop_h, crop_w = int(h * 0.3), int(w * 0.3)
    #     x = np.random.randint(0, w - crop_w)
    #     y = np.random.randint(0, h - crop_h)
    #     img_aug[y:y+crop_h, x:x+crop_w] = 0 
        
    # 3. Coupled 2D-3D Augmentation (Horizontal Flip)
    if np.random.rand() > 0.5:
        img_aug = cv2.flip(img_aug, 1) 
        pts_aug[:, 0] = -pts_aug[:, 0]
        box_aug[:, 0] = -box_aug[:, 0]

    return pts_aug, box_aug, img_aug

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
        
        # --- Augmentation Logic ---
        pc_plot = pc
        bbox_plot = bbox
        title_prefix = "Original"

        if apply_aug:
            # Extract exactly the points belonging to this instance
            valid_pixels = inst_mask_2d > 0 # Shape: (h, w)
            pts = pc[:, valid_pixels].T  # Shape: (N, 3)
            
            if len(pts) > 0: 
                # 2. Apply augmentation
                pts_aug, box_aug, crop_aug = augment_instance(pts, bbox[i], crop)
                
                # 3. Inject augmented points back into a temporary copy 
                #    so plot_instance can read them seamlessly
                pc_plot = pc.copy()
                pc_plot[:, valid_pixels] = pts_aug.T
                
                bbox_plot = bbox.copy()
                bbox_plot[i] = box_aug
                
                crop = crop_aug
                title_prefix = "Augmented"
        
        ax_rgb.imshow(crop)
        ax_rgb.set_title(f"Instance {i}: {title_prefix} RGB")
        ax_rgb.axis("off")
        
        # --- Row 2: 3D Visualization ---
        ax_3d = fig.add_subplot(2, num_instances, num_instances + i + 1, projection="3d")
        
        # Pass the (potentially modified) full arrays to the untouched plot_instance
        plot_instance(pc_plot, mask, bbox_plot, instance_idx=i, ax=ax_3d)
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