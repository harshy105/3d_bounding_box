import numpy as np
import cv2 

from typing import Optional, Tuple

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