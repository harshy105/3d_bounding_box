import numpy as np
import cv2 
import torch
import torch.nn.functional as F

from typing import Optional, Tuple
from torch import Tensor

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
        return np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
        
    v_min, v_max = np.where(rows)[0][[0, -1]]
    u_min, u_max = np.where(cols)[0][[0, -1]]
    
    h, w = rgb_image.shape[:2]
    
    v1 = max(0, v_min - padding_px)
    v2 = min(h, v_max + padding_px)
    u1 = max(0, u_min - padding_px)
    u2 = min(w, u_max + padding_px)
    
    crop = rgb_image[v1:v2, u1:u2]
    mask_crop = inst_mask_2d[v1:v2, u1:u2] # Crop the mask alongside the RGB
    
    if crop.size == 0:
        return np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
        
    crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Resize the mask (using nearest neighbor to avoid interpolation blurring)
    mask_resized = cv2.resize(mask_crop.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
    
    # Ensure binary format: 1 for instance, 0 otherwise
    mask_resized = (mask_resized > 0).astype(np.uint8)
    
    # Expand dims to (64, 64, 1) and concatenate to create (64, 64, 4)
    mask_resized = np.expand_dims(mask_resized, axis=-1)
    crop_4d = np.concatenate([crop_resized, mask_resized], axis=-1)
    
    return crop_4d

def augment_instance(pc_pts: np.ndarray, bbox_3d: np.ndarray, img_crop: np.ndarray) -> Tuple[np.ndarray]:
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
        
        # --- Reorder the corners to maintain orientation ---
        # [0,1,2,3] is the bottom ring, [4,5,6,7] is the top ring.
        # Left/Right pairs: (0,1), (3,2), (4,5), (7,6)
        
        swap_indices = [1, 0, 3, 2, 5, 4, 7, 6] 
        box_aug = box_aug[swap_indices]

    return pts_aug, box_aug, img_aug

def extract_3d_bbox_params(box: Tensor) -> Tuple[Tensor]:
    """
    Converts an (8, 3) bounding box into its center, dimensions, and 6D rotation.
    Generate 6D representation of 3D angles based on
    Zhou et. al. On the Continuity of Rotation Representations in Neural Networks 
    
    Args:
        box (torch.Tensor): Shape (8, 3), corners ordered consistently.
        
    Returns:
        center (torch.Tensor): Shape (3,)
        dims (torch.Tensor): Shape (3,) -> [Width, Height, Length]
        rot_6d (torch.Tensor): Shape (6,) -> Continuous 6D rotation representation
    """
    # 1. Extract center and raw vectors using strict RHS indices
    center = box.mean(dim=0)
    vec_1 = box[1] - box[0] 
    vec_2 = box[4] - box[0] 
    vec_3 = box[1] - box[2] 
    
    vectors = [vec_1, vec_2, vec_3]
    lengths = [torch.norm(v) for v in vectors]
    
    # 2. Sort by length to identify physical features
    sorted_indices = torch.argsort(torch.tensor(lengths)) 
    
    vec_short = vectors[sorted_indices[0]]
    vec_mid = vectors[sorted_indices[1]]
    vec_long = vectors[sorted_indices[2]]
    
    # 3. Apply Planar Rules ("Thin along Z")
    canonical_z = vec_short # Thickness becomes Z
    canonical_x = vec_long  # Primary length becomes X
    
    # 4. Remove 180-degree Ambiguities (Anchor the vectors)
    # Force the thickness normal (Z) to generally point "Up" in the global world
    # Assuming global Z is your depth/up axis. Adjust index [2] if your global Up is Y [1]
    if canonical_z[2] < 0: 
        canonical_z = -canonical_z
        
    # Force the main heading (X) to always point into the positive X hemisphere
    if canonical_x[0] < 0:
        canonical_x = -canonical_x

    # 5. Enforce strict Right-Hand System (RHS)
    # Normalize our locked X and Z directions
    x_dir = canonical_x / torch.norm(canonical_x)
    z_dir = canonical_z / torch.norm(canonical_z)
    
    # Mathematically forge Y. In RHS: Z cross X = Y.
    # This guarantees the box doesn't flip inside out, and perfectly defines the middle edge.
    y_dir = torch.cross(z_dir, x_dir)
    
    # 6. Assign Dimensions based on our new axes
    w = torch.norm(canonical_x)  # Width corresponds to X (Longest)
    h = torch.norm(vec_mid)      # Height corresponds to Y (Derived)
    l = torch.norm(canonical_z)  # Length corresponds to Z (Shortest / Thickness)
    dims = torch.stack([w, h, l])
    
    # 7. Create the 6D output (Requires X and Y vectors)
    rot_6d = torch.cat([x_dir, y_dir], dim=0)
    
    return center, dims, rot_6d

def reconstruct_unique_box(center: Tensor, dims: Tensor, rot_6d: Tensor) -> Tensor:
    """
    Reconstructs unique (8, 3) bounding box from parameters.
    3D rotation is unique and continous as shown in
    Zhou et. al. On the Continuity of Rotation Representations in Neural Networks 
    
    Args:
        center (torch.Tensor): Shape (3,)
        dims (torch.Tensor): Shape (3,) -> [Width, Height, Length]
        rot_6d (torch.Tensor): Shape (6,)
        
    Returns:
        box (torch.Tensor): Shape (8, 3), reconstructed bounding box
    """
    w, h, l = dims
    
    # 1. Unpack the 6D representation back into raw X and Y vectors
    v1 = rot_6d[:3]
    v2 = rot_6d[3:]
    
    # 2. Check the if 6D representation of 3D rotation is correct
    assert abs(torch.norm(v1) - 1.0) < 1e-5, "Roll axis (X) is not normalized"
    assert abs(torch.norm(v2) - 1.0) < 1e-5, "Pitch axis (Y) is not normalized"
    assert abs(torch.dot(v1, v2)) < 1e-5,  "Roll (X) and Pitch (Y) axis are not orthogonal"
    
    # 3. Mathematically enforce the Z axis via Cross Product (Right-Hand Rule)
    # This guarantees the determinant is +1 and prevents mirrored boxes
    v3 = torch.cross(v1, v2)
    
    # 4. Build the 3x3 Rotation Matrix
    R = torch.stack([v1, v2, v3], dim=1)
    
    # 5. Define the 8 corners in local space (centered at origin, 0 rotation)
    # The order MUST match the assumptions made in the forward function
    x_coords = torch.tensor([-w/2,  w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2])
    y_coords = torch.tensor([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2])
    z_coords = torch.tensor([-l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2])
    
    local_corners = torch.stack([x_coords, y_coords, z_coords], dim=1) # Shape (8, 3)
    
    # 6. Apply Rotation and Translation
    # Rotate: local_corners @ R.T (Matrix multiplication)
    rotated_corners = torch.matmul(local_corners, R.t())
    
    # Translate to center
    box = rotated_corners + center
    
    return box

def reorder_original_box(original_box: Tensor, reconstructed_box: Tensor) -> Tensor:
    """
    Reorders the corners of the original bounding box to perfectly align 
    with the sequence of the reconstructed bounding box based on minimum spatial distance.
    
    Args:
        original_box (torch.Tensor): Shape (8, 3), the unordered original corners.
        reconstructed_box (torch.Tensor): Shape (8, 3), the canonical reconstructed corners.
        
    Returns:
        reordered_box (torch.Tensor): Shape (8, 3), the original box physically sorted 
                                      to match the reconstructed box's order.
    """
    # Computes the distance from every point in the Reconstructed Box 
    # to every point in the Original Box. Output shape: (8, 8)
    dists = torch.cdist(reconstructed_box, original_box)
    
    # For each point in the reconstructed box (dim 0), find the index of the 
    # physically closest point in the original box (dim 1)
    closest_indices = torch.argmin(dists, dim=1)
    
    # Reindex the original box using the matched assignments
    reordered_box = original_box[closest_indices]
    
    return reordered_box