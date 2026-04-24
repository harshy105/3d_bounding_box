import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Optional, Tuple
from torch import Tensor

def augment_instance(pc_pts: np.ndarray, bbox_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies augmentations to a single instance.
    pc_pts: (N, 6) 3D points + RGB values of the instance
    bbox_3d: (8, 3) 3D bounding box corners
    """
    pts_aug = pc_pts.copy()
    box_aug = bbox_3d.copy()
    
    # Separate geometry and color for easier manipulation
    xyz = pts_aug[:, :3]
    rgb = pts_aug[:, 3:]
    
    # 1. 3D Geometric Augmentations (Rotation)
    if np.random.rand() > 0.3:
        theta = np.random.uniform(-np.pi / 4, np.pi / 4)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([
            [ cos_t, 0, sin_t],
            [     0, 1,     0],
            [-sin_t, 0, cos_t]
        ])
        xyz = xyz @ R.T
        box_aug = box_aug @ R.T

    # 2. 3D Geometric Augmentations (Translation/Shift)
    if np.random.rand() > 0.3:
        shift = np.random.uniform(-0.05, 0.05, size=(1, 3)) 
        xyz += shift
        box_aug += shift
        
    # 3. Color Augmentations (Brightness Jittering)
    if np.random.rand() > 0.3:
        factor = np.random.uniform(0.7, 1.3)
        # Safely scale colors and clip to prevent blowing out the values
        max_val = 255.0 if rgb.max() > 1.0 else 1.0
        rgb = np.clip(rgb * factor, 0, max_val)
        
    # 4. Coupled Geometric Augmentation (Horizontal Flip)
    if np.random.rand() > 0.5:
        xyz[:, 0] = -xyz[:, 0]
        box_aug[:, 0] = -box_aug[:, 0]
        
        # --- Reorder the corners to maintain orientation ---
        swap_indices = [1, 0, 3, 2, 5, 4, 7, 6] 
        box_aug = box_aug[swap_indices]

    # Recombine augmented XYZ and RGB
    pts_aug[:, :3] = xyz
    pts_aug[:, 3:] = rgb

    return pts_aug, box_aug

def extract_3d_bbox_params(box: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
    y_dir = torch.cross(z_dir, x_dir, dim=0)
    
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
        center (torch.Tensor): Shape (3,) or (B, 3)
        dims (torch.Tensor): Shape (3,) or (B, 3) -> [Width, Height, Length]
        rot_6d (torch.Tensor): Shape (6,) or or (B, 6)
        
    Returns:
        box (torch.Tensor): Shape (8, 3) or (B, 8, 3), reconstructed bounding box
    """
    # 0. Automatically handle batched vs unbatched inputs
    is_batched = center.dim() == 2
    if not is_batched:
        center = center.unsqueeze(0)
        dims = dims.unsqueeze(0)
        rot_6d = rot_6d.unsqueeze(0)
        
    B = center.shape[0]

    # 1. Unpack the 6D representation back into raw X and Y vectors
    v1_raw = rot_6d[:, :3]
    v2_raw = rot_6d[:, 3:]

    # 2. Apply Gram-Schmidt Orthogonalization
    # Normalize X
    v1 = F.normalize(v1_raw, dim=1)
    
    # Make Y orthogonal to X, then normalize
    dot_product = torch.sum(v2_raw * v1, dim=1, keepdim=True)
    v2_proj = v2_raw - (dot_product * v1)
    v2 = F.normalize(v2_proj, dim=1)

    # 3. Mathematically enforce the Z axis via Cross Product
    v3 = torch.cross(v1, v2, dim=1)

    # 4. Build the 3x3 Rotation Matrix (v1, v2, v3 as columns)
    R = torch.stack([v1, v2, v3], dim=2) # Shape: (B, 3, 3)

    # 5. Define the 8 corners in local space based on dims (w, h, l)
    w, h, l = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    
    x_coords = torch.cat([-w/2,  w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2], dim=1)
    y_coords = torch.cat([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2], dim=1)
    z_coords = torch.cat([-l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2], dim=1)

    
    local_corners = torch.stack([x_coords, y_coords, z_coords], dim=2) # Shape: (B, 8, 3)

    # 6. Apply Rotation and Translation
    # Batched matrix multiplication: (B, 8, 3) @ (B, 3, 3) -> (B, 8, 3)
    rotated_corners = torch.bmm(local_corners, R.transpose(1, 2))
    
    # Translate to center: (B, 8, 3) + (B, 1, 3)
    box = rotated_corners + center.unsqueeze(1)

    # Return original shape if it wasn't batched
    if not is_batched:
        box = box.squeeze(0)
        
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

def apply_weights(m: nn.Module) -> None:
    """
    Global weight initialization function.
    Raises a ValueError if an unexpected leaf layer is encountered.
    """
    # 1. Skip parent modules and containers (like nn.Sequential or the Network itself)
    # We only want to initialize and strictly check the 'leaf' modules.
    if len(list(m.children())) > 0:
        return

    # 2. Apply specific initializations based on the exact layer type
    if isinstance(m, nn.Conv1d):
        # Kaiming is optimal for ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
    elif isinstance(m, nn.Linear):
        # --- The Zero-Output Trick for Final Prediction Layers ---
        # m.out_features == 9: Final BBox Regression (dims + 6D rot)
        # m.out_features == 3: Final Voting Module offsets (dx, dy, dz)
        if m.out_features in [3, 9]:
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        else:
            # Normal Kaiming for all other hidden Linear layers inside MLPs
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            
    elif isinstance(m, nn.BatchNorm1d):
        # Standard BatchNorm init: weight (gamma) = 1, bias (beta) = 0
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    
    elif isinstance(m, nn.ReLU):
        # ReLU has no parameters, so we simply pass
        pass
              
    else:
        # 3. Raise an error if any other leaf layer is found
        raise ValueError(
            f"Strict Init Error: Unexpected layer type encountered -> {type(m).__name__}. "
            f"Please add it to the apply_weights function."
        )