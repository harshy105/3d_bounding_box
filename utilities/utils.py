import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from typing import Optional, Tuple
from torch import Tensor

def augment_instance(pc_pts: np.ndarray, bbox_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Applies random augmentations like rotation, translation, and color jitter to a point cloud instance and its bounding box."""
    pts_aug = pc_pts.copy()
    box_aug = bbox_3d.copy()
    
    # Separate geometry and color for easier manipulation
    xyz = pts_aug[:, :3]
    rgb = pts_aug[:, 3:]
    
    # Random rotation
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

    # Random translation/shift
    if np.random.rand() > 0.3:
        shift = np.random.uniform(-0.05, 0.05, size=(1, 3)) 
        xyz += shift
        box_aug += shift
        
    # Color jitter
    if np.random.rand() > 0.3:
        factor = np.random.uniform(0.7, 1.3)
        # Safely scale colors and clip
        max_val = 255.0 if rgb.max() > 1.0 else 1.0
        rgb = np.clip(rgb * factor, 0, max_val)
        
    # Horizontal flip
    if np.random.rand() > 0.5:
        xyz[:, 0] = -xyz[:, 0]
        box_aug[:, 0] = -box_aug[:, 0]
        
        # Reorder corners to maintain orientation after flip
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
    # Extract center and raw vectors
    center = box.mean(dim=0)
    vec_1 = box[1] - box[0] 
    vec_2 = box[4] - box[0] 
    vec_3 = box[1] - box[2] 
    
    vectors = [vec_1, vec_2, vec_3]
    lengths = [torch.norm(v) for v in vectors]
    
    # Sort by length to identify physical dimensions
    sorted_indices = torch.argsort(torch.tensor(lengths)) 
    
    vec_short = vectors[sorted_indices[0]]
    vec_mid = vectors[sorted_indices[1]]
    vec_long = vectors[sorted_indices[2]]
    
    # Apply rules to define a canonical orientation (e.g., "thin" dimension is Z)
    canonical_z = vec_short # Thickness is Z
    canonical_x = vec_long  # Length is X
    
    # Remove 180-degree ambiguities by anchoring the vectors.
    # Force Z to generally point "up" and X to point into the positive hemisphere.
    if canonical_z[2] < 0: 
        canonical_z = -canonical_z
        
    if canonical_x[0] < 0:
        canonical_x = -canonical_x

    # Enforce a strict Right-Hand-System (RHS)
    x_dir = canonical_x / torch.norm(canonical_x)
    z_dir = canonical_z / torch.norm(canonical_z)
    
    # Y is the cross product of Z and X to guarantee a valid rotation.
    y_dir = torch.cross(z_dir, x_dir, dim=0)
    
    # Assign dimensions based on our new canonical axes
    w = torch.norm(canonical_x)  # Width corresponds to X (Longest)
    h = torch.norm(vec_mid)      # Height corresponds to Y (Derived)
    l = torch.norm(canonical_z)  # Length corresponds to Z (Shortest / Thickness)
    dims = torch.stack([w, h, l])
    
    # The 6D representation is just the X and Y direction vectors concatenated.
    rot_6d = torch.cat([x_dir, y_dir], dim=0)
    
    return center, dims, rot_6d

def reconstruct_unique_box(center: Tensor, dims: Tensor, rot_6d: Tensor, 
                    output_rot_mat: Optional[bool] = False) -> Tuple[Tensor, Optional[Tensor]]:
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
    # Automatically handle batched vs unbatched inputs
    is_batched = center.dim() == 2
    if not is_batched:
        center = center.unsqueeze(0)
        dims = dims.unsqueeze(0)
        rot_6d = rot_6d.unsqueeze(0)
        
    B = center.shape[0]

    # Unpack the 6D representation into X and Y direction vectors
    v1_raw = rot_6d[:, :3]
    v2_raw = rot_6d[:, 3:]

    # Apply Gram-Schmidt to get a valid rotation matrix
    # Normalize X
    v1 = F.normalize(v1_raw, dim=1)
    
    # Make Y orthogonal to X and normalize it
    dot_product = torch.sum(v2_raw * v1, dim=1, keepdim=True)
    v2_proj = v2_raw - (dot_product * v1)
    v2 = F.normalize(v2_proj, dim=1)

    # Z is the cross product of X and Y
    v3 = torch.cross(v1, v2, dim=1)

    # Build the rotation matrix from the basis vectors
    R = torch.stack([v1, v2, v3], dim=2) # Shape: (B, 3, 3)

    # Define the 8 corners in local, un-rotated space
    w, h, l = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    
    x_coords = torch.cat([-w/2,  w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2], dim=1)
    y_coords = torch.cat([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2], dim=1)
    z_coords = torch.cat([-l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2], dim=1)

    
    local_corners = torch.stack([x_coords, y_coords, z_coords], dim=2) # Shape: (B, 8, 3)

    # Apply rotation and translation to get the final world coordinates
    # Batched matrix multiplication
    rotated_corners = torch.bmm(local_corners, R.transpose(1, 2))
    
    # Translate to the box center
    box = rotated_corners + center.unsqueeze(1)

    # Squeeze if the input was not batched
    if not is_batched:
        box = box.squeeze(0)
        
    if output_rot_mat:
        return box, R
        
    return box

def reorder_original_box(original_box: Tensor, reconstructed_box: Tensor) -> Tensor:
    """Aligns the corner order of an original bounding box to match a reconstructed one.
    This is crucial for stable loss calculation.
    """
    # Find the distance from every reconstructed corner to every original corner
    dists = torch.cdist(reconstructed_box, original_box)
    
    # For each reconstructed corner, find the index of the closest original corner
    closest_indices = torch.argmin(dists, dim=1)
    
    # Re-index the original box to match the reconstructed order
    reordered_box = original_box[closest_indices]
    
    return reordered_box

def apply_weights(m: nn.Module) -> None:
    """Initializes weights for various layers in the network. 
    It's strict and will error on unknown layer types to prevent accidental 
    uninitialized layers.
    """
    # We only want to initialize leaf modules, so skip containers
    if len(list(m.children())) > 0:
        return

    # Apply initialization based on layer type
    if isinstance(m, nn.Conv1d):
        # Kaiming is optimal for ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
            
    elif isinstance(m, nn.Linear):
        # Use a smaller initialization for the final prediction layers
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
        # Standard BatchNorm init
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    
    elif isinstance(m, nn.ReLU):
        # ReLU has no parameters
        pass
              
    else:
        # Throw an error if we find a layer we don't know how to initialize
        raise ValueError(
            f"Strict Init Error: Unexpected layer type encountered -> {type(m).__name__}. "
            f"Please add it to the apply_weights function."
        )