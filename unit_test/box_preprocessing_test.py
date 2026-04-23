import os
from config import Paths
import torch
import numpy as np
from torch import Tensor
from typing import Optional

from utilities.utils import extract_3d_bbox_params, reconstruct_unique_box, reorder_original_box

def have_identical_corner_sets(box_a: Tensor, box_b: Tensor) -> bool:
    """
    Checks if both the boxes contain exact corners with possibly different permutation
    """
    # Convert tensors to lists of tuples so they are hashable
    set_a = set(tuple(row) for row in box_a.tolist())
    set_b= set(tuple(row) for row in box_b.tolist())
    
    # Check 1: Are the sets identical? (Order independent)
    # Check 2: Are the lengths still 8?
    return (set_a == set_b) and (len(set_a) == 8) and (len(set_b) == 8)

def are_corners_close(box_a: Tensor, box_b: Tensor, atol: Optional[float] = 1e-3) -> bool:
    """
    Checks if both the boxes' corners are close enough within some give tolerance
    """
    return torch.allclose(box_a, box_b, atol=atol)

if __name__ == "__main__":
    data_path = Paths.data
    
    # Flag to track if the entire dataset passes
    all_passed = True 
    
    for scene_id in os.listdir(data_path):
        scene_dir = os.path.join(data_path, scene_id)
        
        # Skip if it's not a directory
        if not os.path.isdir(scene_dir):
            continue
            
        # Load the data
        bbox_np = np.load(os.path.join(scene_dir, "bbox3d.npy"))
        
        # Convert the NumPy bounding box to a PyTorch FloatTensor
        bbox_tensor = torch.from_numpy(bbox_np).float()
        
        # Ensure shape is (N, 8, 3) so we can loop over it safely
        if bbox_tensor.ndim == 2:
            bbox_tensor = bbox_tensor.unsqueeze(0)
            
        for i, original_box in enumerate(bbox_tensor):
            
            # 1. Forward Pass: Extract the 6D representation, center, and dims
            center, dims, rot_6d = extract_3d_bbox_params(original_box)
            
            # 2. Reverse Pass: Reconstruct the box from the parameters
            reconstructed_box = reconstruct_unique_box(center, dims, rot_6d)
            
            # 3. Align the original box corner sequence to the new canonical box corner sequence
            reordered_original_box = reorder_original_box(original_box, reconstructed_box)
            
            # 4. Validation
            is_match = (have_identical_corner_sets(reordered_original_box, original_box) and
                are_corners_close(reordered_original_box, reconstructed_box, atol=1e-3))
            
            if not is_match:
                print(f"Scene {scene_id} | Box {i}: Mismatch!")
                print("orignal box: ", original_box)
                print("reconstructed box: ", reconstructed_box)
                # Calculate the maximum absolute error for debugging
                max_error = torch.max(torch.abs(original_box - reconstructed_box))
                print(f"   Max Error Distance: {max_error.item():.6f}")
                all_passed = False

    # Final summary
    print("-" * 40)
    if all_passed:
        print("SUCCESS: All bounding boxes successfully mapped 1-to-1 without ambiguity!")
    else:
        print("WARNING: Mismatches found. Verify that the corner ordering in your data matches the extraction logic.")