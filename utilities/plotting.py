import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from config import Paths

def plot_instance(pc, mask, bbox, instance_idx=0):
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

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot object points (subsampled for speed)
    if len(inst_points) > 2000:
        idx = np.random.choice(len(inst_points), 2000, replace=False)
        pts = inst_points[idx]
    else:
        pts = inst_points

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="blue", alpha=0.5, label="Points")

    # Draw Bounding Box
    for edge in edges:
        ax.plot(inst_bbox[list(edge), 0], inst_bbox[list(edge), 1], inst_bbox[list(edge), 2], 
                c="red", linewidth=2)

    ax.set_title(f"Instance {instance_idx} Visualization")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.close("all")
    
    
if __name__ == "__main__":
    data_path = Paths.data


    # 1. Load the data
    scene_id = "8b061a8a-9915-11ee-9103-bbb8eae05561"
    pc = np.load(os.path.join(data_path, scene_id, "pc.npy"))
    mask = np.load(os.path.join(data_path, scene_id, "mask.npy"))
    bbox = np.load(os.path.join(data_path, scene_id, "bbox3d.npy"))
    
    for instance_idx in range(mask.shape[0]):
        plot_instance(pc=pc, mask=mask, bbox=bbox, instance_idx=instance_idx)