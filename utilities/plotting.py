import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

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
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="blue", alpha=0.5, label="Points")

    # Draw Bounding Box
    for edge in edges:
        ax.plot(inst_bbox[list(edge), 0], inst_bbox[list(edge), 1], inst_bbox[list(edge), 2], 
                c="red", linewidth=2)

    # --- NEW CODE: Plot the Origin (Camera center) and Axes ---
    
    # 1. Plot the origin (0, 0, 0) as a large black star
    ax.scatter(0, 0, 0, color="black", s=200, marker="*", label="Origin (0,0,0)")

    # 2. Dynamically calculate axis length based on distance to the object
    # so the axes lines are visible but don't dwarf the object
    if len(pts) > 0:
        dist_to_obj = np.linalg.norm(np.mean(pts, axis=0))
        axis_len = dist_to_obj * 0.5 if dist_to_obj > 0 else 1.0
    else:
        axis_len = 1.0

    # 3. Draw X (Red), Y (Green), and Z (Blue) axes starting from origin
    ax.plot([0, axis_len], [0, 0], [0, 0], color="red", linewidth=3, label="X-axis")
    ax.plot([0, 0], [0, axis_len], [0, 0], color="green", linewidth=3, label="Y-axis")
    ax.plot([0, 0], [0, 0], [0, axis_len], color="blue", linewidth=3, label="Z-axis")

    # -----------------------------------------------------------

    ax.set_title(f"Instance {instance_idx} Visualization (with Origin)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    # Make sure 0,0,0 is actually visible in the plot limits
    all_x = np.append(pts[:, 0] if len(pts)>0 else [], [0, axis_len])
    all_y = np.append(pts[:, 1] if len(pts)>0 else [], [0, axis_len])
    all_z = np.append(pts[:, 2] if len(pts)>0 else [], [0, axis_len])
    
    # ax.set_xlim([np.min(all_x), np.max(all_x)])
    # ax.set_ylim([np.min(all_y), np.max(all_y)])
    # ax.set_zlim([np.min(all_z), np.max(all_z)])

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    return ax
    
    
if __name__ == "__main__":
    data_path = Paths.data

    # 1. Load the data
    scene_id = "8b061a8b-9915-11ee-9103-bbb8eae05561"
    pc = np.load(os.path.join(data_path, scene_id, "pc.npy"))
    mask = np.load(os.path.join(data_path, scene_id, "mask.npy"))
    bbox = np.load(os.path.join(data_path, scene_id, "bbox3d.npy"))
    
    ax = None
    for instance_idx in range(mask.shape[0]):
        ax = plot_instance(pc=pc, mask=mask, bbox=bbox, instance_idx=instance_idx, ax=ax)
    
    plt.show()
    plt.close("all")