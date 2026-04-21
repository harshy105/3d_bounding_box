import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.image as mpimg

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

    # --- Plot the Origin (Camera center) and Axes ---
    
    #  Plot the origin (0, 0, 0) as a large black star
    ax.scatter(0, 0, 0, color="black", s=200, marker="*", label="Origin (0,0,0)")

    # 2. Dynamically calculate axis length based on distance to the object
    # so the axes lines are visible but don't dwarf the object
    if len(pts) > 0:
        dist_to_obj = np.linalg.norm(np.mean(pts, axis=0))
        axis_len = dist_to_obj * 0.5 if dist_to_obj > 0 else 1.0
    else:
        axis_len = 1.0

    #  Draw X (Red), Y (Green), and Z (Blue) axes starting from origin
    ax.plot([0, axis_len], [0, 0], [0, 0], color="red", linewidth=3, label="X-axis")
    ax.plot([0, 0], [0, axis_len], [0, 0], color="green", linewidth=3, label="Y-axis")
    ax.plot([0, 0], [0, 0], [0, axis_len], color="blue", linewidth=3, label="Z-axis")

    ax.set_title(f"Instance {instance_idx} Visualization (with Origin)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    return ax


def plot_rgb(rgb_path: str):
    """Plots the RGB image alone."""
    img = mpimg.imread(rgb_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_title("Original RGB Image")
    ax.axis("off")
    plt.show()
    

def compare_projection_and_rgb(pc: np.array, mask: np.array, rgb_path: str):
    """Plots the RGB image alongside a 2D projection (Z=0) of the point cloud."""
    img = mpimg.imread(rgb_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: RGB Image
    axes[0].imshow(img)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Right: 2D Point Cloud Projection (X vs Y)
    x, y = pc[0].flatten(), pc[1].flatten()
    num_instances = mask.shape[0]
    cmap = plt.get_cmap("tab10")

    # Plot each instance with a different color
    for i in range(num_instances):
        inst_mask = mask[i].flatten()
        inst_x = x[inst_mask > 0]
        inst_y = y[inst_mask > 0]
        
        color = cmap(i % 10)
        if len(inst_x) > 0:
            axes[1].scatter(inst_x, inst_y, s=1, color=color, label=f"Instance {i}", alpha=0.6)

    axes[1].set_title("2D Projection of Point Cloud (Z=0)")
    axes[1].set_xlabel("X-axis")
    axes[1].set_ylabel("Y-axis")
    
    # Invert Y axis because image coordinates (Y down) 
    # usually oppose standard 3D plot coordinates (Y up).
    axes[1].invert_yaxis()
    
    axes[1].legend(loc="upper right")
    plt.tight_layout()
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
    
    # Compare the 2D projection and the RGB image side-by-side
    compare_projection_and_rgb(pc=pc, mask=mask, rgb_path=rgb_path)

    # (Optional) Plot 3D instances
    ax = None
    for instance_idx in range(mask.shape[0]):
        ax = plot_instance(pc=pc, mask=mask, bbox=bbox, instance_idx=instance_idx, ax=ax)
    plt.show()
    plt.close("all")