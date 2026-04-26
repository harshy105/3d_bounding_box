import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import Tensor

from config import Paths, DataLoaderConfig
from data_loader import LMDBInstanceDataset
from network.trainer import TrainerLitModule
from utilities.utils import reconstruct_unique_box


def draw_axes(center, dims, rot6d, color, ax):
    w, h, _ = dims # Extract Width and Height for scaling
    
    v1_raw = rot6d[:3]
    v2_raw = rot6d[3:]
    
    # Gram-Schmidt Orthogonalization (numpy equivalent)
    v1 = v1_raw / (np.linalg.norm(v1_raw) + 1e-8)
    v2_proj = v2_raw - np.dot(v2_raw, v1) * v1
    v2 = v2_proj / (np.linalg.norm(v2_proj) + 1e-8)
    
    # Scale axes by half the dimension size to visually match the box boundaries
    x_scaled = v1 * (w / 2.0)
    y_scaled = v2 * (h / 2.0)
    
    # Draw X-axis (solid)
    ax.quiver(center[0], center[1], center[2], 
              x_scaled[0], x_scaled[1], x_scaled[2], 
              color=color, arrow_length_ratio=0.15, linewidth=2)
    # Draw Y-axis (dotted to distinguish from X)
    ax.quiver(center[0], center[1], center[2], 
              y_scaled[0], y_scaled[1], y_scaled[2], 
              color=color, arrow_length_ratio=0.15, linewidth=2, linestyle=":")
        

def visualize_eval_sample(pc_tensor: Tensor, 
                          gt_box: np.ndarray, pred_box: np.ndarray, 
                          gt_c: np.ndarray, pred_c: np.ndarray,
                          gt_s: np.ndarray, pred_s: np.ndarray,
                          gt_rot6d: np.ndarray, pred_rot6d: np.ndarray,
                          sample_key: str):
    """
    Plots the Colored Point Cloud, GT Box (Green), Pred Box (Red), Centers, and Axes.
    """
    fig = plt.figure(figsize=(10, 8))
    
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    ax_3d.set_title(f"Sample: {sample_key}\nColored Point Cloud & Bounding Boxes")
    
    pc_pts = pc_tensor.cpu().numpy()
    
    # Filter out zero-padded rows based on XYZ coordinates
    non_zero_mask = np.any(pc_pts[:, :3] != 0, axis=1)
    valid_pts = pc_pts[non_zero_mask]
    
    # Subsample points for speed
    if len(valid_pts) > 2000:
        idx = np.random.choice(len(valid_pts), 2000, replace=False)
        pts = valid_pts[idx]
    else:
        pts = valid_pts

    if len(pts) > 0:
        xyz = pts[:, :3]
        colors = pts[:, 3:]
        assert colors.max() <= 1.0
        
        ax_3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=5, c=colors, alpha=0.8, label="Point Cloud")

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4), # Top
        (0, 4), (1, 5), (2, 6), (3, 7)  # Pillars
    ]

    # Draw Ground Truth Box (Green)
    for i, edge in enumerate(edges):
        ax_3d.plot(gt_box[list(edge), 0], gt_box[list(edge), 1], gt_box[list(edge), 2], 
                   c="green", linewidth=2, label="Ground Truth" if i == 0 else "")

    # Draw Predicted Box (Red)
    for i, edge in enumerate(edges):
        ax_3d.plot(pred_box[list(edge), 0], pred_box[list(edge), 1], pred_box[list(edge), 2], 
                   c="red", linewidth=2, linestyle="--", label="Prediction" if i == 0 else "")

    # --- Visualize Centers ---
    ax_3d.scatter(gt_c[0], gt_c[1], gt_c[2], c="green", s=150, marker="*", label="GT Center")
    ax_3d.scatter(pred_c[0], pred_c[1], pred_c[2], c="red", s=150, marker="*", label="Pred Center")

    # Draw Axes
    draw_axes(gt_c, gt_s, gt_rot6d, color="green", ax=ax_3d)
    draw_axes(pred_c, pred_s, pred_rot6d, color="red", ax=ax_3d)

    # --- Overlay Exact Dimension Values ---
    info_text = (f"GT Dims (W,H,L): {gt_s[0]:.2f}, {gt_s[1]:.2f}, {gt_s[2]:.2f}\n"
                 f"PR Dims (W,H,L): {pred_s[0]:.2f}, {pred_s[1]:.2f}, {pred_s[2]:.2f}")
    ax_3d.text2D(0.05, 0.95, info_text, transform=ax_3d.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
    ax_3d.view_init(elev=-90, azim=-90)
    ax_3d.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.show()
    plt.close("all")


def evaluate_model(checkpoint_path: str, split: str = "test", num_vis_samples: int = 5):
    # 1. Setup Device and Configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_cfg = DataLoaderConfig()
    
    # Disable augmentation and shuffling for pure evaluation
    dl_cfg.apply_aug = False 
    dl_cfg.shuffle = False
    
    # 2. Load Dataset
    dataset_path = os.path.join(Paths.parsed_data, split)
    dataset = LMDBInstanceDataset(dataset_path, data_loader_config=dl_cfg, apply_aug=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dl_cfg.batch_size, shuffle=False)
    
    print(f"Loaded {len(dataset)} samples for {split} evaluation.")
    
    # 3. Load Model from Checkpoint
    from config import NetConfig, TrainConfig
    print(f"Loading model from {checkpoint_path}...")
    model = TrainerLitModule.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    onnx_path =  os.path.splitext(checkpoint_path)[0] + ".onnx"
    print(f"Exporting model to ONNX: {onnx_path}")
    
    # Create a dummy input to trace the model's graph. 
    num_channels = 3 + model.model.input_feature_dim
    dummy_input = torch.randn(1, dl_cfg.max_number_pc_pts, num_channels, device=device)
    
    model.to_onnx(
        onnx_path,
        dummy_input,
        export_params=True,
        opset_version=14,
        input_names=["pc_pts"],
        output_names=["center", "size", "rot_6d"],
        dynamic_axes={
            "pc_pts": {0: "batch_size", 1: "num_points"},
            "center": {0: "batch_size"},
            "size":   {0: "batch_size"},
            "rot_6d": {0: "batch_size"}
        }
    )
    print(f"ONNX export successfully saved alongside the checkpoint!")

    # Metrics trackers
    total_center_err = 0.0
    total_dim_err = 0.0
    total_corner_err = 0.0
    num_samples = 0
    vis_count = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move data to device
            pc_pts = batch["pc_pts"].to(device)
            targ_c = batch["bbox_center"].to(device)
            targ_s = batch["bbox_dims"].to(device)
            targ_rot6d = batch["bbox_rot_6d"].to(device)
            targ_corners = batch["bbox_3d"].to(device)
            
            # Forward Pass
            pred_c, pred_s, pred_rot6d = model(pc_pts)
            
            # Reconstruct Predicted Corners
            if pred_rot6d is not None and pred_s is not None:
                pred_corners = reconstruct_unique_box(pred_c, pred_s, pred_rot6d)
            else:
                pred_corners = pred_c.unsqueeze(1).repeat(1, 8, 1)
            
            # Setting sides and angles to zero if not predicted
            pred_s = torch.zeros_like(pred_c) if pred_s is None else pred_s
            pred_rot6d = torch.concat(
                [
                    torch.zeros_like(pred_c),
                    torch.zeros_like(pred_c)
                ], dim = -1,
            ) if pred_rot6d is None else pred_rot6d            
            
            # --- Quantitative Metrics ---
            # 1. Center Error (L2 Distance in meters)
            center_dist = torch.norm(pred_c - targ_c, dim=1)
            
            # 2. Dimension Error (Mean Absolute Error across W, H, L)
            dim_err = torch.abs(pred_s - targ_s).mean(dim=1)
            
            # 3. Corner Error (Mean L2 distance between corresponding corners)
            corner_dist = torch.norm(pred_corners - targ_corners, dim=2).mean(dim=1)
            
            # Accumulate
            batch_size = pc_pts.shape[0]
            total_center_err += center_dist.sum().item()
            total_dim_err += dim_err.sum().item()
            total_corner_err += corner_dist.sum().item()
            num_samples += batch_size
            
            # --- Qualitative Visualization ---
            if vis_count < num_vis_samples:
                for i in range(batch_size):
                    if vis_count >= num_vis_samples:
                        break
                    
                    visualize_eval_sample(
                        pc_tensor=batch["pc_pts"][i],
                        gt_box=targ_corners[i].cpu().numpy(),
                        pred_box=pred_corners[i].cpu().numpy(),
                        gt_c=targ_c[i].cpu().numpy(),
                        pred_c=pred_c[i].cpu().numpy(),
                        gt_s=targ_s[i].cpu().numpy(),
                        pred_s=pred_s[i].cpu().numpy(),
                        gt_rot6d=targ_rot6d[i].cpu().numpy(),
                        pred_rot6d=pred_rot6d[i].cpu().numpy(),
                        sample_key=batch["key"][i]
                    )
                    vis_count += 1

    avg_center_err = total_center_err / num_samples
    avg_dim_err = total_dim_err / num_samples
    avg_corner_err = total_corner_err / num_samples
    
    print("\n" + "="*40)
    print(" EVALUATION RESULTS ")
    print("="*40)
    print(f"Split Analyzed : {split.upper()}")
    print(f"Total Samples  : {num_samples}")
    print("-" * 40)
    print(f"Average Center Error   : {avg_center_err:.4f} meters")
    print(f"Average Dimension Error: {avg_dim_err:.4f} meters")
    print(f"Average Corner Error   : {avg_corner_err:.4f} meters")
    print("="*40)

if __name__ == "__main__":
    ckpt_name = "20260426_150126_base-epoch=21-val_loss=0.1847.ckpt"
    CKPT_PATH = os.path.join(Paths.ckpts, ckpt_name.split("-")[0], ckpt_name)
    
    evaluate_model(
        checkpoint_path=CKPT_PATH, 
        split="test", 
        num_vis_samples=0,
    )