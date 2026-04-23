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

def visualize_eval_sample(img_tensor: Tensor, pc_tensor: Tensor, 
                          gt_box: np.ndarray, pred_box: np.ndarray, sample_key: str):
    """
    Custom visualization for evaluation.
    Plots the RGB image with mask on the left.
    Plots the Point Cloud, GT Box (Green), and Pred Box (Red) on the right.
    """
    fig = plt.figure(figsize=(14, 6))
    
    # --- Left: RGB + Mask ---
    ax_img = fig.add_subplot(1, 2, 1)
    img_display = img_tensor[:3].permute(1, 2, 0).cpu().numpy()
    ax_img.imshow(img_display)
    
    mask_display = img_tensor[3].cpu().numpy()
    ax_img.imshow(np.ma.masked_where(mask_display == 0, mask_display), cmap='gray_r', vmin=0, vmax=1, alpha=0.8)
    ax_img.set_title(f"Sample: {sample_key}\nRGB + Mask")
    ax_img.axis("off")
    
    # --- Right: 3D Point Cloud & Boxes ---
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    pc_pts = pc_tensor.cpu().numpy()
    
    # Subsample points for speed
    non_zero_mask = np.any(pc_pts != 0, axis=1)
    valid_pts = pc_pts[non_zero_mask]
    if len(valid_pts) > 2000:
        idx = np.random.choice(len(valid_pts), 2000, replace=False)
        pts = valid_pts[idx]
    else:
        pts = valid_pts

    if len(pts) > 0:
        ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="blue", alpha=0.3, label="Point Cloud")

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

    ax_3d.set_xlabel("X"); ax_3d.set_ylabel("Y"); ax_3d.set_zlabel("Z")
    ax_3d.view_init(elev=-90, azim=-90)
    ax_3d.legend(loc="upper right")
    
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
    torch.serialization.add_safe_globals([NetConfig, TrainConfig])
    print(f"Loading model from {checkpoint_path}...")
    model = TrainerLitModule.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

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
            targ_corners = batch["bbox_3d"].to(device)
            
            # Forward Pass
            pred_c, pred_s, pred_rot6d = model(pc_pts)
            
            # Reconstruct Predicted Corners
            pred_corners = reconstruct_unique_box(pred_c, pred_s, pred_rot6d)
            
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
                        img_tensor=batch["img_crop"][i],
                        pc_tensor=batch["pc_pts"][i],
                        gt_box=targ_corners[i].cpu().numpy(),
                        pred_box=pred_corners[i].cpu().numpy(),
                        sample_key=batch["key"][i]
                    )
                    vis_count += 1

    # --- Final Output ---
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
    ckpt_name = "20260423_183658_base-epoch=48-val_loss=0.2153.ckpt"
    CKPT_PATH = os.path.join(Paths.ckpts, ckpt_name.split("-")[0], ckpt_name)
    
    evaluate_model(
        checkpoint_path=CKPT_PATH, 
        split="test",            # Change to "val" if you just want to check validation
        num_vis_samples=5        # Number of side-by-side plots to generate
    )