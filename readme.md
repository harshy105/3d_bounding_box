# 3D Bounding Box Prediction Pipeline

**Tech Stack:** PyTorch, PyTorch Lightning, ONNX, PointNet++  
**Author:** Harsh  

> ⚠️ **DATASET NOTICE:** Due to strict privacy and licensing constraints, the dataset (RGB, ground truth 3D bounding boxes, point clouds, and instance segmentation masks) used to train and evaluate this model **cannot be made public**. The code is provided for demonstration, architectural review, and structural reference.

---

## 1. Project Overview & Architecture

This repository contains a lightweight end-to-end deep learning pipeline for predicting full 3D bounding boxes for highly occluded objects. Because classical methods (like PCA) fail when geometry is missing, this project utilizes an amodal, learning-based instance regression approach. ⚠️ The approach still lack perfect bounding box prediction capabilities, nonetheless it does cover some of the important aspect of the 3D Bounding Boxes problem. 

### 1.1 Data Formulation
* **Input Representation:** Fused **(N, 6)** point clouds (XYZ + normalized RGB) cached in LMDB for high-speed I/O.
* **Canonical Targeting:** All ground truth boxes are reordered into a **unique canonical representation** prior to training to prevent ambiguous loss signals.
* **Instance Isolation:** Instead of searching the whole scene, the pipeline uses provided instance masks to isolate objects, simplifying the task to pure geometry regression.

### 1.2 Streamlined Instance Regression Network
* **Backbone:** A smaller version of the **PointNet++** backbone (adapted from VoteNet) extracts global features. It consists of three Set Abstraction (SA) layers.
* **Optimization:** Complex Voting and Proposal modules were removed. Since the input is already a cropped instance, **global pooling** of the `sa3` seed features proved highly efficient.
* **Rotation Strategy:** Uses the **6D Continuous Rotation Representation** ([Zhou et al., 2019](https://arxiv.org/abs/1812.07035)). Instead of regressing Euler angles or Quaternions (which suffer from discontinuities), the network predicts two continuous 3D vectors that are mapped to a perfect 3x3 rotation matrix via Gram-Schmidt orthogonalization.
* **Output (12-DoF):** Center (3), Dimensions (3), and 6D Rotation (6).

### 1.3 Loss Formulation
* **Center & Dimensions:** Huber Loss (Smooth L1).
* **Rotation:** Mean Squared Error (MSE) on the raw 6D vectors, providing a smooth, bounded optimization landscape.
* **Corner Loss:** Inspired by *Frustum-PointNet*, Huber loss is applied to the 8 physical 3D corners. This acts as a critical coupling term, forcing the network to align its center, size, and rotation predictions simultaneously.

### 1.4 Training & Deployment
* **Training Loop:** Managed via **PyTorch Lightning** with a `ReduceLROnPlateau` scheduler and a custom Early Stopping trigger based on minimum learning rate thresholds.
* **Inference Optimization (ONNX):** The evaluation script (`eval.py`) features automated graph tracing to export the `.ckpt` to **ONNX format**. It utilizes `dynamic_axes` for `batch_size` and `num_points`, ensuring the graph accepts varying point cloud densities dynamically for TensorRT deployment.

---

## 2. Environment Setup

This project requires a specific 3D computer vision environment to compile and install the **PointNet++** custom CUDA operators on WSL/Linux using Conda.

### 2.1 Create and Activate Environment
Ensure you have the `environment.yml` file in your current directory. This file manages the specific versions of the CUDA Toolkit (11.8) and the C++ compilers (GCC 11) required for compatibility.

```bash
# Create the environment from the YAML file
conda env create -f environment.yml

# Activate the new environment
conda activate 3d_bb
```

### 2.2 Configure Build Environment
To build custom CUDA extensions, the compiler needs to know exactly where the CUDA headers and binaries are located within your Conda environment.

```bash
# Set environment variables for the compiler
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/include:$CPATH
```

### 2.3 Clone and Fix PointNet2 Operators
The original repository uses deprecated PyTorch syntax (AT_CHECK) which causes compilation failures on modern PyTorch versions (2.0+). We apply a suppression fix using sed to update the source code to use TORCH_CHECK.

```bash
# Clone the repository
cd ~/
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git

# Navigate to the operator library
cd ~/Pointnet2_PyTorch/pointnet2_ops_lib

# Apply "Suppression Stuff" - Update old AT_CHECK syntax to modern TORCH_CHECK
sed -i 's/AT_CHECK/TORCH_CHECK/g' pointnet2_ops/_ext-src/src/*.cpp
sed -i 's/AT_CHECK/TORCH_CHECK/g' pointnet2_ops/_ext-src/src/*.cu

```
### 2.4. Compile and Install
Build the custom C++ and CUDA kernels. We disable build isolation to ensure the installation uses the PyTorch and CUDA versions already present in the Conda environment.

```bash
# Final Build and installation in editable mode
pip install --no-build-isolation -e .
```
### 2.5 Return to Main Project
Once the installation is successful, move back to your original project root to begin training or inference.

```bash
# Navigate back to the main directory
cd ../..
```
## Verification
To verify that the operators were installed correctly and can interface with your GPU, run:

```bash
python -c "import pointnet2_ops; print('PointNet2 Ops successfully installed')"
```