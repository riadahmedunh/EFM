# Enhanced Consistent Flow Matching Policy (EFM)

> **Enhanced Consistent Flow Matching Policy for Robust 3D Vision-Based Robot Manipulation**

This repository contains the implementation of **EFM** — an enhanced consistency flow matching framework for 3D vision-based robotic manipulation. EFM builds upon consistency flow matching principles and introduces several key improvements to the training and inference pipeline, resulting in more accurate and robust policy generation.

## ✨ Key Contributions

EFM introduces **three core enhancements** over standard consistency flow matching policies:

### 1. RK4 Integration for Inference
We replace single-step Euler integration with **4th-order Runge-Kutta (RK4)** integration during inference, enabling more accurate trajectory estimation from noise to action space:

$$z_{t+1} = z_t + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

### 2. Multi-Step Trajectory Consistency Loss
A **multi-step consistency training loss** ensures the learned velocity field produces globally consistent paths across the full trajectory — not just single time-step predictions. This improves the coherence of generated action sequences.

### 3. Velocity Regularization
A **velocity regularization term** enforces smoothness and global consistency of the learned velocity field, leading to more stable flow dynamics.

### Enhanced Training Objective

The total training loss combines these components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + 0.5 \cdot \mathcal{L}_{\text{multi-step}} + 0.5 \cdot \mathcal{L}_{\text{vel-reg}} + 0.1 \cdot \mathcal{L}_{\text{action-MSE}}$$

### Additional Features
- **Dual-View Point Cloud Encoding**: Processes two camera views (agentview + eye-in-hand) with separate PointNet encoders for richer 3D scene understanding.
- **Real-Robot Data Pipeline**: Tools to create Zarr datasets from HDF5 robot demonstrations with voxel grid downsampling for point clouds.
- **Multiple Action Normalizers**: Diffusion Policy-style, hybrid (pose + gripper), and enhanced (working movement + gripper) normalization strategies.
- **Exact Rectified Flow**: An alternative implementation using exact rectified flow matching.

## 🤖 Supported Setup

| Setup | Description |
|---|---|
| **Single-View** | AgentView point cloud only |
| **Dual-View** | AgentView + Eye-in-Hand point clouds with separate PointNet encoders |

# 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/riadahmedunh/EFM
cd EFM
```

### 2. Install PyTorch (CUDA 12.8)

```bash
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade
```

> Adjust the CUDA version in the URL to match your system (e.g., `cu118`, `cu121`).

### 3. Install dependencies

```bash
pip install zarr omegaconf termcolor einops hydra-core dill wandb tqdm
```

### 4. Install FlowPolicy

```bash
cd FlowPolicy && pip install -e . && cd ..
```

For the full environment setup (MuJoCo, mujoco-py, pytorch3d, etc.), see [install.md](install.md).

# 📚 Data

For real-robot tasks, you need to prepare your demonstration data as an **HDF5 file** and then convert it to **Zarr format** using the provided conversion scripts.

### Step 1: Prepare Your HDF5 File

Your HDF5 demonstration file must follow this structure:

```
demo.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions                    # (T, action_dim) float64 — robot actions per timestep
│   │   └── obs/
│   │       ├── agentview_rgb          # (T, 240, 320, 3) uint8 — RGB image from external camera
│   │       ├── agentview_rgb_depth    # (T, 240, 320) uint16 — depth image from external camera
│   │       ├── eye_in_hand_rgb        # (T, 240, 320, 3) uint8 — RGB image from wrist camera
│   │       ├── eye_in_hand_rgb_depth  # (T, 240, 320) uint16 — depth image from wrist camera
│   │       ├── joint_states           # (T, 7) float64 — 7-DOF arm joint positions
│   │       ├── ee_states              # (T, 16) float64 — end-effector 4×4 pose (flattened, column-major)
│   │       └── gripper_states         # (T, 1) or (T, 2) float64 — gripper finger positions
│   ├── demo_1/
│   │   ├── actions
│   │   └── obs/
│   │       └── ...
│   └── ...
```

Where `T` is the number of timesteps in each demonstration (can vary per demo).

**Field details:**

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `actions` | `(T, action_dim)` | float64 | Robot actions per timestep (e.g., 7D for position + rotation + gripper) |
| `agentview_rgb` | `(T, 240, 320, 3)` | uint8 | RGB image from the external (agent-view) camera |
| `agentview_rgb_depth` | `(T, 240, 320)` | uint16 | Depth image from the external camera (in millimeters, divide by 1000 for meters) |
| `eye_in_hand_rgb` | `(T, 240, 320, 3)` | uint8 | RGB image from the wrist-mounted (eye-in-hand) camera |
| `eye_in_hand_rgb_depth` | `(T, 240, 320)` | uint16 | Depth image from the wrist camera (in millimeters) |
| `joint_states` | `(T, 7)` | float64 | 7-DOF arm joint positions (radians) |
| `ee_states` | `(T, 16)` | float64 | End-effector 4×4 transformation matrix, flattened column-major |
| `gripper_states` | `(T, 1)` or `(T, 2)` | float64 | Gripper finger position(s). If 1D, symmetric gripper is assumed |

> **Camera intrinsics** used by the converter (defaults): `fx=320.0, fy=320.0, cx=160.0, cy=120.0`, image size `320×240`. Modify these in `HDF5ToZarrConverter.__init__()` if your cameras differ.

### Step 2: Convert HDF5 → Zarr

Use the conversion script to generate point clouds from RGB-D data and save everything in Zarr format:

```python
# Edit the paths in EFM/create_zarr_real_robot_faster.py, then run:
python EFM/create_zarr_real_robot_faster.py
```

Inside the script, set your paths:
```python
hdf5_path = "/path/to/your/demo.hdf5"
output_path = "/path/to/output/dataset.zarr"
max_points = 8192  # points per point cloud
```

The converter will:
1. Read each demo's RGB-D images from both cameras
2. Back-project depth to 3D point clouds (`[x, y, z, r, g, b]`)
3. Downsample via voxel grid sampling to `max_points` per cloud
4. Extract the 21D robot state vector from `joint_states`, `ee_states`, and `gripper_states`
5. Save everything into a Zarr file

### Zarr Output Format

The resulting Zarr dataset has the following structure:

```
dataset.zarr/
├── data/
│   ├── agentview_point_cloud       # (N_total, 8192, 6) float32
│   ├── eye_in_hand_point_cloud     # (N_total, 8192, 6) float32
│   ├── state                       # (N_total, 21) float32
│   └── action                      # (N_total, action_dim) float64
└── meta/
    ├── episode_ends                # (num_episodes,) int — cumulative timestep index where each episode ends
    └── attrs:                      # metadata (num_episodes, total_timesteps, shapes, config, etc.)
```

Where `N_total` is the sum of all timesteps across all demonstrations.

**Point cloud format** — each point has 6 channels: `[x, y, z, r, g, b]`
- `x, y, z` — 3D coordinates in meters (camera frame)
- `r, g, b` — RGB color values (0–255)
- Padded with zeros if fewer than 8192 valid points

**State vector** — 21 dimensions:

| Indices | Dim | Description |
|---|---|---|
| 0–6 | 7 | Joint positions (7-DOF arm) |
| 7–9 | 3 | End-effector XYZ position |
| 10–18 | 9 | End-effector rotation matrix (flattened) |
| 19–20 | 2 | Gripper finger positions |

**Episode ends** — cumulative indices marking where each episode ends. For example, if you have 3 demos of lengths 100, 150, 120, then `episode_ends = [100, 250, 370]`.

# 🛠️ Training

Results are logged via `wandb` — run `wandb login` before training to track results and videos.

Two training scripts are provided depending on your camera setup:

### Single-View (AgentView only)

Uses only the external camera point cloud. Edit the config inside the script to set your Zarr path and hyperparameters, then run:

```bash
python EFM/FlowPolicy/train_real_robot_flowpolicy.py
```

Key config fields to modify:
```python
'zarr_path': '/path/to/your/dataset.zarr',  # Your Zarr dataset
'horizon': 4,              # Action prediction horizon
'n_obs_steps': 2,          # Number of observation steps
'n_action_steps': 4,       # Number of action steps to execute
'batch_size': 64,
'num_epochs': 200,
```

Observation space: `point_cloud (8192, 6)` + `agent_pos (21,)` → Action space: `(7,)`

### Dual-View (AgentView + Eye-in-Hand)

Uses both the external camera and wrist camera point clouds, processed by **separate PointNet encoders** with concatenated features. Run:

```bash
python EFM/FlowPolicy/train_real_robot_flowpolicy_dual_view.py
```

Key config fields to modify:
```python
'zarr_path': '/path/to/your/dataset.zarr',  # Must contain both point clouds
'horizon': 4,
'n_obs_steps': 2,
'n_action_steps': 4,
'batch_size': 32,
'num_epochs': 400,
'demo_fraction': 1.0,      # Fraction of demos to use (1.0 = all)
```

Observation space: `point_cloud (8192, 6)` + `eye_in_hand_point_cloud (8192, 6)` + `agent_pos (21,)` → Action space: `(7,)`

### Training Features

Both real-robot training scripts include:
- **Enhanced CFM loss** with multi-step trajectory consistency and velocity regularization
- **RK4 integration** during inference for higher accuracy
- **Identity action normalizer** (raw actions, no normalization)
- **EMA model** for stable training
- **Automatic mixed precision** (AMP) for faster training
- **Gradient accumulation** (effective batch = `batch_size × gradient_accumulate_every`)
- **Per-batch and per-epoch loss logging** to CSV files in the checkpoint directory
- **Automatic checkpoint saving** and resume support

# 📁 Project Structure

```
.
├── FlowPolicy/
│   ├── train_real_robot_flowpolicy.py          # Single-view training script
│   ├── train_real_robot_flowpolicy_dual_view.py # Dual-view training script
│   ├── train.py                                # Base training workspace
│   ├── eval_flowpolicy.py                      # Evaluation script
│   ├── action_normalizer.py                    # DP-style action normalizer
│   ├── action_normalizer_hybrid.py             # Hybrid action normalizer
│   ├── action_normalizer_dp.py                 # Diffusion Policy action normalizer
│   ├── train_exact_rectified_flow.py           # Exact rectified flow training
│   └── flow_policy_3d/
│       ├── consistencyfm/   # Conditional flow matching implementation
│       ├── policy/          # FlowPolicy & enhanced variants
│       ├── model/           # ConditionalUnet1D, PointNet encoders
│       ├── dataset/         # Data loaders
│       ├── env/             # Environment wrappers
│       ├── env_runner/      # Rollout / evaluation runners
│       ├── config/          # Hydra configs for all tasks
│       ├── losses.py        # CFM loss functions
│       └── losses_rectified_flow.py
├── create_zarr_real_robot.py          # HDF5 → Zarr conversion
├── create_zarr_real_robot_faster.py   # HDF5 → Zarr conversion (optimized)
├── visualize_zarr_pointclouds.py      # Zarr point cloud visualization
├── scripts/                           # Shell scripts (train, eval, demo gen)
├── third_party/                       # Dependencies (gym, Metaworld, mujoco-py, pytorch3d, VRL3)
├── install.md                         # Installation instructions
└── README.md
```

# 🏷️ License

This repository is released under the MIT license.

# 🙏 Acknowledgement

Our code builds upon [FlowPolicy](https://github.com/zql-kk/FlowPolicy), [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), [Consistency Flow Matching](https://github.com/YangLing0818/consistency_flow_matching), [VRL3](https://github.com/microsoft/VRL3), and [Metaworld](https://github.com/Farama-Foundation/Metaworld). We thank the authors for their excellent works.



