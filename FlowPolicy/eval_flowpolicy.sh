#!/bin/bash

# Evaluate FlowPolicy on real robot dataset
# Usage: ./eval_flowpolicy.sh [checkpoint_path] [demo_idx]

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pfp_env

# Default paths
CHECKPOINT_PATH=${1:-"/home/carl_lab/Riad/graspbility/FlowPolicy/FlowPolicy/flowpolicy_checkpoints/best_flowpolicy_epoch_1.pth"}
DATASET_PATH="/home/carl_lab/Riad/graspbility/GraspSplats/new_background_grasp_data_low_depth_with_delta_agent_new.hdf5"
DEMO_IDX=${2:-""}

echo "Starting FlowPolicy evaluation..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_PATH"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    echo "Available checkpoints:"
    ls -la ./flowpolicy_checkpoints/*.pth 2>/dev/null || echo "No checkpoints found in ./flowpolicy_checkpoints/"
    exit 1
fi

# Run evaluation
if [ -n "$DEMO_IDX" ]; then
    echo "Evaluating single demo: $DEMO_IDX"
    python FlowPolicy/eval_flowpolicy.py \
        --checkpoint $CHECKPOINT_PATH \
        --dataset $DATASET_PATH \
        --demo_idx $DEMO_IDX \
        --device cuda
else
    echo "Evaluating multiple demos"
    python FlowPolicy/eval_flowpolicy.py \
        --checkpoint $CHECKPOINT_PATH \
        --dataset $DATASET_PATH \
        --num_demos 5 \
        --device cuda
fi

echo "Evaluation completed!"
