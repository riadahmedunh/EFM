#!/bin/bash

# Train FlowPolicy on real robot dataset
# Usage: ./train_flowpolicy.sh [path_to_hdf5_file]

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robodiff

# Default data path (change this to your actual data path)
DATA_PATH=${1:-"/Riad/diffusion_policy/new_background_grasp_data_low_depth_with_delta_agent_new.hdf5"}

# Training parameters
BATCH_SIZE=16
NUM_EPOCHS=500
LEARNING_RATE=1e-4
N_OBS_STEPS=2
HORIZON=4
N_ACTION_STEPS=4
IMAGE_SIZE=84
OUTPUT_DIR="./flowpolicy_checkpoints"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting FlowPolicy training..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"

# Run training
python train_real_robot_flowpolicy.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --n_obs_steps $N_OBS_STEPS \
    --horizon $HORIZON \
    --n_action_steps $N_ACTION_STEPS \
    --image_size $IMAGE_SIZE \
    --val_split 0.2 \
    --device cuda

echo "Training completed!"
