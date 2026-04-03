#!/bin/bash
# UniDTI multimodal training script
# Author  : Ethan Huang
# File    : run.sh
# Time    : 2026/3/25

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.:$PYTHONPATH

# Parameters
DATA_NAME="biosnap" # DAVIS #bindingdb #biosnap #glass
SPLIT="warm" # warm
CONTACT_MAP_DIR="../prot-gnn-data/glass/contact_maps_p2rank"

echo "Starting multimodal UniDTI training..."
echo "Dataset: $DATA_NAME"
echo "Split: $SPLIT"
echo "Contact map directory: $CONTACT_MAP_DIR"
echo "========================================"

python main.py \
  --data $DATA_NAME \
  --split $SPLIT \
  --contact_map_dir $CONTACT_MAP_DIR

echo "Training completed!"
