#!/bin/bash

USE_WANDB="--use_wandb"
CHECKPOINT_DIR="checkpoints/"
# export CUDA_VISIBLE_DEVICES=0,1

# source ../venv/bin/activate

python3 dpo.py \
    $USE_WANDB \
    --checkpoint_dir $CHECKPOINT_DIR