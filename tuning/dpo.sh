#!/bin/bash
#SBATCH -c 16                                  # Number of CPU cores
#SBATCH --mem=80gb                             # Memory allocation
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH --exclude tea,frappe,boba,latte        # Specific node
#SBATCH -t 7-00:00                             # Maximum runtime
#SBATCH -p healthyml                           # Partition
#SBATCH -q healthyml-main                      # QoS
#SBATCH -N 1                                   # Number of nodes
#SBATCH -o output_%j.log                       # Output log file (%j = JobID)
#SBATCH -e error_%j.log                        # Error log file (%j = JobID)


USE_WANDB="--use_wandb"
CHECKPOINT_DIR="checkpoints_v2/"
export CUDA_VISIBLE_DEVICES=0,1

source ../venv/bin/activate

python3 dpo.py \
    $USE_WANDB \
    --checkpoint_dir $CHECKPOINT_DIR