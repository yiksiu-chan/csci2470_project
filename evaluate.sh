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

source venv/bin/activate

TOKENIZERS_PARALLELISM=false 

MODEL_ID="utter-project/EuroLLM-1.7B-Instruct" #"tuning/model_dpo_v3"
DATA_DIR="data/xsafety_benchmark/"
SAVE_DIR="results/"

python3 evaluate_model.py \
    --model-id $MODEL_ID \
    --data-dir $DATA_DIR \
    --save-dir $SAVE_DIR \
