#!/bin/bash

source ../venv/bin/activate

# File paths
DATA_DIR="../data/xsafety_small/"
SAVE_DIR="../results/dpo-3/"

# Model parameters
MODEL_ID="../tuning/model_dpo_v3"
# "yiksiu/EuroLLM-1.7B-DPO-v2"
MAX_TOKENS=256
TEMPERATURE=0.1
TOP_P=0

python3 inference.py \
    --model-id $MODEL_ID \
    --data-dir $DATA_DIR \
    --save-dir $SAVE_DIR \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top-p $TOP_P