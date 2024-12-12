#!/bin/bash

source venv/bin/activate

TOKENIZERS_PARALLELISM=false 

#"utter-project/EuroLLM-1.7B-Instruct" 
# MODEL_ID= "tuning/models/model_dpo_v4"
DATA_DIR="data/xsafety_benchmark/"
SAVE_DIR="results/dpo-4/"

python3 evaluate_model.py \
    --model-id "tuning/models/model_dpo_v4" \
    --data-dir $DATA_DIR \
    --save-dir $SAVE_DIR \
