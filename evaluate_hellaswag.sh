#!/bin/bash
#SBATCH -c 16                                  # Number of CPU cores
#SBATCH --mem=80gb                             # Memory allocation
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH -t 7-00:00                             # Maximum runtime
#SBATCH -N 1                                   # Number of nodes
#SBATCH -o output_%j.log                       # Output log file (%j = JobID)
#SBATCH -e error_%j.log                        # Error log file (%j = JobID)

models=(
    "utter-project/EuroLLM-1.7B-Instruct"
    "tuning/models/model_dpo_v3"  
    "tuning/models/model_dpo_v4"
    "yiksiu/EuroLLM-1.7B-DPO-v3"
    "yiksiu/EuroLLM-1.7B-DPO-v4"
)

data_folder="data/m_hellaswag/"
sample_size=5000

mkdir -p results/hellaswag/

for model in "${models[@]}"
do
    echo "-----------------------------------------"
    echo "Evaluating model: $model"
    echo "-----------------------------------------"
    
    safe_model_name=$(echo $model | tr '/' '_')
    
    log_file="results/hellaswag/${safe_model_name}_evaluation.log"
    
    python3 evaluate_hellaswag.py --model_name "$model" --data_folder "$data_folder" --sample_size "$sample_size" | tee "$log_file"
    
    echo "Evaluation for model $model completed. Results saved to $log_file"
    echo ""
done

echo "All evaluations completed. Check the 'results' directory for detailed logs."