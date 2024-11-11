import argparse
import random
import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset
from trl import DPOConfig

from trainer import DPOTuningTrainer

import pandas as pd
from datasets import Dataset


def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_args():
    parser = argparse.ArgumentParser(description="DPO Model Training Script")

    parser.add_argument("--model_name", type=str, default="utter-project/EuroLLM-1.7B-Instruct", help="Pretrained model name")
    parser.add_argument("--seed", type=int, default=42, help="Set the seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for optimizer")
    parser.add_argument("--log_interval", type=int, default=1000, help="Steps between logging training status")
    parser.add_argument("--eval_interval", type=int, default=5000, help="Steps between evaluation/checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--dataset_name", type=str, default="PKU-Alignment/PKU-SafeRLHF", help="Dataset name from Hugging Face Datasets library")
    parser.add_argument("--tokenizer", type=str, default="utter-project/EuroLLM-1.7B-Instruct", help="Tokenizer")

    return parser.parse_args()


def get_safety_pairwise_data(data_file) -> Dataset:
    """
    Load a safety-paired dataset from a single JSON file, filter, and format it.

    Args:
        data_file (str): Path to the JSON file with prompt-response data.
        train_perc (float): Percentage of data to keep for training.
        sanity_check (bool): Whether to limit the dataset to a smaller sample for testing.

    Returns:
        Dataset: Hugging Face Dataset with columns 'prompt', 'chosen', and 'rejected'.
    """

    # Load data from JSON file
    # with open(data_file, "r") as f:
    #     data = json.load(f)

    data = load_dataset("PKU-Alignment/PKU-SafeRLHF", name='default')["train"]
    processed_data = []

    # Process each sample based on the given criteria
    for sample in data:
        prompt = sample["prompt"]

        # Selected categories
        # categories = (sample["response_0_harm_category"]["Physical Harm"] 
        #                 or sample["response_1_harm_category"]["Physical Harm"] 
        #                 or sample["response_0_harm_category"]["Insulting Behavior"] 
        #                 or sample["response_1_harm_category"]["Insulting Behavior"] 
        #                 or sample["response_0_harm_category"]["Discriminatory Behavior"] 
        #                 or sample["response_1_harm_category"]["Discriminatory Behavior"] 
        #                 or sample["response_0_harm_category"]["Violence"] 
        #                 or sample["response_1_harm_category"]["Violence"] 
        #                 or sample["response_0_harm_category"]["Psychological Harm"] 
        #                 or sample["response_1_harm_category"]["Psychological Harm"])

        if sample["is_response_0_safe"] == True and sample["is_response_1_safe"] == True:
            pass 
        # elif categories: 
        else: 
            if sample["is_response_0_safe"] == False and sample["is_response_1_safe"] == True:
                rejected = sample["response_0"]
            elif sample["is_response_0_safe"] == True and sample["is_response_1_safe"] == False:
                rejected = sample["response_1"]
            else:
                # If both are unsafe, choose based on severity level
                rejected = sample["response_0"] if sample["response_0_severity_level"] > sample["response_1_severity_level"] else sample["response_1"]

            # Define the chosen response for all cases
            chosen = "Sorry, but I cannot assist with this request."
            
            processed_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        # else:
        #     pass

    print(f"Size of finetuning dataset: {len(processed_data)}")
    dataset = Dataset.from_pandas(pd.DataFrame(processed_data))

    return dataset

if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    train_dataset = get_safety_pairwise_data(data_file=args.dataset_name)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if hasattr(model.config, "use_flash_attn"):
        model.config.use_flash_attn = True

    training_args = DPOConfig(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        # max_grad_norm=10,
        logging_steps=args.log_interval,
        save_steps=args.eval_interval,
        save_total_limit=5,
    )

    # Initialize DPO Trainer
    dpo_trainer = DPOTuningTrainer(
        model_name=args.model_name,
        args=training_args,
        train_dataset=train_dataset,
        use_wandb=args.use_wandb,
        checkpoint_dir=args.checkpoint_dir
    )

    if args.use_wandb:
        wandb.login(key="")
        wandb.init(
            project="eurollm_dpo",
            name=f"{args.model_name}_dpo_training_seed{args.seed}",
            config=vars(args)
        )

    # Train the model
    dpo_trainer.train()

    dpo_trainer.save_pretrained("model_dpo_v3/")

    model = AutoModelForCausalLM.from_pretrained("model_dpo_v3/")
    model.push_to_hub("yiksiu/EuroLLM-1.7B-DPO-v3")