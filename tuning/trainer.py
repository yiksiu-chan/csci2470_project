import os
import torch
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer

class DPOTuningTrainer:
    def __init__(self, model_name, args, train_dataset, eval_dataset=None, batch_size=4, log_interval=100, eval_interval=500, use_wandb=False, device='cuda', checkpoint_dir="checkpoints"):
        """
        Args:
            model_name (str): The name of the Hugging Face model.
            args (TrainingArguments): The Hugging Face training arguments.
            train_dataset (Dataset): The training dataset.
            eval_dataset (Dataset, optional): The evaluation dataset.
            batch_size (int): Batch size for training and evaluation.
            log_interval (int): Steps between logging intervals.
            eval_interval (int): Steps between evaluation intervals.
            use_wandb (bool): Whether to log metrics to Weights & Biases.
            device (str): The device to train on ('cuda' or 'cpu').
            checkpoint_dir (str): Directory where model checkpoints will be saved.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.use_wandb = use_wandb
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.trainer = DPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.args
        )
        
    def train(self):
        """Run the training loop."""
        self.trainer.train()

        if self.use_wandb:
            wandb.finish()