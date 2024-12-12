# Final Project for CSCI 2470 at Brown University

This repository contains the official implementation of our final project titled Investigating the (In)Effectiveness of Multilingual Safety Alignment. It investigates the trade-offs between safety alignment and general capabilities in multilingual large language models (LLMs). It involves fine-tuning models using Direct Preference Optimization (DPO) and evaluating their performance across multiple languages and tasks.

---

## Repository Structure

- **`backbones/`**  
  Contains the core backbone models used in the project. Includes a script to dynamically load and initialize specific models based on input parameters (e.g., local models like Llama2).

- **`data/`**  
  Contains raw datasets for training and evaluation:
  - `m_hellaswag`: Dataset for general capability evaluation.
  - `safe_rlhf_train`: Sample subset for DPO safety training. Full dataset can be found at https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF. 
  - `translated_advbench`: Translated adversarial benchmark for testing.
  - The XSafety evaluation dataset can be found at https://github.com/Jarviswang94/Multilingual_safety_benchmark. 

- **`eval/`**  
  Contains scripts for evaluating the performance of models on various tasks, including safety and general capabilities.

- **`scripts/`**  
  Contains the scripts for key steps in the methodology, including:
  - Inference
  - Translation
  - Unsafe rate (ASR) calculation

- **`results/`**  
  Contains all experimental results, organized into subfolders:
  - `base_model/`: Results for the base, unaligned model.
  - `dpo-v1/`, `dpo-v2/`, `dpo-v3/` `dpo-v4/`: Results for DPO-trained models.
  - `hellaswag_results/`: Results from general capability evaluation.

- **`translation/`**  
  Implements translation tools for preparing multilingual datasets and analyzing responses.

- **`tuning/`**  
  Contains scripts and configurations for performing DPO training.

- **`utils/`**  
  Utility functions for data processing, prompt cleaning, JSON validation, partitioning data, and text processing (e.g., reducing repeated phrases).

---

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yiksiu-chan/csci2470_project.git
cd csci2470_project
```

### 2. DPO Tuning
```bash
cd tuning
sbatch dpo.sh
```
### 3. Evaluate models. 
```bash
sbatch evaluate.sh
sbatch evaluate_hellaswag.sh
```
