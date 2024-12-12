import os
import json
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def load_validation_data(folder_path):
    """
    Load all validation JSON files from the specified folder.
    Returns a dictionary mapping language codes to a dictionary of ind: example.
    """
    validation_files = [f for f in os.listdir(folder_path) if f.endswith('_validation.json')]
    data = {}
    
    for file in validation_files:
        language_code = file.split('_')[0]  
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                examples = json.load(f)
                ind_to_example = {example['ind']: example for example in examples if 'ind' in example}
                data[language_code] = ind_to_example
                print(f"Loaded {len(ind_to_example)} examples for language: {language_code}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file}: {e}")
    
    return data

def select_common_inds(data, sample_size=100):
    """
    Select a common set of 'ind' values present across all languages.
    Returns a list of selected 'ind's.
    """
    ind_sets = [set(examples.keys()) for examples in data.values()]
    common_inds = set.intersection(*ind_sets)
    print(f"Total common 'ind's across all languages: {len(common_inds)}")
    
    if len(common_inds) < sample_size:
        raise ValueError(f"Not enough common samples. Required: {sample_size}, Available: {len(common_inds)}")
    
    selected_inds = random.sample(common_inds, sample_size)
    print(f"Selected {len(selected_inds)} common 'ind's for evaluation.")
    
    return selected_inds

def compute_accuracy(predictions, references):
    """
    Compute accuracy given predictions and references.
    """
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(references)
    return correct / total if total > 0 else 0

def evaluate_model_on_language(model, tokenizer, samples, device):
    """
    Evaluate the model on a list of samples for a single language.
    Returns the accuracy.
    """
    predictions = []
    references = []
    
    for sample in tqdm(samples, desc="Evaluating"):
        try:
            context = sample["ctx"]
            endings_str = sample["endings"]
            label = sample["label"]
            
            # ensure the label is int
            label_int = int(label)
            
            endings = endings_str
            
            # ensure endings list is a list with at least label+1 elements
            if not isinstance(endings, list):
                print(f"Endings is not a list for sample id {sample.get('id', 'N/A')}. Skipping.")
                continue
            if label_int < 0 or label_int >= len(endings):
                print(f"Label {label_int} out of bounds for sample id {sample.get('id', 'N/A')}. Skipping.")
                continue
            
            # construct prompts for inference
            scores = []
            for choice in endings:
                prompt = f"Context: {context}\nEnding: {choice}\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]
                
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                # compute and sum log probs
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                selected_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                total_log_prob = selected_log_probs.sum().item()
                scores.append(total_log_prob)
            
            pred_label = scores.index(max(scores))
            predictions.append(pred_label)
            references.append(label_int)
        
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for sample id {sample.get('id', 'N/A')}: {e}. Skipping.")
            continue
        except ValueError as e:
            print(f"Value error for sample id {sample.get('id', 'N/A')}: {e}. Skipping.")
            continue
        except Exception as e:
            print(f"Unexpected error for sample id {sample.get('id', 'N/A')}: {e}. Skipping.")
            continue
    
    accuracy = compute_accuracy(predictions, references)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Multilingual Model Evaluation Script")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the pretrained model to evaluate (e.g., 'xlm-roberta-base')"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/m_hellaswag/",
        help="Path to the folder containing validation JSON files"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=5000,
        help="Number of common samples to evaluate"
    )
    args = parser.parse_args()
    
    model_name = args.model_name
    folder_path = args.data_folder
    sample_size = args.sample_size
    
    random.seed(42)
    torch.manual_seed(42)
    
    data = load_validation_data(folder_path)
    
    if not data:
        print("No validation data loaded. Exiting.")
        return
    
    try:
        selected_inds = select_common_inds(data, sample_size=sample_size)
    except ValueError as e:
        print(e)
        return
    
    language_samples = {}
    for lang, examples in data.items():
        samples = []
        missing_inds = 0
        for ind in selected_inds:
            if ind in examples:
                samples.append(examples[ind])
            else:
                missing_inds += 1
        if missing_inds > 0:
            print(f"Language {lang}: Missing {missing_inds} samples out of {len(selected_inds)}.")
        language_samples[lang] = samples
        print(f"Language {lang}: {len(samples)} samples prepared for evaluation.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    results = {}
    for lang, samples in language_samples.items():
        print(f"\nEvaluating language: {lang} with {len(samples)} samples")
        accuracy = evaluate_model_on_language(model, tokenizer, samples, device)
        results[lang] = accuracy
        print(f"Accuracy for {lang}: {accuracy:.4f}")
    
    print("\nFinal Results:")
    for lang, acc in results.items():
        print(f"Language: {lang} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

