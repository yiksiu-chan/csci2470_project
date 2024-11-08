import os
import json
import argparse

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer

def run_inference(prompt, model, tokenizer, max_tokens=256, temperature=0.1, top_p=0):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Follow the model's specific generation format
def generate_prompt(query, system_text="<|im_start|>system\n<|im_end|>\n<|im_start|>user\n", assistant_start="<|im_start|>assistant\n"):
    return f"{system_text}{query}\n<|im_end|>\n{assistant_start}"

def save_results(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Process files and perform inference
def process_files(folder_path, save_dir, model, tokenizer, max_tokens=256, file_prefix="xsafety_harmful_", file_extension=".txt"):
    result = {}
    
    for filename in os.listdir(folder_path):
        if filename.startswith(file_prefix) and filename.endswith(file_extension):
            language_code = filename[-6:-4]  # Extract language code
            file_path = os.path.join(folder_path, filename)
            save_path = os.path.join(save_dir, f"responses_{language_code}.json")
            
            # Load queries from file
            with open(file_path, 'r', encoding='utf-8') as file:
                queries = [line.strip() for line in file]

                queries = queries[:5]

            qa_pairs = load_results(save_path) or {}
            
            if not qa_pairs:  # Only process if responses are not already saved
                for query in tqdm(queries, ascii=True, leave=False):
                    prompt = generate_prompt(query)
                    response = run_inference(prompt, model, tokenizer, max_tokens=max_tokens)
                    print(response)
                    qa_pairs[query] = response

                save_results(qa_pairs, save_path)

            result[language_code] = [{"query": query, "response": qa_pairs.get(query, "")} for query in queries]

    return result


def main():
    parser = argparse.ArgumentParser(description="Run language model inference on text files in a specified directory.")
    parser.add_argument("--model-id", type=str, required=True, help="Model identifier for loading model and tokenizer.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the folder containing the text files.")
    parser.add_argument("--save-dir", type=str, default="results/", help="Path to the folder to save the results.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens for model output.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for inference.")
    parser.add_argument("--top-p", type=float, default=0, help="Top-p sampling value for inference.")
    cmd_args = parser.parse_args()

    os.makedirs(cmd_args.save_dir, exist_ok=True)
    if not os.path.isdir(cmd_args.data_dir):
        raise FileNotFoundError(f"The specified data directory '{cmd_args.data_dir}' does not exist.")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cmd_args.model_id)

    # Process data and save results
    output = process_files(
        folder_path=cmd_args.data_dir,
        save_dir=cmd_args.save_dir,
        model=model,
        tokenizer=tokenizer,
        max_tokens=cmd_args.max_tokens,
    )
    
    print("Inference complete. Results saved to:", cmd_args.save_dir)

if __name__ == "__main__":
    main()
