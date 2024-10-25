import os
import json
import argparse

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_id = "utter-project/EuroLLM-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Function to run inference using the local model
def run_inference(prompt, max_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.00001, top_p=0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to process data and perform inference
def process_data(folder_path, save_dir, max_tokens=256):
    result = {}

    for filename in os.listdir(folder_path):
        if filename.startswith("xsafety_harmful_") and filename.endswith(".txt"):
            # Extract language code from filename
            language_code = filename[-6:-4]

            # Read queries from the file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                queries = [line.strip() for line in file]

            # queries = queries[:50]

            # Save path for the responses based on the language
            save_path = os.path.join(save_dir, f"responses_{language_code}.json")

            # If the responses JSON file does not exist, perform inference
            if not os.path.exists(save_path):
                qa_pairs = {}
                for query in tqdm(queries[0:], ascii=True, leave=False):
                    # Construct the prompt for the model
                    prompt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{query}\n<|im_end|>\n<|im_start|>assistant\n"
                    
                    # Run inference using the model with a token limit
                    response = run_inference(prompt, max_tokens=max_tokens)
                    print(response)
                    qa_pairs[query] = response

                # Save the results to a JSON file
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(qa_pairs, f, indent=4)
            
            # If file exists, load the saved data
            else:
                with open(save_path, "r", encoding="utf-8") as f:
                    qa_pairs = json.load(f)

            # Create the dictionary with query and response pairs
            lang_dict = [{"query": query, "response": qa_pairs.get(query, "")} for query in queries]

            # Add to the result dictionary with the language code as key
            result[language_code] = lang_dict

    return result

# Main execution
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="PATH to the folder containing the text files.")
    parser.add_argument("--save-dir", type=str, help="PATH to the folder to save the results.", default="results/")
    cmd_args = vars(parser.parse_args())

    # Make sure save directory exists
    save_dir = cmd_args["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Ensure the data directory exists
    data_dir = cmd_args["data_dir"]
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The specified data directory '{data_dir}' does not exist.")

    # Process data and save results
    output = process_data(data_dir, save_dir)
    print("Processing complete. Results saved to:", save_dir)
