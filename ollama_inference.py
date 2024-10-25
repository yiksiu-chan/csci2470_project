import os
import json
import argparse

from utils import *
from backbones import *

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="PATH to the folder containing the text files.")
parser.add_argument("--model", type=str, help="Model name.")

parser.add_argument("--device", type=int, help="Device to run models on.", default=0)

cmd_args = vars(parser.parse_args())

# Load model
model = get_backbone(cmd_args["model"])

# Make save dir
save_dir = os.path.join("results/", cmd_args["data"], cmd_args["model"])
os.makedirs(save_dir, exist_ok=True)

# Inference function
def infer(data, model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if not os.path.exists(save_path):
        outputs = model.infer_batch(data, save_path)

        qa_pairs = {}
        for query, response in zip(data, outputs):
            qa_pairs[query] = response if response is not None else ""

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=4)

    else:
        with open(save_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)

    return qa_pairs

def process_data(folder_path, save_dir, model):
    result = {}

    for filename in os.listdir(folder_path):
        if filename.startswith("zou_harmful_behaviors_") and filename.endswith(".txt"):
            language_code = filename[-6:-4]

            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                queries = [line.strip() for line in file]
            
            # Only testing with 50 queries now
            queries = queries[:50]

            save_path = os.path.join(save_dir, f"responses_{language_code}.json")

            qa_pairs = infer(queries, model, save_path)

            lang_dict = [{"query": query, "response": qa_pairs.get(query, "")} for query in queries]
            result[language_code] = lang_dict

    return result

if __name__ == "__main__":
    # Ensure the data directory exists
    data_dir = cmd_args["data"]
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The specified data directory '{data_dir}' does not exist.")

    output = process_data(data_dir, save_dir, model)
    print("Processing complete. Results saved to:", save_dir)
