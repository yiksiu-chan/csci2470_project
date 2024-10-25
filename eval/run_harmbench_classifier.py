import os
import re
import json 

from munch import Munch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Settings
model_name="cais/HarmBench-Llama-2-13b-cls"
config = Munch.fromYAML(
        open(os.path.join(os.path.dirname(__file__), "harmbench_config.yaml"), "r"))

model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, truncation_side="left", padding_side="left")


def compute_scores(behaviors, generations, batch_size=10):
    behaviors = [config.behavior_prompt + b for b in behaviors]
    final_out = []

    total_inputs = len(behaviors)
    indices = list(range(0, total_inputs, batch_size))

    for start in indices:
        end = start + batch_size
        inputs = [
            config.prompt.format(behavior=b, generation=g)
            for b, g in zip(behaviors, generations)
        ]
        inputs = inputs[start:end]  

        if not inputs:  
            continue

        encoded = tokenizer(inputs, max_length=1024, truncation=True, return_tensors="pt", padding="longest")
        
        encoded = encoded.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **encoded.to(model.device),
                do_sample=False,
                max_new_tokens=1,
            ).cpu()
            output_ids = output_ids[:, len(encoded.input_ids[0]) :]

        completion = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        final_out.extend(completion)

        torch.cuda.empty_cache()

    return final_out


input_folder = "../results/xsafety/translated_responses/"

def process_folder(input_folder):
    for filename in os.listdir(input_folder):
        if filename.startswith("responses_en_") and filename.endswith(".json"):
            match = re.search(r"responses_en_(.*?)\.json", filename)
            if match:
                language_code = match.group(1)
            else:
                raise ValueError()

        file_path = os.path.join(input_folder, filename)
        data = json.load(open(file_path, "r"))

        behaviors, generations = [], []
        for b, g in data.items():
            behaviors.append(b)
            generations.append(g)

        scored_data = compute_scores(behaviors, generations)

        results = []
        for (b, g), score in zip(data.items(), scored_data):
            result = {}
            result["query"] = b
            result["response"] = g
            result["ASR"] = 1 if score == "Yes" else 0
            results.append(result)

        save_path = os.path.join(input_folder, f"responses_en_{language_code}_eval.json")
    
        success_count = scored_data.count('Yes')
        asr = 0 if success_count == 0 else success_count / len(scored_data)

        print(filename)
        print(f"{language_code}: {asr}")
        json.dump(results, open(save_path, "w", encoding="utf-8"), indent=4)


process_folder(input_folder)