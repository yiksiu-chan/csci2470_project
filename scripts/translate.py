import os
import re
import json 

from translation import get_translator
from utils import *

translator = get_translator("azure_translator")

input_folder = "results/dpo-2/raw_responses/"
output_folder = "results/dpo-2/translated_responses/"

# If queries are the same in each language
# queries_raw = json.load(open("results/raw_responses/responses_en.json", "r"))
# queries = [query for query in queries_raw.keys()]
# print(len(queries))

# If queries are different in each language
queries = False

# Perform translation on input folder
for filename in os.listdir(input_folder):
    if filename.startswith("responses_") and filename.endswith(".json"):
        match = re.search(r"responses_(.*?)\.json", filename)
        if match:
            language_code = match.group(1)
        else:
             raise ValueError()

        file_path = os.path.join(input_folder, filename)
        data = json.load(open(file_path, "r"))

        # Save translated responses
        save_path = os.path.join(output_folder, f"responses_en_{language_code}.json")
        
        translated_responses = {}
        if not os.path.exists(save_path):
            if queries: 
                for q, (query, response) in zip(queries, data.items()):
                    translated = translator.translate_to_english(response, target="en", save_dir=save_path)
                    translated_responses[q] = extract_query(translated)
            else: 
                for query, response in data.items():
                    q = extract_query(response)
                    r = extract_response(response)

                    translated_q = translator.translate_to_english(q, target="en", save_dir=save_path)
                    translated_r = translator.translate_to_english(r, target="en", save_dir=save_path)

                    translated_responses[translated_q] = translated_r
        
            json.dump(translated_responses, open(save_path, "w", encoding="utf-8"), indent=4)
            print(f"Translated {language_code} to English.")
            