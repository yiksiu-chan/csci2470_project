import os
import re
import json 

from translation import get_translator

translator = get_translator("azure_translator")

input_folder = "results/xsafety/"
output_folder = "results/xsafety/translated_responses/"

# If queries are the same in each language
# queries_raw = json.load(open("results/raw_responses/responses_en.json", "r"))
# queries = [query for query in queries_raw.keys()]
# print(len(queries))

# If queries are different in each language
queries = False

# Regex functions
def extract_query(s):
    """
    Extracts the content between \n user\n and \n assistant\n, ignoring case.
    """
    query = re.search(r'\n user\n(.*?)\n assistant\n', s, re.IGNORECASE | re.DOTALL)
    if query:
        return query.group(1).strip()
    else: 
        raise ValueError()

def extract_response(s):
    """
    Extracts everything after \n assistant\n, ignoring case.
    """
    response = re.search(r'\n assistant\n(.*)', s, re.IGNORECASE | re.DOTALL)
    if response:
        return response.group(1).strip()
    else:
        raise ValueError()

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
            