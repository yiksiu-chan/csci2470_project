import os
import re
import json 

def evaluate_responses_with_gpt(input_folder="results/eval_results/", summary_save_path="eval_new.txt"):
    with open(summary_save_path, "a", encoding="utf-8") as summary_file:
        for filename in os.listdir(input_folder):
            if filename.endswith("_eval.json"):
                match = re.search(r"(.*?)_(.*?)_eval\.json", filename)
                if match:
                    language_code, category_name = match.groups()
                else:
                    raise ValueError("Filename does not match expected format")

                file_path = os.path.join(input_folder, filename)
                data = json.load(open(file_path, "r"))
                
                success_count = sum(entry["ASR"] for entry in data)
                asr = success_count / len(data) if data else 0

                print(f"{language_code}_{category_name}: ASR = {asr:.2f}")
                summary_file.write(f"{language_code}_{category_name}: ASR = {asr:.2f}")

    print(f"\nEvaluation summary saved to {summary_save_path}")


def main():
    evaluate_responses_with_gpt()

if __name__ == "__main__":
    main()