import os
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint_dir = "checkpoints_v3"
latest_checkpoint = max([os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir)], key=os.path.getmtime)

model = AutoModelForCausalLM.from_pretrained(latest_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)

model.save_pretrained("model_dpo_v3/")
tokenizer.save_pretrained("model_dpo_v3/")

model.push_to_hub("yiksiu/EuroLLM-1.7B-DPO-v3")