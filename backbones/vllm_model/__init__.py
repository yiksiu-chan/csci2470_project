import os
import json

from vllm import LLM, SamplingParams
from tqdm import tqdm
from munch import Munch


class VLLMModel:
    def __init__(self, model_name):
        self.model = "Qwen/Qwen2-72B-Instruct"

        params = Munch.fromYAML(
            open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r")
        )

        self.num_predict = params.num_predict
        self.temperature = params.temperature
        self.top_p = params.top_p

        self.sampling_params = SamplingParams(temperature=0, top_p=0.0000001, max_tokens=256)
        self.llm = LLM(model="Qwen/Qwen2-72B-Instruct", tensor_parallel_size=2)


        # self.engine = LLMEngine(
        #     model=self.model,
        #     tensor_parallel_size=3,  # Use 3 GPUs
        # )

    def infer_batch(self, inputs, save_dir):
        responses = (
            json.load(open(save_dir, "r")) if os.path.exists(save_dir) else []
        )
        start_index = len(responses)

        batch_size = 8

        # sampling_params = SamplingParams(
        #     temperature=self.temperature,
        #     top_p=self.top_p,
        #     max_tokens=self.num_predict,
        # )

        for i in tqdm(
            range(start_index, len(inputs), batch_size), ascii=True, leave=False
        ):
            batch_inputs = inputs[i : i + batch_size]

            prompts = []
            prompt_indices = []

            for input_idx, instance in enumerate(batch_inputs):
                if isinstance(instance, list):
                    for sub_idx, instance_item in enumerate(instance):
                        if instance_item != "":
                            prompts.append(instance_item)
                            prompt_indices.append((input_idx, sub_idx))
                        else:
                            prompt_indices.append((input_idx, sub_idx))
                elif isinstance(instance, str):
                    if instance != "":
                        prompts.append(instance)
                        prompt_indices.append((input_idx, None))
                    else:
                        prompt_indices.append((input_idx, None))

            if prompts:
                outputs = self.llm.generate(prompts, self.sampling_params)
            else:
                outputs = []

            batch_responses = [None] * len(batch_inputs)
            for idx, instance in enumerate(batch_inputs):
                if isinstance(instance, list):
                    batch_responses[idx] = ["" for _ in instance]
                elif isinstance(instance, str):
                    batch_responses[idx] = ""

            output_idx = 0
            for idx_in_prompts, (input_idx, sub_idx) in enumerate(prompt_indices):
                if sub_idx is not None:
                    if batch_responses[input_idx][sub_idx] == "":
                        if output_idx < len(outputs):
                            output_text = outputs[output_idx].outputs[0].text
                            batch_responses[input_idx][sub_idx] = output_text
                            output_idx += 1
                else:
                    if batch_responses[input_idx] == "":
                        if output_idx < len(outputs):
                            output_text = outputs[output_idx].outputs[0].text
                            batch_responses[input_idx] = output_text
                            output_idx += 1

            responses.extend(batch_responses)
            
            json.dump(responses, open(save_dir, "w"), indent=4)

        return responses
