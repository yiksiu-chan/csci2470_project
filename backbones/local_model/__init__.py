import os
import json

import ollama
from tqdm import tqdm
from munch import Munch


class LocalModel:
    def __init__(self, model_name):
        self.model = model_name

        params = Munch.fromYAML(
            open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r")
        )

        self.num_predict = params.num_predict
        self.temperature = params.temperature
        self.top_p = params.top_p


    def infer_batch(self, inputs, save_dir):
        responses = (
            json.load(open(save_dir, "r")) if os.path.exists(save_dir) else list()
        )
        start_index = len(responses)

        for instance in tqdm(inputs[start_index:], ascii=True, leave=False):
            if type(instance) == list:
                response = [
                    (
                        ollama.generate(model=self.model, prompt=instance_item, options={
                            "num_predict": self.num_predict,
                            "top_p": self.top_p,
                            "temperature": self.temperature})
                            [
                            "response"
                            ]
                        if instance_item != ""
                        else ""
                    )
                    for instance_item in instance
                ]
            elif type(instance) == str:
                response = ollama.generate(model=self.model, prompt=instance)[
                    "response"
                ]

            responses.append(response)

            json.dump(responses, open(save_dir, "w"), indent=4)

        return responses
