import os
import json
import re
import multiprocessing

import openai
from openai import OpenAI
from openai import AzureOpenAI
from munch import Munch
from tqdm import tqdm

from utils import *


def get_prompt_response(instance, model, temperature, max_tokens, top_p):
    chat_model = any(chat_model_name in model for chat_model_name in CHAT_MODELS)

    if chat_model:
        response = (
            client.chat.completions.create(
                messages=[{"role": "user", "content": instance}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            .choices[0]
            .message.content
        )
    else:
        response = (
            client.completions.create(
                model=model, prompt=instance, temperature=temperature
            )
            .choices[0]
            .text
        )
    return response


def query_worker(model, inputs, process_id, save_dir, lock, temperature, max_tokens, top_p):
    """ """

    def query_loop(model, instance, temperature, max_tokens, top_p):
        """ """
        # Edge case: If an empty string is passed, return an empty string
        if instance == "":
            return ""

        server_error = False

        response = None
        while response == None:
            try:
                response = get_prompt_response(instance, model, temperature, max_tokens, top_p)

            except openai.InternalServerError:
                # If first time encountering error, change model to long-context version
                if server_error == False:
                    server_error = True
                    model += "-16k"

                # Otherwise, adding prompt did not help, so exit
                else:
                    print(
                        "[InternalServerError]: Likely generated invalid Unicode output."
                    )
                    print(instance)
                    exit()
            except openai.BadRequestError:
                print(
                    "[BadAPIRequest]: Likely input too long or invalid settings (e.g. temperature > 2)."
                )
                print(instance)
                return "Sorry, I cannot assist with that."

            except openai.Timeout:
                continue

        return response

    with lock:
        bar = tqdm(
            desc=f"Process {process_id+1}",
            total=len(inputs),
            position=process_id + 1,
            leave=False,
        )

    # If partially populated results file exists, load and continue
    responses = json.load(open(save_dir, "r")) if os.path.exists(save_dir) else list()
    start_index = len(responses)

    for instance in inputs[start_index:]:
        with lock:
            bar.update(1)

        # If instance is a list, pass contents one by one. Otherwise, pass instance
        if type(instance) == list:
            response = [
                (
                    query_loop(model, instance_item, temperature, max_tokens, top_p)
                    if instance_item != ""
                    else ""
                )
                for instance_item in instance
            ]
        elif type(instance) == str:
            response = query_loop(model, instance, temperature, max_tokens, top_p)

        responses.append(response)

        json.dump(responses, open(save_dir, "w"), indent=4)

    with lock:
        bar.close()

    return responses


class OpenAIModel:
    def __init__(self, model_name="Vanilla-GPT4o"):
        """ """
        global client
        args = Munch.fromYAML(
            open(os.path.join(os.path.dirname(__file__), "gpt_config.yaml"), "r")
        )

        client = OpenAI(api_key=args.api_key)

        # client = AzureOpenAI(
        #     api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        #     api_version = "2024-02-01",
        #     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        # )

        self.model = model_name
        self.num_processes = args.num_processes
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.top_p = args.top_p 

    def infer_batch(self, inputs, save_dir):
        """ """
        # Partition instances
        paritioned_inputs = partition(inputs, self.num_processes)

        # Start multiprocess instances
        lock = multiprocessing.Manager().Lock()
        pool = multiprocessing.Pool(processes=self.num_processes)

        worker_results = []
        for process_id in range(len(paritioned_inputs)):
            async_args = (
                self.model,
                paritioned_inputs[process_id],
                process_id,
                save_dir.replace(
                    "." + save_dir.split(".")[-1],
                    f"-process={process_id}{'.' + save_dir.split('.')[-1]}",
                ),
                lock,
                self.temperature,
                self.max_tokens,
                self.top_p
            )

            # Run each worker
            worker_results.append(pool.apply_async(query_worker, args=async_args))

        pool.close()
        pool.join()

        responses = []
        for worker_result in worker_results:
            responses += worker_result.get()

        return responses