import os
import sys

import json
import multiprocessing

from tqdm import tqdm
from langdetect import detect
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError


from utils import partition

def get_translator(translator):
    """ """
    
    if "azure_translator" in translator.lower():
        # from azure_translator_wrapper import AzureTranslatorWrapper
        return AzureTranslatorWrapper()
    
    elif "google_translate" in translator.lower():
        # from google_translate_wrapper import GoogleCloudTranslator
        return GoogleCloudTranslator()
    
    else:
        raise NotImplementedError


class AzureTranslatorWrapper:
    def __init__(self, target=None, num_processes=10):

        config = json.load(
            open(os.path.join(os.path.dirname(__file__), "config.json"), "r")
        )

        self.num_processes = num_processes
        self.translator = TextTranslationClient(
            credential=AzureKeyCredential(config["api_key"]), region=config["region"]
        )

        if target is not None:
            self.target = target
    
    def translate_multilingual(self, sentence, source=None, target=None):
        """
        Translate English sentences to various target languages.
        """
        if not sentence:  # Handles the empty string edge case
            return ""

        if target is None:
            target = self.target

        try:
            if target == "en" and source == "en":
            # and self.is_english(sentence): removing this condition since Swahili is often mis-detected as English
                print("Sentence is in English.")
                return sentence
            else:
                translated = self.perform_translation(sentence, target)                
                return translated
            
        except HttpResponseError as exception:
            print(sentence)
            if exception.error is not None:
                print(f"Error Code: {exception.error.code}")
                print(f"Message: {exception.error.message}")
                        
            raise
            
    def translate_to_english(self, sentence, source=None, target=None, save_dir=None):
        """
        Translate sentences in various source languages to English. 
        """
        result_log = json.load(open(save_dir, "r")) if save_dir != None and os.path.exists(save_dir) else {}
        
        if not sentence:  # Handles the empty string edge case
            return ""

        if target is None:
            target = self.target

        try:
            if target == "en" and source == "en":
            # and self.is_english(sentence): removing this condition since Swahili is often mis-detected as English
                print("Sentence is in English.")
                result_log[sentence] = sentence
                json.dump(result_log, open(save_dir, "w"), indent=4)
                return sentence
            elif sentence in result_log:
                return result_log[sentence]
            else:
                translated = self.perform_translation(sentence, target)
                result_log[sentence] = translated
                
                json.dump(result_log, open(save_dir, "w"), indent=4)
                
                return translated
            
        except HttpResponseError as exception:
            print(sentence)
            if exception.error is not None:
                print(f"Error Code: {exception.error.code}")
                print(f"Message: {exception.error.message}")
                        
            raise

    def is_english(self, sentence):
        """Detect if the sentence is in English."""
        try:
            source_lang = detect(sentence)
        except:
            source_lang = "error"
        return source_lang == "en"

    def perform_translation(self, sentence, target):
        """Translate the sentence to the target language."""
        if isinstance(sentence, list):
            translated = self.translator.translate(body=sentence, to_language=[target])
            return [t.translations[0].text for t in translated]
        elif isinstance(sentence, str):
            translated = self.translator.translate(
                body=[sentence], to_language=[target]
            )
            return translated[0].translations[0].text

    def translate_batch(self, sentences, save_dir=None, source=None, target=None):
        if save_dir is not None:
            base_save_dir = str(save_dir)
        if target is None:
            target = self.target

        # Partition instances
        partitioned_sentences = partition(sentences, self.num_processes)

        # Start multiprocess instances
        lock = multiprocessing.Manager().Lock()
        pool = multiprocessing.Pool(processes=self.num_processes)

        worker_results = []
        for process_id in range(len(partitioned_sentences)):
            if save_dir is not None:
                save_dir = base_save_dir.replace(
                    f".{save_dir.split('.')[-1]}",
                    f"-process={process_id}.{save_dir.split('.')[-1]}",
                )

            async_args = (
                self.translate,
                partitioned_sentences[process_id],
                process_id,
                save_dir,
                target,
                lock,
            )

            # Run each worker
            worker_results.append(pool.apply_async(query_worker, args=async_args))

        pool.close()
        pool.join()

        responses = []
        for worker_result in worker_results:
            responses += worker_result.get()

        return responses


def query_worker(translate, inputs, process_id, save_dir, target, lock):
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

        # If instance is a list, pass contents one by one.
        # Otherwise, pass instance
        if isinstance(instance, list):
            response = [
                translate(instance_item, target=target) if instance_item != "" else ""
                for instance_item in instance
            ]

        elif isinstance(instance, str):
            response = translate(instance, target=target)

        responses.append(response)

        if save_dir is not None:
            json.dump(responses, open(save_dir, "w"), indent=4)

    with lock:
        bar.close()

    return responses


import os
import json

from google.cloud import translate_v2 as translate
from google.oauth2 import service_account


class GoogleCloudTranslator:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(os.path.join(os.path.dirname(__file__), "google_credentials.json"))
        self.client = translate.Client(credentials=credentials)

    def translate_to_english(self, text, source=None, target=None, save_dir=None):
        """
        Translate a single string or a list of strings. If the target language 
        is the same as the source language, it returns the original text. It also supports caching 
        translations in a log file if `save_dir` is provided.

        Args:
            text (str or list of str): The text to be translated. It can be a single string or a list of strings.
            source (str): The source language code (ISO 639-1) of the text to be translated. Defaults to None.
            target (str): The target language code (ISO 639-1) for the translation. Defaults to None.
            save_dir (str): The path to save/load the translation cache. If None, no caching is performed.

        Returns:
            str or list of str: The translated text. If the input is a single string, the return value will be a single string.
            If the input is a list of strings, the return value will be a list of strings where each string is the translation
            of the corresponding input string.
        """
        # Load existing log if save_dir is provided
        result_log = json.load(open(save_dir, "r")) if save_dir and os.path.exists(save_dir) else {}

        if not text:  # Handle empty string or empty list edge case
            return text

        # Default the target language if not provided
        if target is None:
            target = self.target

        if source is None: 
            source = self.detect_language(text)

        # Check for already cached translations
        if isinstance(text, str):
            if text in result_log:
                return result_log[text]
            elif source == target:
                result_log[text] = text
                if save_dir:
                    json.dump(result_log, open(save_dir, "w"), indent=4)
                return text
            else:
                translated = self.perform_translation(text, target)
                result_log[text] = translated
                if save_dir:
                    json.dump(result_log, open(save_dir, "w"), indent=4)
                return translated
        elif isinstance(text, list):
            translations = []
            for sentence in text:
                if sentence in result_log:
                    translations.append(result_log[sentence])
                elif source == target:
                    result_log[sentence] = sentence
                    translations.append(sentence)
                else:
                    translated = self.perform_translation(sentence, target)
                    result_log[sentence] = translated
                    translations.append(translated)
            if save_dir:
                json.dump(result_log, open(save_dir, "w"), indent=4)
            return translations
    
    def detect_language(self, text):
        """Detects the text's language."""

        # Text can also be a sequence of strings, in which case this method
        # will return a sequence of results for each text.
        result = self.client.detect_language(text)

        print(f"Text: {text}")
        print("Confidence: {}".format(result["confidence"]))
        print("Language: {}".format(result["language"]))

        return result["language"]


    def perform_translation(self, text, target_language):
        """Helper function to perform the actual translation."""
        return self.client.translate(text, target_language=target_language)["translatedText"]