import os
import json

from google.cloud import translate_v2 as translate
from google.oauth2 import service_account


class GoogleCloudTranslator:
    def __init__(self):
        credentials = service_account.Credentials.from_service_account_file(os.path.join(os.path.dirname(__file__), "google_credentials.json"))
        self.client = translate.Client(credentials=credentials)

    def translate_text(self, text, source_language=None, target_language=None, save_dir=None):
        """
        Translates text into the target language, with functionality similar to `translate_to_english`.

        This method can translate a single string or a list of strings. If the target language 
        is the same as the source language, it returns the original text. It also supports caching 
        translations in a log file if `save_dir` is provided.

        Args:
            text (str or list of str): The text to be translated. It can be a single string or a list of strings.
            source_language (str): The source language code (ISO 639-1) of the text to be translated. Defaults to None.
            target_language (str): The target language code (ISO 639-1) for the translation. Defaults to None.
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
        if target_language is None:
            target_language = self.target

        # Check for already cached translations
        if isinstance(text, str):
            if text in result_log:
                return result_log[text]
            elif source_language == target_language:
                result_log[text] = text
                if save_dir:
                    json.dump(result_log, open(save_dir, "w"), indent=4)
                return text
            else:
                translated = self.perform_translation(text, target_language)
                result_log[text] = translated
                if save_dir:
                    json.dump(result_log, open(save_dir, "w"), indent=4)
                return translated
        elif isinstance(text, list):
            translations = []
            for sentence in text:
                if sentence in result_log:
                    translations.append(result_log[sentence])
                elif source_language == target_language:
                    result_log[sentence] = sentence
                    translations.append(sentence)
                else:
                    translated = self.perform_translation(sentence, target_language)
                    result_log[sentence] = translated
                    translations.append(translated)
            if save_dir:
                json.dump(result_log, open(save_dir, "w"), indent=4)
            return translations


    def perform_translation(self, text, target_language):
        """Helper function to perform the actual translation."""
        return self.client.translate(text, target_language=target_language)["translatedText"]