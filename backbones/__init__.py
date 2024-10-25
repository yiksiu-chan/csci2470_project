import os
import sys


def get_backbone(model_name):
    """
    Load and return the appropriate model object based on the given model name.

    This function dynamically adds the current directory to the system path,
    checks the provided model name, and imports and initializes the corresponding
    model class.asdasd

    Parameters:
    model (str): The name of the model to load. Expected values are substrings 
                 that indicate the type of model, such as "openai" for OpenAI models 
                 or other strings for local Llama2 models.

    Returns:
    object: An instance of the specified model class, either OpenAIModel or Llama2Model.
    """
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))

    # from vllm_model import VLLMModel
    # model = VLLMModel(model_name)
    from local_model import LocalModel
    model = LocalModel(model_name)
        
    return model