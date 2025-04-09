from .model_architectures.lstm_self_attention import SelfAttention, LSTM_SelfAttention
import torch.nn as nn
from typing import Dict

class ModelFactory:
    """
    A factory class for creating different model architectures.

    Currently, the following model implementations are available:

    1. **LSTM with Self-Attention** (`lstm_and_self_attention`):
       - Combines Long Short-Term Memory (LSTM) with a self-attention mechanism.
       - Used for sequence processing tasks where long-term dependencies and context awareness are crucial.

    Usage:
        To create a model, call the static method `get_model` with the desired model name.
        
        Example:
            model = ModelFactory.get_model('lstm_and_self_attention', **kwargs)

    Methods:
        get_model(model_name: str, **kwargs):
            Returns the model instance based on the provided `model_name`. 
            Raises a `ValueError` if the model name is not supported.

    Parameters:
        model_name (str): The name of the model to create. Currently supported models:
                          'lstm_and_self_attention'.
        **kwargs: Additional arguments to pass to the model constructor.
    """

    @staticmethod
    def get_model(model_name, **kwargs):
        model_map: Dict[str, nn.Module] = {
            # 'lstm': LSTMModel
            'lstm_and_self_attention': LSTM_SelfAttention,
            # 'transformer': Transformer
        }
        if model_name not in model_map:
            raise ValueError(f'''{model_name} not supported currently. Refer docstrings for current
                             available model implementations''')
        
        return model_map[model_name](**kwargs)