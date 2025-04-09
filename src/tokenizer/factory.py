from .base import Tokenizer
from .character import CharacterTokenizer
from .nltk import NLTKTokenizer
from .pretrained import PretrainedTokenizer
from .whitespace import WhitespaceTokenizer
from .trainable import TrainableTokenizer
from typing import Optional, List, Any

class TokenizerFactory:
    """
    Factory class that maps tokenizer types to specific Tokenizer classes.
    """
    tokenizer_map = {
        'whitespace': WhitespaceTokenizer,
        'characters': CharacterTokenizer,
        'nltk': NLTKTokenizer,
        'pretrained': PretrainedTokenizer,
        'trainable': TrainableTokenizer,
    }

    @staticmethod
    def get_tokenizer(tokenizer_type: str, **kwargs) -> Tokenizer:
        """
        Returns an instance of the appropriate tokenizer based on the tokenizer type.

        Parameters:
        - tokenizer_type (str): Type of tokenizer ('whitespace', 'characters', 'nltk', 'pretrained', 'trainable')
        
        Kwargs for different tokenizer types:
            For pretrained:
                - pretrained_model (str): Name of the pretrained model
                    Options: 'google-bert/bert-base-uncased', 'dmis-lab/biobert-v1.1', 'neuml/pubmedbert-base-embeddings'
            
            For trainable:
                - corpus (List[str]): Training text corpus
                - algorithm (str): Tokenization algorithm ('bpe' or 'wordpiece')
                - vocab_size (int, optional): Maximum vocabulary size. Default: 30000
                - min_frequency (int, optional): Minimum token frequency. Default: 2
                - special_tokens (List[str], optional): Special tokens to include

        Returns:
        - Tokenizer: The corresponding tokenizer object.
        """
        if tokenizer_type not in TokenizerFactory.tokenizer_map:
            raise ValueError(f"Unknown tokenizer type '{tokenizer_type}'. "
                           f"Available types: {list(TokenizerFactory.tokenizer_map.keys())}")

        tokenizer_class = TokenizerFactory.tokenizer_map[tokenizer_type]

        if tokenizer_type == 'pretrained':
            required_params = {'corpus', 'pretrained_model'}
            missing_params = required_params - set(kwargs.keys())
            if missing_params:
                raise ValueError(f"Missing required parameters for pre-trained tokenizer: {missing_params}")


            return tokenizer_class(
                pretrained_model=kwargs['pretrained_model'],
                corpus=kwargs['corpus']
                )

        elif tokenizer_type == 'trainable':
            required_params = {'corpus', 'algorithm'}
            missing_params = required_params - set(kwargs.keys())
            if missing_params:
                raise ValueError(f"Missing required parameters for trainable tokenizer: {missing_params}")
            
            return tokenizer_class(
                corpus=kwargs['corpus'],
                algorithm=kwargs['algorithm'],
                vocab_size=kwargs.get('vocab_size', 30000),
                min_frequency=kwargs.get('min_frequency', 2),
                special_tokens=kwargs.get('special_tokens')
            )

        return tokenizer_class()