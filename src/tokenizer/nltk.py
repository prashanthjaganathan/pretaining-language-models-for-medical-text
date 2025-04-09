from typing import List
from .base import Tokenizer
from nltk.tokenize import word_tokenize


class NLTKTokenizer(Tokenizer):
    """
    Tokenizer that uses NLTK's word_tokenize method.
    """
    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)