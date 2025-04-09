from typing import List
from .base import Tokenizer


class CharacterTokenizer(Tokenizer):
    """
    Tokenizer that splits text by characters.
    """
    def tokenize(self, text: str) -> List[str]:
        return list(text)