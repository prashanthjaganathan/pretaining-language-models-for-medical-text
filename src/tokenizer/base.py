from typing import List, Any
from abc import ABC, abstractmethod


class Tokenizer:
    """
    Base class for all tokenizers. All custom tokenizers must inherit from this.
    """
    @abstractmethod
    def tokenize(self, text=None) -> Any:
        pass