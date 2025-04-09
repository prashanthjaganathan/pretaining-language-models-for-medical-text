from typing import List
from .base import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizer
import pandas as pd

class PretrainedTokenizer(Tokenizer):
    """
    Tokenizer that uses a pretrained model from HuggingFace's Transformers library.
    """
    def __init__(self, pretrained_model: str, corpus):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        if isinstance(corpus, pd.Series):
            self.corpus = corpus.dropna().astype(str).tolist()  # Drop NaN and ensure str format
        elif isinstance(corpus, list):
            self.corpus = [str(text) for text in corpus]  # Ensure list of strings
        else:
            raise TypeError("corpus must be a Pandas Series or a list of strings")
    
    def tokenize(self) -> List[str]:
        print(self.corpus)
        return self.tokenizer.tokenize(self.corpus)
        # return self.tokenizer(self.corpus, padding=True, truncation=True, return_tensors='pt')