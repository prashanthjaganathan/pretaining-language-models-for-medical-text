from enum import Enum
from tokenizers import Tokenizer, models, trainers
from typing import List, Optional, Union
from tokenizers import Tokenizer as PreDefTokenizer
from tokenizers.models import WordPiece, BPE
import os
import pandas as pd
from .base import Tokenizer as MyTokenizer

class TokenizerAlgorithm(Enum):
    BPE = "bpe"
    WORDPIECE = "wordpiece"
    # TODO: Unigram Tokenizer
    # TODO: WordLevel Tokenizer


# BUG: BPE and wordpience tokenization doesn't seem to work, always returns [UNK] token
class TrainableTokenizer(MyTokenizer):
    """
    A configurable tokenizer that can be trained on a corpus using different algorithms.
    
    Supported algorithms:
    - BPE (Byte Pair Encoding)
    - WordPiece
    
    Usage:
        tokenizer = TrainableTokenizer(
            corpus=texts,
            algorithm="bpe", 
            vocab_size=30000,
            min_frequency=2
        )
        tokenizer.train()
        tokens = tokenizer.tokenize(text)
    """
    
    def __init__(self,
                 corpus: List[str],
                 algorithm: str,
                 vocab_size: int = 30000,
                 min_frequency: int = 2,
                 special_tokens: Optional[List[str]] = None):
        """
        Initialize the trainable tokenizer.
        
        Args:
            corpus: List of texts to train on
            algorithm: The tokenization algorithm to use ('bpe' or 'wordpiece')
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum frequency for a token to be included
            special_tokens: List of special tokens to add to vocabulary
        """
        try:
            self.algorithm = algorithm.lower()
        except ValueError:
            raise ValueError(f"Algorithm must be one of: {[algo.value for algo in TokenizerAlgorithm]}")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        if isinstance(corpus, pd.Series):
            self.corpus = corpus.dropna().astype(str).tolist()  # Drop NaN and ensure str format
        elif isinstance(corpus, list):
            self.corpus = [str(text) for text in corpus]  # Ensure list of strings
        else:
            raise TypeError("corpus must be a Pandas Series or a list of strings")
        
        # Initialize tokenizer and trainer
        self.tokenizer = self._initialize_tokenizer()
        self.trainer = self._initialize_trainer()
        self.trained: bool = self.is_trained()

    def is_trained(self) -> bool:
        if os.path.exists(f'trained_models/tokenizers/trained_{self.algorithm}.py'):
            return True
        
        return False
        
    def _initialize_tokenizer(self) -> PreDefTokenizer:
        """Initialize the appropriate tokenizer based on selected algorithm"""
        if self.algorithm == TokenizerAlgorithm.BPE:
            return PreDefTokenizer(BPE(unk_token="[UNK]"))
        else:  # WordPiece
            return PreDefTokenizer(WordPiece(unk_token="[UNK]"))
            
    def _initialize_trainer(self):
        """Initialize the appropriate trainer based on selected algorithm"""
        trainer_config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "show_progress": True
        }
        
        if self.algorithm == TokenizerAlgorithm.BPE:
            return trainers.BpeTrainer(**trainer_config)
        else:  # WordPiece
            return trainers.WordPieceTrainer(**trainer_config)
            
    def train(self) -> None:
        """Train the tokenizer on the corpus"""
        self.tokenizer.train_from_iterator(self.corpus, trainer=self.trainer)
        
    def save(self, path: str) -> None:
        """Save the trained tokenizer to disk"""
        self.tokenizer.save(path)
            
    def load(self, path: str) -> None:
        """Load a trained tokenizer from disk"""
        self.tokenizer = PreDefTokenizer.from_file(path)
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not self.trained:
            self.train()
            self.save(f'trained_models/tokenizers/trained_{self.algorithm}.py')
            self.trained = True
            
        self.load(f'trained_models/tokenizers/trained_{self.algorithm}.py')
        print(text)
        print(self.tokenizer)
        encoding = self.tokenizer.encode(text)
        return encoding.tokens