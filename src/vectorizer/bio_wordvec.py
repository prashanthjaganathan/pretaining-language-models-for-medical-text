from typing import List
import os
import numpy as np
import torch
from gensim.models import FastText
from .base import Embedding
from tqdm import tqdm

class BioWordVecModel(Embedding):
    """FastText-based BioWordVec embedding model."""
    def __init__(self, model_path: str):
        self.path = model_path
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            return FastText.load_fasttext_format(model_path)
        else:
            raise ValueError(f'Unable to find BioWordVec model in path: {model_path}')

    def embed(self, tokenized_data: List[List[str]], abbreviations: List[str] = None):
        """Returns word-to-word embeddings for a given tokenized document with padding.
        
        Args:
            tokenized_data (List[List[str]]): List of tokenized documents (each document is a list of words).
            abbreviations (List[str], optional): List of abbreviations to mark with {ABBR}.
        
        Returns:
            np.ndarray: Padded word embeddings for each token in each document.
        """
        all_embeddings = []
        abbreviations = set(abbreviations)
        max_length = max(len(doc) for doc in tokenized_data)  # Find max sequence length

        print('In func')
        
        for doc in tqdm(tokenized_data, desc='documents', total=len(tokenized_data)):
            doc_embeddings = []
            for token in doc:
                # Mark abbreviations
                token = f"{'[ABBR]'}{token}[/ABBR]" if token in abbreviations else token
                
                # Get word embedding
                if token in self.model:
                    embedding = self.model[token]
                else:
                    embedding = self.model.get_word_vector(token)  # Handle OOV with FastText subword representation
                
                doc_embeddings.append(embedding)
            
            # Pad embeddings to match max_length
            while len(doc_embeddings) < max_length:
                doc_embeddings.append(np.zeros(self.model.vector_size))  # Padding with zero vectors
            
            all_embeddings.append(doc_embeddings)
        
        return np.array(all_embeddings)