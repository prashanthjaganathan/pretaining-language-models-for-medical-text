from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from .base import Embedding
import numpy as np

class RoBERTaModel(Embedding):
    """
    RoBERTa-based embedding model.
    NOTE: Tokenization not required
    """
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, untokenized_data):
        all_embeddings = []

        for doc in untokenized_data:
            inputs = self.tokenizer(doc, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Take the [CLS] token embedding (first token's embedding) or use the mean of all token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            all_embeddings.append(embeddings)

        return np.array(all_embeddings)