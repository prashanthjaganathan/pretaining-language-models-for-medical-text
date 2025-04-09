from typing import List, Union
import torch
from transformers import AutoModel, AutoTokenizer
from .base import Embedding
import os
import gensim
import numpy as np


class GoogleNewsModel(Embedding):
    """Google News Word2Vec embedding model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self.load_model(model_path)
        self.aggregation_method = 'avergae' # TODO: Tie with config.yaml

    def load_model(self, model_path: str):
        if os.path.exists(model_path):
            self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        else:
            raise ValueError(f'Unable to find model at path: {model_path}')

    def embed(self, tokenized_data: List[List[str]]):
        all_embeddings = []

        for doc in tokenized_data:
            doc_embeddings = []
            for token in doc:
                if token in self.model:
                    doc_embeddings.append(self.model[token])
                else:
                    # For OOV tokens, you can assign zeros or some default vector
                    # doc_embeddings.append(np.zeros(self.model.vector_size))
                    doc_embeddings.append(self.model.get_word_vector('[UNK]'))
            
            doc_embedding = Embedding.build_doc_embedding(doc_embeddings, self.aggregation_method)
            all_embeddings.append(doc_embedding)

        return np.array(all_embeddings)