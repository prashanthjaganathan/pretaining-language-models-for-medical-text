import torch.nn as nn
from abc import ABC, abstractmethod


class Embedding(ABC, nn.Module):
    """
    Base class for all embedding models. It provides methods to create and return embeddings.
    """

    def __init__(self):
        super(Embedding, self).__init__()

    
    @abstractmethod
    def embed(self, data):
        """
        Given some data, return the embeddings (either precomputed or generated).
        """
        pass