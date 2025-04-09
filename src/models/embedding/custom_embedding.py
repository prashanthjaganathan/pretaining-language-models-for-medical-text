import torch
import torch.nn as nn

class CustomEmbedding(nn.Module):
    """
    Base class for custom embedding models. This class should be subclassed for custom behavior.
    """
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
    def get_embeddings(self, data: torch.Tensor):
        """
        This method should be overridden by subclasses to define how embeddings are generated.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def build_embeddings(self, data: torch.Tensor):
        """
        This method should be overridden by subclasses to build embeddings from the data.
        """
        raise NotImplementedError("Subclasses should implement this method.")
