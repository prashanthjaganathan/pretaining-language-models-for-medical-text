from abc import ABC, abstractmethod
from typing import List, Union, Any
import numpy as np
import torch

class Embedding(ABC):
    """Base class for all embedding models."""
    
    @abstractmethod
    def embed(self, tokenized_data: Union[str, List[str]]) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate embeddings for input text(s).
        
        Args:
            texts: Input text or list of texts to embed
            
        Returns:
            numpy array or torch tensor of embeddings
        """
        pass

    @staticmethod
    def build_doc_embedding(doc_embeddings, aggregation_method: str):
        
        doc_embeddings = np.array(doc_embeddings)

        if aggregation_method == "average":
            doc_embedding = doc_embeddings.mean(axis=0)

        elif aggregation_method == "max":
            doc_embedding = doc_embeddings.max(axis=0)

        elif aggregation_method == "sum":
            doc_embedding = doc_embeddings.sum(axis=0)

        elif aggregation_method == "normal":
            # Apply normalization (L2-normalization)
            doc_embedding = doc_embeddings.mean(axis=0)
            doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)

        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")


        return np.array(doc_embedding)