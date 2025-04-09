from typing import List, Union, Literal
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from .base import Embedding

class BERTModel(Embedding):
    """BERT-based embedding model with configurable output dimension."""

    def __init__(self, 
                 model_name: str,
                 output_dim: int = 768,
                 pooling: Literal['cls', 'mean', 'max'] = 'cls'):
        """
        Initialize BERT embedding model.
        
        Args:
            model_name: Name of pretrained BERT model
            output_dim: Desired embedding dimension (default: 768)
            pooling: Token pooling strategy ('cls', 'mean', or 'max')
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.pooling = pooling
        self.output_dim = output_dim
        
        # Add dimension reduction if needed
        if output_dim != self.model.config.hidden_size:
            self.dim_reduction = nn.Linear(
                self.model.config.hidden_size, 
                output_dim
            ).to(self.device)
        else:
            self.dim_reduction = None

    def _pool_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings using specified strategy."""
        if self.pooling == 'cls':
            return token_embeddings[:, 0, :]  # [CLS] token
        elif self.pooling == 'mean':
            return token_embeddings.mean(dim=1)  # Average all tokens
        elif self.pooling == 'max':
            return token_embeddings.max(dim=1)[0]  # Max pooling
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

    def embed(self, data: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input text(s)."""
        if isinstance(data, str):
            data = [data]
            
        all_embeddings = []
        batch_size = 32
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Pool tokens
                embeddings = self._pool_tokens(outputs.last_hidden_state)
                
                # Reduce dimension if needed
                if self.dim_reduction is not None:
                    embeddings = self.dim_reduction(embeddings)
                    
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)