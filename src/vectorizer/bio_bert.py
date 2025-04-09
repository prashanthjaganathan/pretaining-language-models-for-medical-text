from typing import List, Tuple, Union, Literal
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from .base import Embedding

class BioBERTModel(Embedding):
    """BioBERT-based embedding model with configurable output dimension for abbreviation disambiguation."""

    def __init__(self, 
                 model_name: str,
                 output_dim: int = 768,
                 pooling: Literal['cls', 'mean', 'max'] = 'mean'):
        """
        Initialize BioBERT embedding model.
        
        Args:
            model_name: Name of pretrained BioBERT model (e.g., 'dmis-lab/biobert-base-cased-v1.1')
            output_dim: Desired embedding dimension (default: 768)
            pooling: Token pooling strategy ('cls', 'mean', or 'max')
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.pooling = pooling
        self.output_dim = output_dim
        
        # Add dimension reduction if needed. Autoencoder implementation
        if output_dim != self.model.config.hidden_size:
            self.dim_reduction = nn.Linear(
                self.model.config.hidden_size, 
                output_dim
            ).to(self.device)
        else:
            self.dim_reduction = None

    def _pool_tokens(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool token/abbr embeddings using specified strategy"""
        if self.pooling == 'cls':
            return token_embeddings[:, 0, :]  # [CLS] token (full context)
        elif self.pooling == 'mean':
            return token_embeddings.mean(dim=1)  # Average all tokens
        elif self.pooling == 'max':
            return token_embeddings.max(dim=1)[0]  # Max pooling
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")


    def embed(self, data: Union[str, List[str]], abbreviations: Union[str, List[str]], ids: List[int]) -> List[dict]:
        """
        Generate embeddings for input text(s), focusing on abbreviation disambiguation.
        
        Args:
            data: A single sentence or a list of sentences.
            abbreviations: A single abbreviation or a list of abbreviations (must match the input sentence).
        
        Returns:
            List of dicts containing context and abbreviation embeddings.
        """
        if isinstance(data, str):
            data = [data]
        if isinstance(abbreviations, str):
            abbreviations = [abbreviations]
        if isinstance(ids, int):
            ids = [ids]
        

        all_embeddings = {
            'abstract_ids': [],
            'context_embeddings': [],
            'abbreviation_embeddings': [],
        }

        batch_size = 64 # TODO: pull from config
        
        for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", unit="batch"):
            batch = data[i:i + batch_size]
            abbr_batch = abbreviations[i:i + batch_size]
            ids_batch = ids[i: i+batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get model outputs for entire batch
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                
                # Get abbreviation indices for all sentences in batch
                abbr_indices = [self._get_abbreviation_token_index(inputs, abbr, idx) 
                            for idx, abbr in enumerate(abbr_batch)]
                
                # Extract abbreviation embeddings for entire batch
                abbreviation_embeddings = []
                for idx, (abbr_idx, abbr_len) in enumerate(abbr_indices):
                    if abbr_idx is not None:
                        # Select token embeddings for the abbreviation
                        abbr_tokens = token_embeddings[idx, abbr_idx:abbr_idx+abbr_len, :]
                        # Pool the abbreviation tokens
                        abbr_embedding = self._pool_tokens(abbr_tokens.unsqueeze(0))
                        abbreviation_embeddings.append(abbr_embedding)
                    else:
                        # Fallback for cases where abbreviation is not found
                        abbreviation_embeddings.append(torch.zeros(1, self.model.config.hidden_size).to(self.device))
                
                abbreviation_embeddings = torch.cat(abbreviation_embeddings, dim=0)
                
                # Extract context embeddings for entire batch
                context_embeddings = self._pool_tokens(token_embeddings)
                
                # Reduce dimensionality if needed
                if self.dim_reduction is not None:
                    abbreviation_embeddings = self.dim_reduction(abbreviation_embeddings)
                    context_embeddings = self.dim_reduction(context_embeddings)
                
                # Convert to numpy and store results
                all_embeddings['context_embeddings'].extend(context_embeddings)
                all_embeddings['abbreviation_embeddings'].extend(abbreviation_embeddings)
                all_embeddings['abstract_ids'].extend(ids_batch)
        
        return all_embeddings # returns a dict of abtract_ids, abbreviation_embeddings, and context_embeddings tensors

    def _get_abbreviation_token_index(self, inputs: torch.Tensor, abbreviation: str, batch_idx: int) -> Tuple[int, int]:
        """
        Get the token index of the abbreviation in the tokenized input sentence.
        
        Args:
            inputs: Tokenized sentence inputs.
            abbreviation: Abbreviation to find in the sentence.
            batch_idx: Index of the current sentence in the batch.
        
        Returns:
            int: Index of the token in the tokenized sentence corresponding to the abbreviation.
        """
        # Tokenize the abbreviation to find its subwords
        abbr_tokens = self.tokenizer.tokenize(abbreviation)
        token_ids = inputs['input_ids'][batch_idx]
        
        # Find the matching token(s) in the tokenized sentence
        for idx, token_id in enumerate(token_ids):
            token = self.tokenizer.decode([token_id])
            if token in abbr_tokens:
                # print(f'Found {token} in {abbr_tokens}')
                return idx, len(abbr_tokens)  # Return the index of the abbreviation token and it's size
        return None  # If abbreviation not found, return None (handle error later)
