from transformers import AutoModel, AutoTokenizer
import torch
from .base import Embedding

class PretrainedEmbedding(Embedding):
    """
    A class that uses a pretrained embedding model from HuggingFace Transformers (e.g., BERT).
    """

    def __init__(self, model_name: str, tokenizer=None):
        super(PretrainedEmbedding, self).__init__()
        # Load the pretrained model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        if not tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

    def build_embeddings(self, data):
        """
        Given a dataset (list of text), build embeddings using a pretrained model.
        """
        # Tokenize the data (list of sentences)
        encoding = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Get the embeddings (last hidden state)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state

    def get_embeddings(self, data):
        """
        Given a list of text, return embeddings for that text using a pretrained model.
        """
        encoding = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state