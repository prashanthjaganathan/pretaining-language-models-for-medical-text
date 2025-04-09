import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)  # Linear layer to compute attention score for each hidden state

    def forward(self, lstm_out, mask=None):
        # lstm_out is of shape (batch_size, seq_len, hidden_dim)
        
        # Apply mask if provided (mask is of shape (batch_size, seq_len))
        if mask is not None:
            # Expand mask to match the hidden_dim * 2 size
            mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            lstm_out = lstm_out * mask  # Apply mask: (batch_size, seq_len, hidden_dim * 2)
        
        # Compute attention scores
        attn_scores = self.attn(lstm_out)  # (batch_size, seq_len, 1)
        
        # Normalize scores using softmax (along the sequence length)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Compute the weighted sum of the LSTM outputs (context vector)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        return context_vector


class LSTM_SelfAttention(nn.Module):
    def __init__(
            self, 
            embedding_dim,
            embedding_model, 
            lstm_hidden_dim, 
            lstm_units, 
            num_classes, 
            dropout=0.5,
            create_embedding_layer: bool = False, 
            max_seq_len=None
            ):
        """
        Initialize the LSTM with Self-Attention model, and embed input using pre-trained GloVe embeddings, trainable with nn.Embedding.
        
        Args:
            embedding_model (GloVeEmbedding): The GloVe embedding model containing pre-trained GloVe embeddings.
            lstm_hidden_dim (int): The hidden dimension of the LSTM.
            lstm_units (int): The number of layers in the LSTM.
            num_classes (int): The number of classes for the classification task.
            dropout (float, optional): Dropout probability (default is 0.5).
            max_seq_len (int, optional): Maximum sequence length for padding/truncation.
        """
        super().__init__()

        self.create_embedding_layer = create_embedding_layer

        if create_embedding_layer:
            self.embedding = nn.Embedding(
                num_embeddings=len(embedding_model.word_to_idx),  # Vocabulary size
                embedding_dim=embedding_model.embedding_dim,  # Embedding dimension
                padding_idx=embedding_model.word_to_idx.get("<PAD>", None),  # Optional: specify padding token index
                _freeze=False
            )


        # LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_units, batch_first=True, bidirectional=True)

        # Self-Attention Layer
        self.attention = SelfAttention(lstm_hidden_dim * 2)  # BiLSTM doubles the hidden size

        # Fully Connected Layer for classification
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 1024),
            nn.Linear(1024, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Optional: Max sequence length (used for padding/truncation)
        self.max_seq_len = max_seq_len


    def forward(self, x, mask=None):
        """
        Forward pass through the LSTM with Self-Attention mechanism.
        
        Args:
            x (torch.Tensor): The input tensor (batch_size, seq_len).
            mask (torch.Tensor, optional): The attention mask indicating valid tokens (batch_size, seq_len).
        
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        x = x.long()

        if self.create_embedding_layer:
            embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

            # Apply padding if needed
            if self.max_seq_len is not None:
                seq_len = embedded.size(1)
                if seq_len < self.max_seq_len:
                    pad = torch.zeros((embedded.size(0), self.max_seq_len - seq_len, embedded.size(2)), dtype=torch.float32)
                    embedded = torch.cat([embedded, pad], dim=1)
                elif seq_len > self.max_seq_len:
                    embedded = embedded[:, :self.max_seq_len, :]

        else:
            embedded = x

        # LSTM Output: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)

        # Apply Self-Attention
        context_vector = self.attention(lstm_out, mask)

        # Classification using the context vector
        output = self.fc(self.dropout(context_vector))  # (batch_size, num_classes)

        return output
