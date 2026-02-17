"""
Implements learnable positional embeddings for a decoder-only Transformer.

Why?
Transformers have no recurrence or convolution.
They have no inherent sense of token order.
So we inject position information manually.

We use a learnable embedding table of size:
(max_seq_len, embed_dim)

For input shape:
    (B, T)
We generate:
    token_embedding  -> (B, T, C)
    position_embedding -> (T, C)
Then broadcast and add:
    output -> (B, T, C)
"""

import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, dim_model):
        """
        :param max_seq_length : Context window
        :param dim_model: Size of each tokens vector representation
        """
        super().__init__()
        # Learnable position embeddings
        # Shape: (max_seq_len, embed_dim)
        self.positional_embedding=nn.Embedding(num_embeddings=max_seq_length, embedding_dim=dim_model)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Create position indices
        positions=torch.arange(0, seq_len, device=x.device)
        positions=positions.unsqueeze(0).expand(batch_size, seq_len)
        
        return self.positional_embedding(positions)
        