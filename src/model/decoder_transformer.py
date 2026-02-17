"""
Implements a Decoder-Only Transformer.

Architecture:

Input tokens -> Token Embedding
            -> Positional Embedding
            -> N Transformer Blocks
            -> Final LayerNorm
            -> Linear projection to vocab

Output:
    logits of shape (B, T, vocab_size)

Important Shapes:
    B = batch size
    T = sequence length
"""

import torch
import torch.nn as nn
from src.model.positional_embedding import PositionalEmbedding
from src.model.transformer import TransformerBlock

class DecoderTransformer(nn.Module):
    def __init__(self, vocabulary_size, max_seq_len=128, dim_model=256, num_layers=4, num_heads=4, dim_ff=1024, dropout=0.1):
        super().__init__()

        self.token_embedding=nn.Embedding(vocabulary_size, dim_model)
        self.positional_embedding=PositionalEmbedding(max_seq_len, dim_model)

        self.blocks=nn.ModuleList([
            TransformerBlock(dim_model, num_heads=num_heads, dim_ff=dim_ff, dropout=dropout) for _  in range(num_layers)
        ])

        self.ln_f=nn.LayerNorm(dim_model)
        self.head=nn.Linear(dim_model, vocabulary_size)

        self.max_seq_len=max_seq_len

    def generate_causal_mask(self, seq_length, device):
        """
        Creates upper triangular matrix:
        Prevents attending to future tokens.

        Shape: (T, T)
        """
        mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device),
            diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits: (B, T, vocab_size)
        """
        batch_size, seq_length = x.size()

        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(x)

        x = token_emb + pos_emb

        attn_mask = self.generate_causal_mask(seq_length, x.device)

        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits