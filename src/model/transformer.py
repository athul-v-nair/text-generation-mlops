"""
Core building blocks of a decoder-only Transformer:

1. Multi-Head Self Attention
2. Feed Forward Network
3. Transformer Block (Pre-LN architecture)

Important Shapes:
B = batch size
T = sequence length
C = embedding dimension
H = number of heads
head_dim = C // H

Attention math:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
"""

import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim_model, num_heads, dropout, dim_ff):
        super().__init__()

        self.ln1=nn.LayerNorm(dim_model)
        self.attn=nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2=nn.LayerNorm(dim_model)
        self.ff=nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, dim_model),
            nn.Dropout(dropout)
        )

        self.dropout1 = nn.Dropout(dropout)  # For attention residual


    # Pre-LN	Norm → Attention → Add
    def forward(self,x,attn_mask):
        # Pre Layer Normalization Attention
        x_norm=self.ln1(x)
        attn_output,_ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False
        )

        x=x + self.dropout1(attn_output) 

        # Pre-LN feed forward
        x_norm=self.ln2(x)
        ff_output=self.ff(x_norm)
        
        x=x+ff_output

        return x