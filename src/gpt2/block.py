import torch
import torch.nn as nn
from .mlp import MLP
from .attention import CausalSelfAttention


class Block(nn.Module):
    """
    A single transformer block.
    This contains the self-attention and the feed-forward network.
    They are both preceeded by a layer normalization.
    The output of the self attention is fed into the feed-forward network.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
