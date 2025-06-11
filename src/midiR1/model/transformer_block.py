import torch
from torch import nn
from midiR1.model.MLA import MultiHeadLatentAttention
from midiR1.model.ffn import StandardFFN, MoEFFN


class DenseBlock(nn.Module):
    """
    Layers MLA + standard FFN
    """
    def __init__(self, hidden_dim:int, latent_dim:int, dropout:float):
        super().__init__()
        self.attn = MultiHeadLatentAttention(hidden_dim, latent_dim)
        self.ffn = StandardFFN(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, key_cache=None, value_cache=None):
        attn_out, key_cache, value_cache = self.attn(self.norm1(x), freqs, key_cache, value_cache)
        h = x + attn_out
        h = h + self.ffn(self.norm2(h))
        return h, key_cache, value_cache

class MoEBlock(nn.Module):
    """
    Layers MLA + MoE FFN
    """
    def __init__(self, hidden_dim:int, latent_dim:int, moe_experts:int, top_moe_k:int):
        super().__init__()
        self.attn = MultiHeadLatentAttention(hidden_dim, latent_dim)
        self.ffn = MoEFFN(hidden_dim, moe_experts, top_moe_k)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, key_cache=None, value_cache=None):
        attn_out, key_cache, value_cache = self.attn(self.norm1(x), freqs, key_cache, value_cache)
        h = x + attn_out
        h = h + self.ffn(self.norm2(h))
        return h, key_cache, value_cache

