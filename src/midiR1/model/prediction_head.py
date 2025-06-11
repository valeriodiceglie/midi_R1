import torch
from torch import nn


class PredictionHead(nn.Module):
    """
    Single MTP head with transformer block
    """
    
    def __init__(self, vocab_size:int, hidden_dim:int, n_heads:int, dropout:float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation='gelu'
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden: torch.Tensor, input_embeds: torch.Tensor):
        # hidden: (batch, seq, hidden)
        # input_embeds: (batch, seq, hidden)
        x = self.norm(hidden)
        x = torch.cat([x, input_embeds], dim=-1)
        x = self.proj(x)
        x = x.permute(1, 0, 2)
        x = self.block(x)
        x = x.permute(1, 0, 2)
        return x, self.out(x)
