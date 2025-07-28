import torch
from torch import nn
import torch.nn.functional as fn

class ExpertFFN(nn.Module):
    """Feed-Forward Network (FFN) for MoE experts."""
    def __init__(self, hidden_dim: int, expansion_factor: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        intermediate_dim = hidden_dim * expansion_factor
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        # Add dropout after activation
        self.dropout = nn.Dropout(dropout_rate)
        self.down = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout after activation
        return self.down(self.dropout(self.gelu(self.up(x))))