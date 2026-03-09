import torch
from torch import nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """
    SwiGLU Expert FFN (DeepSeek-V2).

    Architecture: output = W_down( SiLU(W_gate(x)) * W_up(x) )

    Used for routed MoE experts (smaller intermediate_dim),
    shared experts (full intermediate_dim), and the dense FFN
    in the first layer.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, hidden_dim)
        Returns:
            (*, hidden_dim)
        """
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))
