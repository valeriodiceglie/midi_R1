import torch
from torch import nn


class StandardFFN(nn.Module):
    """Simple two-layer FFN with activation."""
    def __init__(self, hidden_dim:int, dropout:float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)

class MoEFFN(nn.Module):
    """
    MoE with learned centroids and sigmoid routing.
    """
    def __init__(self, hidden_dim:int, moe_experts:int, top_moe_k:int):
        super().__init__()
        self.moe_top_k = top_moe_k
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim))
            for _ in range(moe_experts)
        ])
        # shared expert index 0 is shared; others are individual
        self.centroids = nn.Parameter(torch.randn(moe_experts, hidden_dim))

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, d)
        # compute affinities
        # (batch*seq_len, d)
        b, s, d = x.size()
        u = x.view(b*s, d)                               # (B*S, D)
        # router
        scores = u @ self.centroids.t()                  # (B*S, E)
        gates = torch.sigmoid(scores)
        # keep only top-k
        if self.moe_top_k < gates.size(-1):
            topk_vals, topk_idx = gates.topk(self.moe_top_k, dim=-1)
            mask = torch.zeros_like(gates).scatter_(-1, topk_idx, topk_vals)
            gates = mask / mask.sum(dim=-1, keepdim=True)
        # expert outputs
        # stack: (B*S, E, D)
        all_outs = torch.stack([expert(u) for expert in self.experts], dim=1)
        # weighted sum: (B*S, D)
        out = torch.einsum('be,bed->bd', gates, all_outs)
        return out.view(b, s, d)
