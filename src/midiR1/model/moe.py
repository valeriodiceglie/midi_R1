import torch
from torch import nn
from src.midiR1.model.expert import ExpertFFN
import math
import torch.nn.functional as fn

class MoE(nn.Module):
    """Mixture of Experts (MoE) with auxiliary-loss-free load balancing."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Add dropout
        self.dropout = nn.Dropout(dropout_rate)

        self.shared_expert = ExpertFFN(hidden_dim, dropout_rate=dropout_rate)
        self.experts = nn.ModuleList([ExpertFFN(hidden_dim, dropout_rate=dropout_rate) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_experts))  # For load balancing
        self.bias_update_speed = 0.001
        self.register_buffer("expert_load", torch.zeros(num_experts))

        # Initialize gate with smaller variance
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02 / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]

        # Compute gating scores with bias for routing
        scores = fn.sigmoid(self.gate(x_flat) + self.bias)  # [bs * seq_len, num_experts]
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)  # [bs * seq_len, top_k]

        # Normalize top scores for weighting
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-6)

        # Update load balancing bias (keep your original implementation)
        mask = fn.one_hot(top_indices, self.num_experts).sum(dim=1).float()  # [bs * seq_len, num_experts]
        expert_load = mask.sum(dim=0)  # [num_experts]
        self.bias.data += self.bias_update_speed * (expert_load - self.expert_load)
        self.expert_load.lerp_(expert_load, 0.1)  # Exponential moving average

        # Apply experts with proper weighting
        combined = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_indices = top_indices[:, i]  # [batch_size * seq_len]
            coefficient = top_scores[:, i].unsqueeze(-1)  # [batch_size * seq_len, 1]

            # Process inputs for each expert
            for expert_idx, expert in enumerate(self.experts):
                # Find inputs that should go to this expert
                mask = (expert_indices == expert_idx)
                if mask.any():
                    # Get inputs for this expert
                    expert_inputs = x_flat[mask]
                    # Process inputs and apply coefficient
                    expert_outputs = expert(expert_inputs) * coefficient[mask]
                    # Add outputs to the combined tensor
                    combined.index_add_(0, torch.where(mask)[0], expert_outputs)

        # Add shared expert output
        shared_out = self.shared_expert(x_flat) * 0.1  # Scale shared output
        combined = combined + shared_out

        # Apply additional dropout to final output
        combined = self.dropout(combined)

        # Reshape back to original dimensions
        return combined.view(batch_size, seq_len, hidden_dim)