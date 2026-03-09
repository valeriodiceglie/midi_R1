import torch
from torch import nn
import torch.nn.functional as F
from src.midiR1.model.expert import ExpertFFN
import math


class MoE(nn.Module):
    """
    Mixture of Experts with auxiliary-loss-free load balancing (DeepSeek-V3).

    Architecture:
      output = sum(shared_expert_s(x) for s) + sum(g_i * routed_expert_i(x) for i in TopK)

    Routing:
      1. s_i = sigmoid(gate(x) + bias)          -- per-expert affinity
      2. TopK indices selected from s_i
      3. g_i = s_i / sum(s_j for j in TopK)     -- normalize only selected

    Load balancing (two mechanisms):
      1. Bias terms updated (no grad) based on deviation from uniform load.
      2. Complementary sequence-level auxiliary loss (DeepSeek-V3):
         L_Bal = alpha * sum_i(f_i * s_i)
         where f_i = N_i / (K*T), s_i = mean affinity for expert i.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_routed_experts: int = 6,
        num_shared_experts: int = 2,
        segmentation_factor: int = 4,
        intermediate_dim: int = 2048,
        top_k: int = 6,
        dropout_rate: float = 0.1,
        bias_update_speed: float = 0.001,
        aux_loss_alpha: float = 1e-5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha

        # Routed experts: segmented (smaller intermediate dim)
        self.num_routed = num_routed_experts * segmentation_factor
        routed_intermediate = intermediate_dim // segmentation_factor

        self.routed_experts = nn.ModuleList([
            ExpertFFN(hidden_dim, routed_intermediate, dropout_rate=dropout_rate)
            for _ in range(self.num_routed)
        ])

        # Shared experts: full intermediate dim, always active
        self.num_shared = num_shared_experts
        self.shared_experts = nn.ModuleList([
            ExpertFFN(hidden_dim, intermediate_dim, dropout_rate=dropout_rate)
            for _ in range(num_shared_experts)
        ])

        # Gating network
        self.gate = nn.Linear(hidden_dim, self.num_routed, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02 / math.sqrt(hidden_dim))

        # Bias for auxiliary-loss-free load balancing (not a trained parameter)
        self.bias = nn.Parameter(torch.zeros(self.num_routed), requires_grad=False)
        self.bias_update_speed = bias_update_speed
        self.register_buffer("expert_load_ema", torch.zeros(self.num_routed))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [N, D]
        N = x_flat.shape[0]

        # === Shared experts (always active, no gating) ===
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        # === Routing: sigmoid affinity -> TopK -> normalize ===
        logits = self.gate(x_flat)  # [N, num_routed]

        if self.training:
            scores = torch.sigmoid(logits + self.bias)
        else:
            scores = torch.sigmoid(logits)

        # Select top-K experts per token
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)  # [N, K]

        # Normalize only selected scores
        top_weights = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # === Update load balancing bias (training only, no grad) ===
        if self.training:
            with torch.no_grad():
                expert_counts = torch.zeros(self.num_routed, device=x.device)
                expert_counts.scatter_add_(
                    0,
                    top_indices.reshape(-1),
                    torch.ones(N * self.top_k, device=x.device),
                )
                self.expert_load_ema.lerp_(expert_counts, 0.1)
                target_load = N * self.top_k / self.num_routed
                self.bias.add_(self.bias_update_speed * (target_load - expert_counts))

        # === Complementary sequence-level auxiliary loss (DeepSeek-V3) ===
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training and self.aux_loss_alpha > 0.0:
            # f_i: fraction of tokens routed to each expert
            f = expert_counts / (self.top_k * N + 1e-9)          # [num_routed]
            # s_i: mean affinity score per expert (over all tokens)
            s = scores.mean(dim=0)                                # [num_routed]
            aux_loss = self.aux_loss_alpha * (f * s).sum()

        # === Dispatch to routed experts ===
        combined = torch.zeros_like(x_flat)

        # Flatten top-K selections for efficient grouping
        flat_expert_ids = top_indices.reshape(-1)           # [N*K]
        flat_weights = top_weights.reshape(-1)              # [N*K]
        flat_token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        # Sort by expert for coalesced access
        sort_order = flat_expert_ids.argsort()
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_token_ids = flat_token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]

        # Find boundaries per expert
        boundaries = torch.searchsorted(
            sorted_expert_ids.contiguous(),
            torch.arange(self.num_routed + 1, device=x.device),
        )

        for expert_idx in range(self.num_routed):
            start = boundaries[expert_idx].item()
            end = boundaries[expert_idx + 1].item()
            if start == end:
                continue

            token_ids = sorted_token_ids[start:end]
            weights = sorted_weights[start:end].unsqueeze(-1)  # [M, 1]

            expert_input = x_flat[token_ids]                    # [M, D]
            expert_output = self.routed_experts[expert_idx](expert_input)  # [M, D]

            combined.scatter_add_(
                0,
                token_ids.unsqueeze(-1).expand_as(expert_output),
                expert_output * weights,
            )

        # === Combine shared + routed ===
        output = shared_out + combined
        return output.view(batch_size, seq_len, hidden_dim), aux_loss
