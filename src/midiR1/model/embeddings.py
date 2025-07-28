import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to enhance positional awareness."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Precompute inverse frequencies for efficiency
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        sinusoid = positions[:, None] * self.inv_freq[None, :]  # [seq_len, dim/2]
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
        sin = sin[None, None, :, :].expand(batch_size, num_heads, -1, -1)
        cos = cos[None, None, :, :].expand(batch_size, num_heads, -1, -1)

        # Rotate pairs of dimensions
        x_rot = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1, x2 = x_rot.unbind(dim=-1)
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.view(batch_size, num_heads, seq_len, head_dim)