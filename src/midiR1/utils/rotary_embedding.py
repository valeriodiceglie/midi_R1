import torch
from torch import nn

def stretch_angles(base_freqs: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Linearly interpolate base_freqs from shape (base_len, rotary_dim)
    to (target_len, rotary_dim) using a single GPU‐accelerated call.
    """
    # Transpose → (1, channels, length) for interpolate
    #   channels = rotary_dim, length = base_len
    freqs = base_freqs.transpose(0, 1).unsqueeze(0)  # (1, rotary_dim, base_len)

    # Interpolate to new length
    # align_corners=True ensures the first and last elements map exactly
    stretched = nn.functional.interpolate(
        freqs,
        size=target_len,
        mode="linear",
        align_corners=True
    )  # (1, rotary_dim, target_len)

    # Restore shape (target_len, rotary_dim)
    return stretched.squeeze(0).transpose(0, 1)

def apply_yarn_rotary(
    base_freqs: torch.Tensor,
    target_len: int,
    stage1_len: int,
    max_seq_len: int
) -> torch.Tensor:
    """
    Two‐stage YaRN interpolation of rotary frequencies:
    1) If target_len <= base_len, slice base_freqs.
    2) Else interpolate to stage1_len, then slice or further interpolate to max_seq_len.
    Returns a (target_len, rotary_dim) tensor of angles.
    """
    base_len, rotary_dim = base_freqs.shape
    # 1) short sequence: just slice
    if target_len <= base_len:
        return base_freqs[:target_len]
    # 2) first stretch to stage1_len
    freqs = stretch_angles(base_freqs, stage1_len)
    if target_len <= stage1_len:
        return freqs[:target_len]
    # 3) stretch to max_seq_len
    freqs = stretch_angles(freqs, max_seq_len)
    return freqs[:target_len]


def apply_decoupled_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to the first rotary_dim features of x using decoupled sin/cos.
    x: (batch, seq_len, hidden_dim)
    freqs: (seq_len, rotary_dim)
    """
    batch, seq_len, hidden_dim = x.size()
    rotary_dim = freqs.size(-1)
    half = rotary_dim // 2

    # split into rotating and remainder
    x_rot = x[..., :rotary_dim].view(batch, seq_len, half, 2)
    x_rem = x[..., rotary_dim:]

    # get freqs for half dims
    freqs_half = freqs[:, :half]                      # (seq_len, half)
    cos = torch.cos(freqs_half).unsqueeze(0)           # (1, seq_len, half)
    sin = torch.sin(freqs_half).unsqueeze(0)           # (1, seq_len, half)

    x1 = x_rot[..., 0]                                 # (batch, seq_len, half)
    x2 = x_rot[..., 1]                                 # (batch, seq_len, half)

    # perform rotation
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).view(batch, seq_len, rotary_dim)

    return torch.cat([x_rotated, x_rem], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Generates rotary positional embeddings for a given rotary_dim.
    """
    def __init__(self, rotary_dim: int):
        super().__init__()
        # compute inverse frequencies for half-dimensions
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        )  # shape: (rotary_dim//2,)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> torch.Tensor:
        # seq positions
        t = torch.arange(seq_len, device=self.inv_freq.device,
                         dtype=torch.float32)
        # compute (seq_len, rotary_dim//2)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # expand to (seq_len, rotary_dim)
        return torch.cat([freqs, freqs], dim=-1)