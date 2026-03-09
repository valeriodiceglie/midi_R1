import torch
from src.midiR1.utils.rotary_embedding import RotaryEmbedding, apply_rope_x

def test_rope_identity_at_position0_neox():
    torch.manual_seed(0)
    rope_dim = 8
    rope = RotaryEmbedding(dim=rope_dim, rope_style="neox", max_seq_len=16)

    B, H, T = 2, 3, 5
    x = torch.randn(B, H, T, rope_dim, dtype=torch.float32)

    cos, sin = rope.get_cos_sin(seq_len=T, device=x.device, dtype=x.dtype, offset=0)
    y = apply_rope_x(x, cos, sin, rope_style="neox")

    # p=0 => RoPE should be identity
    torch.testing.assert_close(y[:, :, 0, :], x[:, :, 0, :], atol=1e-6, rtol=1e-6)


def test_rope_pairwise_norm_preserved_neox():
    torch.manual_seed(1)
    rope_dim = 8
    rope = RotaryEmbedding(dim=rope_dim, rope_style="neox", max_seq_len=32)

    B, H, T = 2, 3, 9
    x = torch.randn(B, H, T, rope_dim, dtype=torch.float32)

    cos, sin = rope.get_cos_sin(seq_len=T, device=x.device, dtype=x.dtype, offset=0)
    y = apply_rope_x(x, cos, sin, rope_style="neox")

    # For "neox" layout, pairs are (i, i + D/2)
    D2 = rope_dim // 2
    x1, x2 = x[..., :D2], x[..., D2:]
    y1, y2 = y[..., :D2], y[..., D2:]

    # Pairwise squared norm should be preserved at every (B,H,T,i)
    x_pair_norm2 = x1.pow(2) + x2.pow(2)
    y_pair_norm2 = y1.pow(2) + y2.pow(2)
    torch.testing.assert_close(x_pair_norm2, y_pair_norm2, atol=1e-5, rtol=1e-5)


def test_rope_matches_explicit_rotation_one_pair_neox():
    torch.manual_seed(2)
    rope_dim = 8
    rope = RotaryEmbedding(dim=rope_dim, rope_style="neox", max_seq_len=64)

    B, H, T = 2, 3, 11
    x = torch.randn(B, H, T, rope_dim, dtype=torch.float32)

    cos, sin = rope.get_cos_sin(seq_len=T, device=x.device, dtype=x.dtype, offset=0)
    y = apply_rope_x(x, cos, sin, rope_style="neox")

    # Pick one position p and one pair index i
    p = 7
    i = 2
    D2 = rope_dim // 2

    a = x[:, :, p, i]          # (B,H)
    b = x[:, :, p, i + D2]     # (B,H)

    c = cos[0, 0, p, i]        # scalar
    s = sin[0, 0, p, i]        # scalar

    # Explicit rotation:
    # [a'; b'] = [a cos - b sin; a sin + b cos]
    a_exp = a * c - b * s
    b_exp = a * s + b * c

    torch.testing.assert_close(y[:, :, p, i], a_exp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y[:, :, p, i + D2], b_exp, atol=1e-5, rtol=1e-5)
