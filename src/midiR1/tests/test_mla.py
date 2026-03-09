import pytest
import torch
from src.midiR1.model.MLA import MultiHeadLatentAttention, MLACache


def make_rope_cos_sin(seq_len: int, dim: int, device, dtype):
    """
    Creates broadcastable cos/sin tensors for RoPE with shape (1, 1, T, dim).
    Works with typical apply_rotary_pos_emb(x, cos, sin) that expects broadcasting.
    """
    assert dim % 2 == 0
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (T, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    # Broadcast over (B, H)
    return cos.view(1, 1, seq_len, dim), sin.view(1, 1, seq_len, dim)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_smoke_forward_shapes(device):
    torch.manual_seed(0)

    B, T, hidden_dim = 2, 7, 64
    n_heads = 4
    d_nope = 8
    d_rope = 8
    d_c_kv = 16
    d_c_q = 16

    mla = MultiHeadLatentAttention(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        qk_nope_head_dim=d_nope,
        qk_rope_head_dim=d_rope,
        kv_compression_dim=d_c_kv,
        query_compression_dim=d_c_q,
        dropout=0.0,
        bias=True,
        latent_layernorm=False,
    ).to(device).eval()

    x = torch.randn(B, T, hidden_dim, device=device, dtype=torch.float32)
    cos, sin = make_rope_cos_sin(T, d_rope, device, x.dtype)

    out, attn, cache = mla(x, (cos, sin), use_cache=True, output_attentions=False)

    assert out.shape == (B, T, hidden_dim)
    assert attn is None
    assert cache is not None
    assert cache.c_kv.shape == (B, T, d_c_kv)
    # k_rope is shared across heads => no head dimension stored
    assert cache.k_rope.shape == (B, T, d_rope)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_full_vs_token_by_token_cache_equivalence(device):
    """
    Core correctness test:
      forward on full sequence == concatenation of step-wise cached decoding outputs.
    This validates:
      - causal-with-prefix correctness
      - cache concat order
      - post-RoPE key caching behavior
      - shared k^R broadcasting
    """
    torch.manual_seed(1)

    B, T, hidden_dim = 2, 9, 64
    n_heads = 4
    d_nope = 8
    d_rope = 8
    d_c_kv = 16
    d_c_q = 16

    mla = MultiHeadLatentAttention(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        qk_nope_head_dim=d_nope,
        qk_rope_head_dim=d_rope,
        kv_compression_dim=d_c_kv,
        query_compression_dim=d_c_q,
        dropout=0.0,
        bias=True,
        latent_layernorm=False,
    ).to(device).eval()

    x = torch.randn(B, T, hidden_dim, device=device, dtype=torch.float32)
    cos_full, sin_full = make_rope_cos_sin(T, d_rope, device, x.dtype)

    with torch.no_grad():
        out_full, _, _ = mla(x, (cos_full, sin_full), use_cache=False)

        # token-by-token with cache
        cache = None
        outs = []
        for t in range(T):
            cos_t = cos_full[:, :, t:t+1, :]  # keep (1,1,1,d_rope)
            sin_t = sin_full[:, :, t:t+1, :]
            out_t, _, cache = mla(x[:, t:t+1, :], (cos_t, sin_t), past_kv=cache, use_cache=True)
            outs.append(out_t)

        out_step = torch.cat(outs, dim=1)

    torch.testing.assert_close(out_step, out_full, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_full_vs_chunked_cache_equivalence(device):
    """
    Same as above, but in chunks (prefix + continuation).
    """
    torch.manual_seed(2)

    B, T, hidden_dim = 2, 12, 64
    n_heads = 4
    d_nope = 8
    d_rope = 8
    d_c_kv = 16
    d_c_q = 16
    split = 5

    mla = MultiHeadLatentAttention(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        qk_nope_head_dim=d_nope,
        qk_rope_head_dim=d_rope,
        kv_compression_dim=d_c_kv,
        query_compression_dim=d_c_q,
        dropout=0.0,
        bias=True,
        latent_layernorm=False,
    ).to(device).eval()

    x = torch.randn(B, T, hidden_dim, device=device, dtype=torch.float32)
    cos_full, sin_full = make_rope_cos_sin(T, d_rope, device, x.dtype)

    with torch.no_grad():
        out_full, _, _ = mla(x, (cos_full, sin_full), use_cache=False)

        cache = None
        out1, _, cache = mla(x[:, :split, :], (cos_full[:, :, :split, :], sin_full[:, :, :split, :]),
                             past_kv=cache, use_cache=True)
        out2, _, cache = mla(x[:, split:, :], (cos_full[:, :, split:, :], sin_full[:, :, split:, :]),
                             past_kv=cache, use_cache=True)
        out_chunked = torch.cat([out1, out2], dim=1)

    torch.testing.assert_close(out_chunked, out_full, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_output_attentions_properties(device):
    """
    Validate that attention weights are well-formed:
      - correct shape
      - sums to ~1 on last dim for each query position
      - respects causality (future positions masked)
    """
    torch.manual_seed(3)

    B, T, hidden_dim = 1, 6, 64
    n_heads = 2
    d_nope = 8
    d_rope = 8
    d_c_kv = 16
    d_c_q = 16

    mla = MultiHeadLatentAttention(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        qk_nope_head_dim=d_nope,
        qk_rope_head_dim=d_rope,
        kv_compression_dim=d_c_kv,
        query_compression_dim=d_c_q,
        dropout=0.0,
        bias=True,
        latent_layernorm=False,
    ).to(device).eval()

    x = torch.randn(B, T, hidden_dim, device=device, dtype=torch.float32)
    cos, sin = make_rope_cos_sin(T, d_rope, device, x.dtype)

    with torch.no_grad():
        out, attn, _ = mla(x, (cos, sin), use_cache=False, output_attentions=True)

    assert attn is not None
    assert attn.shape == (B, n_heads, T, T)

    # rows sum to 1 (within numerical tolerance)
    row_sum = attn.sum(dim=-1)
    torch.testing.assert_close(row_sum, torch.ones_like(row_sum), atol=1e-5, rtol=1e-5)

    # causality: for each query position i, attention to j>i should be ~0
    for i in range(T):
        future_mass = attn[..., i, i+1:].sum()
        assert float(future_mass) < 1e-5


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_backward_no_nan(device):
    """
    Gradients should flow without NaNs/Infs.
    """
    torch.manual_seed(4)

    B, T, hidden_dim = 2, 8, 64
    n_heads = 4
    d_nope = 8
    d_rope = 8
    d_c_kv = 16
    d_c_q = 16

    mla = MultiHeadLatentAttention(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        qk_nope_head_dim=d_nope,
        qk_rope_head_dim=d_rope,
        kv_compression_dim=d_c_kv,
        query_compression_dim=d_c_q,
        dropout=0.0,
        bias=True,
        latent_layernorm=False,
    ).to(device).train()

    x = torch.randn(B, T, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)
    cos, sin = make_rope_cos_sin(T, d_rope, device, x.dtype)

    out, _, _ = mla(x, (cos, sin), use_cache=False)
    loss = out.pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(x.grad).all().item()
