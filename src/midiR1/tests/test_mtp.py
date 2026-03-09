import pytest
import torch
import torch.nn as nn
from src.midiR1.model.mtp import MTPBlock, MultiTokenPrediction
from src.midiR1.utils.rotary_embedding import RotaryEmbedding


def _base_config(hidden_dim=64, mtp_depth=2, mtp_use_attention=True):
    return {
        "hidden_dim": hidden_dim,
        "num_heads": 4,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "kv_compression_dim": 16,
        "query_compression_dim": 16,
        "intermediate_dim": 128,
        "dropout_rate": 0.0,
        "vocab_size": 256,
        "max_seq_len": 64,
        "mtp_depth": mtp_depth,
        "mtp_use_attention": mtp_use_attention,
    }


def _make_rope_cos_sin(seq_len, dim, device, dtype=torch.float32):
    rope = RotaryEmbedding(dim=dim, max_seq_len=seq_len)
    return rope.get_cos_sin(seq_len=seq_len, device=device, dtype=dtype)


# ── MTPBlock tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
@pytest.mark.parametrize("use_attention", [True, False])
def test_block_forward_shape(device, use_attention):
    """MTPBlock output shape must match [B, T', D]."""
    torch.manual_seed(0)
    D = 64
    config = _base_config(hidden_dim=D, mtp_use_attention=use_attention)
    block = MTPBlock(config, use_attention=use_attention).to(device).eval()

    B, T_eff = 2, 10
    prev_hidden = torch.randn(B, T_eff, D, device=device)
    token_emb = torch.randn(B, T_eff, D, device=device)
    cos, sin = _make_rope_cos_sin(T_eff, config["qk_rope_head_dim"], device)

    out = block(prev_hidden, token_emb, (cos, sin))
    assert out.shape == (B, T_eff, D)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_block_lite_has_no_attention(device):
    """Lite block (use_attention=False) should not have attention attribute."""
    config = _base_config()
    block = MTPBlock(config, use_attention=False).to(device)
    assert not hasattr(block, "attention")


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_block_full_has_attention(device):
    """Full block (use_attention=True) should have attention attribute."""
    config = _base_config()
    block = MTPBlock(config, use_attention=True).to(device)
    assert hasattr(block, "attention")


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_block_backward_no_nan(device):
    """Gradients through MTPBlock should be finite."""
    torch.manual_seed(1)
    D = 64
    config = _base_config(hidden_dim=D)
    block = MTPBlock(config, use_attention=True).to(device).train()

    B, T_eff = 2, 8
    prev_hidden = torch.randn(B, T_eff, D, device=device, requires_grad=True)
    token_emb = torch.randn(B, T_eff, D, device=device, requires_grad=True)
    cos, sin = _make_rope_cos_sin(T_eff, config["qk_rope_head_dim"], device)

    out = block(prev_hidden, token_emb, (cos, sin))
    loss = out.pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(prev_hidden.grad).all().item()
    assert torch.isfinite(token_emb.grad).all().item()


# ── MultiTokenPrediction tests ─────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_mtp_returns_correct_depth_count(device, depth):
    """MTP should return exactly `depth` logit tensors."""
    torch.manual_seed(2)
    D = 64
    config = _base_config(hidden_dim=D, mtp_depth=depth)

    embedding = nn.Embedding(config["vocab_size"], D).to(device)
    output_head = nn.Linear(D, config["vocab_size"]).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device).eval()

    B, T = 2, 16
    main_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)
    cos, sin = _make_rope_cos_sin(T, config["qk_rope_head_dim"], device)

    with torch.no_grad():
        logits_list = mtp(main_hidden, input_ids, (cos, sin))

    assert len(logits_list) == depth


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_output_shapes_per_depth(device):
    """Each MTP depth k should produce logits of shape [B, T-k-1, vocab_size]."""
    torch.manual_seed(3)
    D = 64
    depth = 3
    config = _base_config(hidden_dim=D, mtp_depth=depth)
    V = config["vocab_size"]

    embedding = nn.Embedding(V, D).to(device)
    output_head = nn.Linear(D, V).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device).eval()

    B, T = 2, 16
    main_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.randint(0, V, (B, T), device=device)
    cos, sin = _make_rope_cos_sin(T, config["qk_rope_head_dim"], device)

    with torch.no_grad():
        logits_list = mtp(main_hidden, input_ids, (cos, sin))

    for k, logits_k in enumerate(logits_list):
        expected_t = T - k - 1
        assert logits_k.shape == (B, expected_t, V), \
            f"Depth {k}: expected ({B}, {expected_t}, {V}), got {logits_k.shape}"


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_weight_sharing_embedding(device):
    """MTP should share the same embedding object (not a copy)."""
    D = 64
    config = _base_config(hidden_dim=D)
    embedding = nn.Embedding(config["vocab_size"], D).to(device)
    output_head = nn.Linear(D, config["vocab_size"]).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device)

    assert mtp.shared_embedding is embedding


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_weight_sharing_output_head(device):
    """MTP should share the same output head object (not a copy)."""
    D = 64
    config = _base_config(hidden_dim=D)
    embedding = nn.Embedding(config["vocab_size"], D).to(device)
    output_head = nn.Linear(D, config["vocab_size"]).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device)

    assert mtp.shared_output_head is output_head


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_weight_sharing_gradient_flow(device):
    """Gradients from MTP loss should flow back to the shared embedding."""
    torch.manual_seed(4)
    D = 64
    config = _base_config(hidden_dim=D, mtp_depth=1)
    V = config["vocab_size"]

    embedding = nn.Embedding(V, D).to(device)
    output_head = nn.Linear(D, V).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device).train()

    B, T = 2, 12
    main_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.randint(0, V, (B, T), device=device)
    cos, sin = _make_rope_cos_sin(T, config["qk_rope_head_dim"], device)

    logits_list = mtp(main_hidden, input_ids, (cos, sin))
    loss = sum(l.sum() for l in logits_list)
    loss.backward()

    # Shared embedding should receive gradients
    assert embedding.weight.grad is not None
    assert embedding.weight.grad.abs().sum().item() > 0.0

    # Shared output head should receive gradients
    assert output_head.weight.grad is not None
    assert output_head.weight.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
@pytest.mark.parametrize("use_attention", [True, False])
def test_mtp_full_vs_lite_both_work(device, use_attention):
    """Both full (MLA+FFN) and lite (FFN-only) modes should produce valid output."""
    torch.manual_seed(5)
    D = 64
    config = _base_config(hidden_dim=D, mtp_depth=1, mtp_use_attention=use_attention)
    V = config["vocab_size"]

    embedding = nn.Embedding(V, D).to(device)
    output_head = nn.Linear(D, V).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device).eval()

    B, T = 2, 10
    main_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.randint(0, V, (B, T), device=device)
    cos, sin = _make_rope_cos_sin(T, config["qk_rope_head_dim"], device)

    with torch.no_grad():
        logits_list = mtp(main_hidden, input_ids, (cos, sin))

    assert len(logits_list) == 1
    assert logits_list[0].shape == (B, T - 1, V)
    assert torch.isfinite(logits_list[0]).all().item()


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_lite_fewer_params_than_full(device):
    """Lite mode should have fewer parameters than full mode (no MLA)."""
    D = 64
    config_full = _base_config(hidden_dim=D, mtp_depth=1, mtp_use_attention=True)
    config_lite = _base_config(hidden_dim=D, mtp_depth=1, mtp_use_attention=False)
    V = config_full["vocab_size"]

    embedding = nn.Embedding(V, D)
    output_head = nn.Linear(D, V)

    mtp_full = MultiTokenPrediction(config_full, embedding, output_head)
    mtp_lite = MultiTokenPrediction(config_lite, embedding, output_head)

    full_params = sum(p.numel() for p in mtp_full.blocks.parameters())
    lite_params = sum(p.numel() for p in mtp_lite.blocks.parameters())

    assert lite_params < full_params


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_short_sequence_graceful(device):
    """MTP with sequence shorter than depth should return fewer logit tensors."""
    torch.manual_seed(6)
    D = 64
    depth = 5
    config = _base_config(hidden_dim=D, mtp_depth=depth)
    V = config["vocab_size"]

    embedding = nn.Embedding(V, D).to(device)
    output_head = nn.Linear(D, V).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device).eval()

    B, T = 2, 3  # T=3 means only depths 0,1 can produce output (t_eff=2,1)
    main_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.randint(0, V, (B, T), device=device)
    cos, sin = _make_rope_cos_sin(T, config["qk_rope_head_dim"], device)

    with torch.no_grad():
        logits_list = mtp(main_hidden, input_ids, (cos, sin))

    # With T=3 and depth=5, only 2 depths can produce output (t_eff=2 and t_eff=1)
    assert len(logits_list) <= T - 1
    assert len(logits_list) == min(depth, T - 1)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_backward_no_nan(device):
    """Full backward pass through MTP should produce finite gradients."""
    torch.manual_seed(7)
    D = 64
    config = _base_config(hidden_dim=D, mtp_depth=2)
    V = config["vocab_size"]

    embedding = nn.Embedding(V, D).to(device)
    output_head = nn.Linear(D, V).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device).train()

    B, T = 2, 12
    main_hidden = torch.randn(B, T, D, device=device, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T), device=device)
    cos, sin = _make_rope_cos_sin(T, config["qk_rope_head_dim"], device)

    logits_list = mtp(main_hidden, input_ids, (cos, sin))
    loss = sum(l.pow(2).mean() for l in logits_list)
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(main_hidden.grad).all().item()

    for name, p in mtp.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), f"NaN gradient in {name}"


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_with_attention_mask(device):
    """MTP should accept and pass through attention masks without error."""
    torch.manual_seed(8)
    D = 64
    config = _base_config(hidden_dim=D, mtp_depth=1)
    V = config["vocab_size"]

    embedding = nn.Embedding(V, D).to(device)
    output_head = nn.Linear(D, V).to(device)
    mtp = MultiTokenPrediction(config, embedding, output_head).to(device).eval()

    B, T = 2, 10
    main_hidden = torch.randn(B, T, D, device=device)
    input_ids = torch.randint(0, V, (B, T), device=device)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    attention_mask[:, -2:] = 0  # mask last 2 positions
    cos, sin = _make_rope_cos_sin(T, config["qk_rope_head_dim"], device)

    with torch.no_grad():
        logits_list = mtp(main_hidden, input_ids, (cos, sin), attention_mask=attention_mask)

    assert len(logits_list) == 1
    assert torch.isfinite(logits_list[0]).all().item()
