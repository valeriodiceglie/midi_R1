import pytest
import torch
from src.midiR1.model.model import MidiR1
from src.midiR1.model.moe import MoE


def _small_config(**overrides):
    """Minimal config for fast tests."""
    cfg = {
        "num_layers": 2,
        "hidden_dim": 64,
        "num_heads": 4,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "kv_compression_dim": 16,
        "query_compression_dim": 16,
        "num_routed_experts": 2,
        "segmentation_factor": 2,
        "num_shared_experts": 1,
        "activated_experts": 2,
        "intermediate_dim": 128,
        "first_k_dense_replace": 1,
        "moe_aux_loss_alpha": 1e-5,
        "mtp_depth": 1,
        "mtp_use_attention": True,
        "mtp_loss_weight": 0.1,
        "dropout_rate": 0.0,
        "vocab_size": 256,
        "max_seq_len": 64,
    }
    cfg.update(overrides)
    return cfg


# ── Construction tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_model_builds(device):
    """Model should instantiate without error."""
    config = _small_config()
    model = MidiR1(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_first_layer_is_dense(device):
    """First k layers should use dense FFN, remaining should use MoE."""
    config = _small_config(first_k_dense_replace=1, num_layers=3)
    model = MidiR1(config).to(device)

    # Layer 0: dense
    assert not isinstance(model.layers[0]["ffn"], MoE)
    # Layers 1, 2: MoE
    assert isinstance(model.layers[1]["ffn"], MoE)
    assert isinstance(model.layers[2]["ffn"], MoE)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_weight_sharing(device):
    """MTP should share embedding and output head weights with main model."""
    config = _small_config(mtp_depth=1)
    model = MidiR1(config).to(device)

    assert model.mtp is not None
    assert model.mtp.shared_embedding is model.embedding
    assert model.mtp.shared_output_head is model.output_head


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_disabled(device):
    """Setting mtp_depth=0 should disable MTP entirely."""
    config = _small_config(mtp_depth=0)
    model = MidiR1(config).to(device)
    assert model.mtp is None


# ── Forward pass tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_training_forward_returns_3_tuple(device):
    """Training with MTP should return (logits, mtp_logits_list, aux_loss)."""
    torch.manual_seed(0)
    config = _small_config(mtp_depth=1)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    target = torch.randint(0, config["vocab_size"], (B, T), device=device)

    out = model(input_ids, attention_mask=mask, target_ids=target)

    assert isinstance(out, tuple)
    assert len(out) == 3

    logits, mtp_list, aux_loss = out
    assert logits.shape == (B, T, config["vocab_size"])
    assert isinstance(mtp_list, list)
    assert len(mtp_list) == 1
    assert aux_loss.shape == ()


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_inference_returns_tensor(device):
    """Inference (eval mode, no target_ids) should return a single logits tensor."""
    torch.manual_seed(1)
    config = _small_config(mtp_depth=1)
    model = MidiR1(config).to(device).eval()

    B, T = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids, attention_mask=mask)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, T, config["vocab_size"])


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_no_mtp_training_returns_tensor(device):
    """Training without MTP (depth=0) returns a single logits tensor."""
    torch.manual_seed(2)
    config = _small_config(mtp_depth=0)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    target = torch.randint(0, config["vocab_size"], (B, T), device=device)

    out = model(input_ids, attention_mask=mask, target_ids=target)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, T, config["vocab_size"])


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_training_no_target_returns_tensor(device):
    """Training mode without target_ids should return just logits (no MTP)."""
    torch.manual_seed(3)
    config = _small_config(mtp_depth=2)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)

    out = model(input_ids)
    assert isinstance(out, torch.Tensor)


# ── MTP output shape tests ─────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_mtp_depth_output_shapes(device, depth):
    """MTP logits at depth k should have shape [B, T-k-1, vocab_size]."""
    torch.manual_seed(4)
    config = _small_config(mtp_depth=depth)
    V = config["vocab_size"]
    model = MidiR1(config).to(device).train()

    B, T = 2, 20
    input_ids = torch.randint(0, V, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    target = torch.randint(0, V, (B, T), device=device)

    logits, mtp_list, _ = model(input_ids, attention_mask=mask, target_ids=target)

    assert len(mtp_list) == depth
    for k, mtp_k in enumerate(mtp_list):
        expected_t = T - k - 1
        assert mtp_k.shape == (B, expected_t, V), \
            f"Depth {k}: expected ({B}, {expected_t}, {V}), got {mtp_k.shape}"


# ── Auxiliary loss tests ────────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_aux_loss_positive_training(device):
    """Aux loss should be > 0 during training with alpha > 0."""
    torch.manual_seed(5)
    config = _small_config(moe_aux_loss_alpha=1e-3)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    target = torch.randint(0, config["vocab_size"], (B, T), device=device)

    _, _, aux_loss = model(input_ids, attention_mask=mask, target_ids=target)
    assert aux_loss.item() > 0.0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_aux_loss_zero_when_disabled(device):
    """Aux loss should be 0 when alpha is 0."""
    torch.manual_seed(6)
    config = _small_config(moe_aux_loss_alpha=0.0)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    target = torch.randint(0, config["vocab_size"], (B, T), device=device)

    _, _, aux_loss = model(input_ids, attention_mask=mask, target_ids=target)
    assert aux_loss.item() == 0.0


# ── Gradient flow tests ────────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_backward_full_loss(device):
    """Full backward pass (main + MTP + aux) should produce finite gradients."""
    torch.manual_seed(7)
    config = _small_config(mtp_depth=1, moe_aux_loss_alpha=1e-3)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    V = config["vocab_size"]
    input_ids = torch.randint(0, V, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    target = torch.randint(0, V, (B, T), device=device)

    logits, mtp_list, aux_loss = model(input_ids, attention_mask=mask, target_ids=target)

    # Simulate combined loss
    main_loss = torch.nn.functional.cross_entropy(logits.view(-1, V), target.view(-1))
    mtp_loss = sum(m.sum() for m in mtp_list)
    total_loss = main_loss + 0.1 * mtp_loss + aux_loss
    total_loss.backward()

    assert torch.isfinite(total_loss).item()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), f"NaN gradient in {name}"


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_backward_no_mtp(device):
    """Backward pass without MTP should also work cleanly."""
    torch.manual_seed(8)
    config = _small_config(mtp_depth=0)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    V = config["vocab_size"]
    input_ids = torch.randint(0, V, (B, T), device=device)
    target = torch.randint(0, V, (B, T), device=device)

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, V), target.view(-1))
    loss.backward()

    assert torch.isfinite(loss).item()


# ── Determinism tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_eval_deterministic(device):
    """Same input should produce identical output in eval mode."""
    torch.manual_seed(9)
    config = _small_config()
    model = MidiR1(config).to(device).eval()

    B, T = 2, 12
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)

    with torch.no_grad():
        out1 = model(input_ids, attention_mask=mask)
        out2 = model(input_ids, attention_mask=mask)

    torch.testing.assert_close(out1, out2)


# ── Config variation tests ─────────────────────────────────────────────────


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_all_dense_layers(device):
    """Setting first_k_dense = num_layers should make all layers dense."""
    config = _small_config(num_layers=3, first_k_dense_replace=3)
    model = MidiR1(config).to(device).train()

    for i, layer in enumerate(model.layers):
        assert not isinstance(layer["ffn"], MoE), f"Layer {i} should be dense"

    B, T = 2, 10
    V = config["vocab_size"]
    input_ids = torch.randint(0, V, (B, T), device=device)
    target = torch.randint(0, V, (B, T), device=device)

    # Should work and return zero aux loss (no MoE layers)
    logits, mtp_list, aux_loss = model(input_ids, target_ids=target)
    assert aux_loss.item() == 0.0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_mtp_lite_mode_e2e(device):
    """End-to-end test with MTP in lite mode (FFN-only, no attention)."""
    torch.manual_seed(10)
    config = _small_config(mtp_depth=2, mtp_use_attention=False)
    model = MidiR1(config).to(device).train()

    B, T = 2, 16
    V = config["vocab_size"]
    input_ids = torch.randint(0, V, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.long, device=device)
    target = torch.randint(0, V, (B, T), device=device)

    logits, mtp_list, aux_loss = model(input_ids, attention_mask=mask, target_ids=target)

    assert logits.shape == (B, T, V)
    assert len(mtp_list) == 2
    assert torch.isfinite(logits).all().item()

    # Backward should work
    loss = logits.sum() + sum(m.sum() for m in mtp_list) + aux_loss
    loss.backward()
    assert torch.isfinite(loss).item()


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_no_attention_mask(device):
    """Model should work without an explicit attention mask."""
    torch.manual_seed(11)
    config = _small_config()
    model = MidiR1(config).to(device).eval()

    B, T = 2, 12
    input_ids = torch.randint(0, config["vocab_size"], (B, T), device=device)

    with torch.no_grad():
        out = model(input_ids)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, T, config["vocab_size"])
    assert torch.isfinite(out).all().item()
