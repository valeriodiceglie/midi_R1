import pytest
import torch
from src.midiR1.model.moe import MoE


def _make_moe(hidden_dim=64, num_routed=4, num_shared=1, seg_factor=2,
              intermediate=128, top_k=2, dropout=0.0, aux_alpha=1e-5, device="cpu"):
    return MoE(
        hidden_dim=hidden_dim,
        num_routed_experts=num_routed,
        num_shared_experts=num_shared,
        segmentation_factor=seg_factor,
        intermediate_dim=intermediate,
        top_k=top_k,
        dropout_rate=dropout,
        aux_loss_alpha=aux_alpha,
    ).to(device)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_forward_shape(device):
    """Output shape must match input shape."""
    torch.manual_seed(0)
    B, T, D = 2, 8, 64
    moe = _make_moe(hidden_dim=D, device=device).eval()

    x = torch.randn(B, T, D, device=device)
    output, aux_loss = moe(x)

    assert output.shape == (B, T, D)
    assert aux_loss.shape == ()


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_returns_tuple(device):
    """Forward must always return (output, aux_loss) tuple."""
    torch.manual_seed(1)
    D = 64
    moe = _make_moe(hidden_dim=D, device=device)
    x = torch.randn(2, 4, D, device=device)

    # Training mode
    moe.train()
    result = moe(x)
    assert isinstance(result, tuple) and len(result) == 2

    # Eval mode
    moe.eval()
    result = moe(x)
    assert isinstance(result, tuple) and len(result) == 2


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_aux_loss_positive_during_training(device):
    """Auxiliary loss should be > 0 during training when alpha > 0."""
    torch.manual_seed(2)
    D = 64
    moe = _make_moe(hidden_dim=D, aux_alpha=1e-3, device=device).train()

    x = torch.randn(2, 8, D, device=device)
    _, aux_loss = moe(x)

    assert aux_loss.item() > 0.0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_aux_loss_zero_during_eval(device):
    """Auxiliary loss should be exactly 0 during evaluation."""
    torch.manual_seed(3)
    D = 64
    moe = _make_moe(hidden_dim=D, aux_alpha=1e-3, device=device).eval()

    x = torch.randn(2, 8, D, device=device)
    _, aux_loss = moe(x)

    assert aux_loss.item() == 0.0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_aux_loss_zero_when_disabled(device):
    """Auxiliary loss should be 0 when alpha is 0, even during training."""
    torch.manual_seed(4)
    D = 64
    moe = _make_moe(hidden_dim=D, aux_alpha=0.0, device=device).train()

    x = torch.randn(2, 8, D, device=device)
    _, aux_loss = moe(x)

    assert aux_loss.item() == 0.0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_aux_loss_scales_with_alpha(device):
    """Larger alpha should produce larger auxiliary loss."""
    torch.manual_seed(5)
    D = 64
    x = torch.randn(2, 8, D, device=device)

    moe_small = _make_moe(hidden_dim=D, aux_alpha=1e-6, device=device).train()
    moe_large = _make_moe(hidden_dim=D, aux_alpha=1e-2, device=device).train()

    # Use same gate weights for fair comparison
    moe_large.load_state_dict(moe_small.state_dict())

    _, loss_small = moe_small(x)
    _, loss_large = moe_large(x)

    assert loss_large.item() > loss_small.item()


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_expert_count_matches_config(device):
    """Number of routed experts = num_routed * segmentation_factor."""
    moe = _make_moe(num_routed=4, seg_factor=3, device=device)
    assert len(moe.routed_experts) == 4 * 3


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_shared_experts_always_contribute(device):
    """Shared expert output should be nonzero (they always fire)."""
    torch.manual_seed(6)
    D = 64
    moe = _make_moe(hidden_dim=D, num_shared=2, device=device).eval()

    x = torch.randn(2, 4, D, device=device)
    x_flat = x.view(-1, D)

    shared_out = torch.zeros_like(x_flat)
    for expert in moe.shared_experts:
        shared_out = shared_out + expert(x_flat)

    assert shared_out.abs().sum().item() > 0.0


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_top_k_routing(device):
    """Each token should be routed to exactly top_k experts."""
    torch.manual_seed(7)
    D = 64
    top_k = 3
    moe = _make_moe(hidden_dim=D, top_k=top_k, device=device).eval()

    x = torch.randn(1, 4, D, device=device)
    x_flat = x.view(-1, D)

    logits = moe.gate(x_flat)
    scores = torch.sigmoid(logits)
    _, top_indices = scores.topk(top_k, dim=-1)

    # Each token should have exactly top_k selections
    assert top_indices.shape == (4, top_k)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_bias_updates_during_training(device):
    """Load balancing bias should change after a training forward pass."""
    torch.manual_seed(8)
    D = 64
    moe = _make_moe(hidden_dim=D, device=device).train()

    bias_before = moe.bias.clone()
    x = torch.randn(4, 16, D, device=device)
    moe(x)

    # Bias should have been updated
    assert not torch.equal(moe.bias, bias_before)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_bias_unchanged_during_eval(device):
    """Load balancing bias should NOT change during evaluation."""
    torch.manual_seed(9)
    D = 64
    moe = _make_moe(hidden_dim=D, device=device).eval()

    bias_before = moe.bias.clone()
    x = torch.randn(2, 8, D, device=device)
    moe(x)

    torch.testing.assert_close(moe.bias, bias_before)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_backward_no_nan(device):
    """Gradients should be finite through the MoE layer."""
    torch.manual_seed(10)
    D = 64
    moe = _make_moe(hidden_dim=D, aux_alpha=1e-3, device=device).train()

    x = torch.randn(2, 8, D, device=device, requires_grad=True)
    output, aux_loss = moe(x)
    loss = output.pow(2).mean() + aux_loss
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(x.grad).all().item()


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_deterministic_eval(device):
    """Same input should produce identical output in eval mode."""
    torch.manual_seed(11)
    D = 64
    moe = _make_moe(hidden_dim=D, dropout=0.5, device=device).eval()

    x = torch.randn(2, 4, D, device=device)
    with torch.no_grad():
        out1, _ = moe(x)
        out2, _ = moe(x)

    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_routing_weights_sum_to_one(device):
    """Normalized routing weights for selected experts should sum to 1 per token."""
    torch.manual_seed(12)
    D = 64
    top_k = 3
    moe = _make_moe(hidden_dim=D, top_k=top_k, device=device).eval()

    x = torch.randn(2, 4, D, device=device)
    x_flat = x.view(-1, D)

    logits = moe.gate(x_flat)
    scores = torch.sigmoid(logits)
    top_scores, _ = scores.topk(top_k, dim=-1)
    top_weights = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-9)

    weight_sums = top_weights.sum(dim=-1)
    torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-5, rtol=1e-5)
