import pytest
import torch
from src.midiR1.model.expert import ExpertFFN


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_forward_shape(device):
    """Output shape must match input shape on the hidden dimension."""
    torch.manual_seed(0)
    hidden_dim, intermediate_dim = 64, 128
    ffn = ExpertFFN(hidden_dim, intermediate_dim, dropout_rate=0.0).to(device).eval()

    x = torch.randn(2, 7, hidden_dim, device=device)
    out = ffn(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_forward_2d_input(device):
    """ExpertFFN should also work with 2-D input [N, D] (flattened batch)."""
    torch.manual_seed(1)
    hidden_dim, intermediate_dim = 32, 64
    ffn = ExpertFFN(hidden_dim, intermediate_dim, dropout_rate=0.0).to(device).eval()

    x = torch.randn(10, hidden_dim, device=device)
    out = ffn(x)
    assert out.shape == (10, hidden_dim)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_backward_no_nan(device):
    """Gradients should be finite after backward pass."""
    torch.manual_seed(2)
    hidden_dim, intermediate_dim = 64, 128
    ffn = ExpertFFN(hidden_dim, intermediate_dim, dropout_rate=0.0).to(device).train()

    x = torch.randn(2, 5, hidden_dim, device=device, requires_grad=True)
    out = ffn(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert torch.isfinite(loss).item()
    assert torch.isfinite(x.grad).all().item()
    for p in ffn.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all().item()


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_deterministic_eval(device):
    """Same input should produce identical output in eval mode (no dropout)."""
    torch.manual_seed(3)
    hidden_dim, intermediate_dim = 32, 64
    ffn = ExpertFFN(hidden_dim, intermediate_dim, dropout_rate=0.5).to(device).eval()

    x = torch.randn(2, 4, hidden_dim, device=device)
    with torch.no_grad():
        out1 = ffn(x)
        out2 = ffn(x)

    torch.testing.assert_close(out1, out2)


@pytest.mark.parametrize("device", (["cuda"] if torch.cuda.is_available() else ["cpu"]))
def test_swiglu_structure(device):
    """Verify the SwiGLU gating: output = W_down( SiLU(W_gate(x)) * W_up(x) )."""
    torch.manual_seed(4)
    hidden_dim, intermediate_dim = 16, 32
    ffn = ExpertFFN(hidden_dim, intermediate_dim, dropout_rate=0.0).to(device).eval()

    x = torch.randn(1, 3, hidden_dim, device=device)
    with torch.no_grad():
        gate_out = torch.nn.functional.silu(ffn.gate_proj(x))
        up_out = ffn.up_proj(x)
        expected = ffn.down_proj(gate_out * up_out)
        actual = ffn(x)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
