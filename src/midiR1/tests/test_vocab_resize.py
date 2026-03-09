import pytest
import torch

from src.midiR1.model.model import MidiR1
from src.midiR1.utils.vocab_resize import resize_model_vocab


# ── helpers ───────────────────────────────────────────────────────────────


def _make_small_model(vocab_size=100):
    """Build a minimal MidiR1 for testing."""
    config = {
        "num_layers": 1,
        "hidden_dim": 32,
        "num_heads": 2,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "kv_compression_dim": 16,
        "query_compression_dim": 16,
        "num_routed_experts": 2,
        "segmentation_factor": 1,
        "num_shared_experts": 1,
        "activated_experts": 2,
        "first_k_dense_replace": 1,
        "mtp_depth": 1,
        "mtp_use_attention": False,
        "dropout_rate": 0.0,
        "vocab_size": vocab_size,
        "max_seq_len": 64,
        "min_seq_len": 5,
    }
    return MidiR1(config)


# ── tests ─────────────────────────────────────────────────────────────────


def test_old_weights_preserved():
    model = _make_small_model(vocab_size=100)
    old_emb = model.embedding.weight.data.clone()
    old_head = model.output_head.weight.data.clone()

    resize_model_vocab(model, 116)

    assert model.embedding.weight.shape[0] == 116
    assert model.output_head.weight.shape[0] == 116
    assert torch.equal(model.embedding.weight.data[:100], old_emb)
    assert torch.equal(model.output_head.weight.data[:100], old_head)


def test_new_rows_initialised():
    model = _make_small_model(vocab_size=100)
    resize_model_vocab(model, 116, init_method="normal")

    new_emb_rows = model.embedding.weight.data[100:]
    # Very unlikely all 16*32 values are exactly 0 after normal init
    assert new_emb_rows.abs().sum() > 0


def test_mean_init():
    model = _make_small_model(vocab_size=100)
    old_mean = model.embedding.weight.data.mean(dim=0)
    resize_model_vocab(model, 116, init_method="mean")

    # New rows should be initialised to the mean (before old weights are copied)
    # After copy, new rows [100:116] should be the mean
    new_row = model.embedding.weight.data[100]
    assert torch.allclose(new_row, old_mean, atol=1e-6)


def test_mtp_references_updated():
    model = _make_small_model(vocab_size=100)
    resize_model_vocab(model, 116)

    assert model.mtp.shared_embedding is model.embedding
    assert model.mtp.shared_output_head is model.output_head


def test_config_updated():
    model = _make_small_model(vocab_size=100)
    resize_model_vocab(model, 116)
    assert model.config["vocab_size"] == 116


def test_noop_same_size():
    model = _make_small_model(vocab_size=100)
    old_emb = model.embedding
    resize_model_vocab(model, 100)
    assert model.embedding is old_emb  # same object, no resize


def test_shrink_raises():
    model = _make_small_model(vocab_size=100)
    with pytest.raises(ValueError, match="Cannot shrink"):
        resize_model_vocab(model, 50)


def test_forward_after_resize():
    """Model should still produce valid outputs after resize."""
    model = _make_small_model(vocab_size=100)
    resize_model_vocab(model, 116)
    model.eval()

    input_ids = torch.randint(0, 116, (1, 10))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, 10, 116)
