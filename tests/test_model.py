import torch
from midi_r1.config import Config
from midi_r1.model.foundational import R1FoundationModel
from midi_r1.config import init_tokenizer

def test_forward():
    cfg = Config()
    tokenizer = init_tokenizer(cfg)
    vocab_size = tokenizer.vocab_size
    model = R1FoundationModel(cfg, vocab_size)
    dummy = torch.randint(0, vocab_size, (2, cfg.data.seq_len))
    mask = dummy.ne(0)
    out = model(dummy, attention_mask=mask)
    assert out.shape == (2, cfg.data.seq_len, vocab_size)