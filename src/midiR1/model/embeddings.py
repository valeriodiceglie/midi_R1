import torch
import math
from torch import nn
from midiR1.utils.rotary_embedding import (
    apply_yarn_rotary,
    RotaryEmbedding
)

class Embeddings(nn.Module):
    """
    Token embeddings with two-stage YaRN decoupled RoPE.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        rotary_dim: int,
        dropout_rate: float,
        base_seq_len: int,
        stage1_seq_len: int,
        max_seq_len: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.rotary = RotaryEmbedding(rotary_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.base_seq_len = base_seq_len
        self.stage1_seq_len = stage1_seq_len
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.LongTensor):
        # Token‐embed + scale
        x = self.token_embed(input_ids) * math.sqrt(self.hidden_dim)
        seq_len = input_ids.size(1)

        # Always compute the base‐level angles once
        base_freqs = self.rotary(self.base_seq_len)  # (base_seq_len, rotary_dim)

        # Let apply_yarn_rotary figure out:
        #    - if seq_len <= base_seq_len       → slice base_freqs
        #    - elif seq_len <= stage1_seq_len   → interpolate base→stage1 then slice
        #    - else                              → interpolate base→stage1→max then slice
        freqs = apply_yarn_rotary(
            base_freqs,
            target_len=seq_len,
            stage1_len=self.stage1_seq_len,
            max_seq_len=self.max_seq_len,
        )  # (seq_len, rotary_dim)

        return self.dropout(x), freqs
