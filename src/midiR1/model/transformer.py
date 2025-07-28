from torch import nn
from src.midiR1.model.moe import ShareExpertMOE
from src.midiR1.model.MLA import MLA
import math


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model=512,
            n_heads=4,
            rope_theta=10_000,
            max_len=1024,
            d_expert=1024,
            n_shared=2,
            n_experts=8,
            top_k=2,
        ):
        """
        Transformer block
        :param d_model: Model hidden dimension
        :param n_heads: Number of attention heads
        :param attn_dropout: Dropout probability applied to attention weights
        :param max_position_embedding: The maximum number of positional indices supported by RoPE
        :param rope_theta: Base frequency used by RoPE
        :param v_head_dim: Dimension of value vectors per head
        :param d_kv_comp: KV compression dimension
        :param d_q_comp: Query compression dimension
        :param d_rotate: Rotary embedding dimension
        :param d_expert: Expert embedding dimension:
        :param n_shared: number of shared experts
        :param n_experts: number of routed experts
        :param top_k: top k experts per token
        """
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = MLA(
            d_model=d_model,
            n_heads=n_heads,
            rope_theta=rope_theta,
            max_len=max_len
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.moe = ShareExpertMOE(
            d_model=d_model,
            n_expert=n_experts,
            shared=n_shared,
            top_k=top_k,
        )

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        attn_out, kv = self.attn(x)
        x = x + attn_out
        x = self.norm2(x)
        moe_out, moe_loss = self.moe(x)
        x = x + moe_out
        return x, kv

