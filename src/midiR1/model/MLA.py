import torch
from torch import nn
import math
from midiR1.utils.rotary_embedding import apply_decoupled_rope


class MultiHeadLatentAttention(nn.Module):
    """
    MLA: low-rank QKV down-projection + up-projection + decoupled RoPE.
    """
    def __init__(self, hidden_dim:int, d_latent:int):
        super().__init__()
        # fused down projection for QKV
        self.w_dkv = nn.Linear(hidden_dim, 2 * d_latent, bias=False)
        self.w_dq = nn.Linear(hidden_dim, d_latent, bias=False)
        # up projections
        self.w_uk = nn.Linear(d_latent, hidden_dim, bias=False)
        self.w_uv = nn.Linear(d_latent, hidden_dim, bias=False)
        self.w_uq = nn.Linear(d_latent, hidden_dim, bias=False)
        # output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, key_cache=None, value_cache=None):
        # x: (batch, seq_len, d_hidden)
        # freqs: rotary embeddings
        # Down-proj
        dkv = self.w_dkv(x)  # (batch, seq_len, 2*d_latent)
        k_latent, v_latent = dkv.chunk(2, dim=-1)
        q_latent = self.w_dq(x)
        # Up-proj
        k = self.w_uk(k_latent)
        v = self.w_uv(v_latent)
        q = self.w_uq(q_latent)
        # apply decoupled RoPE to separate Q and K
        q_rope = apply_decoupled_rope(q, freqs)
        k_rope = apply_decoupled_rope(k, freqs)
        # scaled dot-product attention fallback
        attn_scores = torch.einsum('bqd,bkd->bqk', q_rope, k_rope) / math.sqrt(q_rope.size(-1))
        if key_cache is None:
            key_cache = k_rope
            value_cache = v
        else:
        # only append last timestep
            key_cache = torch.cat([key_cache, k_rope[:, -1:, :]], dim=1)
            value_cache = torch.cat([value_cache, v[:, -1:, :]], dim=1)
        # scaled dot-product
        # for inference we only compute new q_rope[:,-1:] vs full key_cache
        q_query = q_rope[:, -1:, :]  # (b,1,d)
        attn_scores = torch.einsum('bqd,bkd->bqk', q_query, key_cache) / math.sqrt(q_query.size(-1))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum('bqk,bkd->bqd', attn_probs, value_cache)
        # output projection
        out = self.out_proj(out)
        return out, key_cache, value_cache
