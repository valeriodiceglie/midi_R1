
import torch
import torch.nn as nn
import torch.nn.functional as fn
from typing import Optional, Tuple
from dataclasses import dataclass
from src.midiR1.utils.rms_norm import DeepseekV3RMSNorm
from src.midiR1.utils.rotary_embedding import RotaryEmbedding, apply_rope_x


@dataclass
class MLACache:
    c_kv: torch.Tensor
    k_rope: torch.Tensor

    def append(self, c_kv_new: torch.Tensor, k_rope_new: torch.Tensor) -> "MLACache":
        self.c_kv = torch.cat([self.c_kv, c_kv_new], dim=1)
        self.k_rope = torch.cat([self.k_rope, k_rope_new], dim=1)
        return self

    def trim(self, max_len: int) -> "MLACache":
        """Trim cache to max_len entries (discards positions beyond max_len)."""
        self.c_kv = self.c_kv[:, :max_len, :]
        self.k_rope = self.k_rope[:, :max_len, :]
        return self

class MultiHeadLatentAttention(nn.Module):
    """Implements Multi-head Latent Attention (MLA)"""

    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 qk_nope_head_dim: int,
                 qk_rope_head_dim: int,
                 kv_compression_dim: int,
                 query_compression_dim: int,
                 v_head_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 bias: bool = False,
                 latent_layernorm: bool = True,
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.d_nope = qk_nope_head_dim
        self.d_rope = qk_rope_head_dim
        self.d_qk = self.d_nope + self.d_rope
        self.d_c_kv = kv_compression_dim
        self.d_c_q = query_compression_dim
        self.d_v = v_head_dim if v_head_dim is not None else self.d_nope
        self.dropout = dropout
        
        if self.d_rope % 2 != 0:
            raise ValueError("Rotary dimension must be even for RoPE to work correctly")
        
        self.dropout = float(dropout)
        self.latent_layernorm = latent_layernorm

        # Q low-rank compression
        self.q_down = nn.Linear(hidden_dim, self.d_c_q, bias=bias)
        self.q_norm = DeepseekV3RMSNorm(self.d_c_q) if latent_layernorm else nn.Identity()

        # Q up-projections
        self.qc_up = nn.Linear(self.d_c_q, n_heads * self.d_nope, bias=bias)
        self.qr_up = nn.Linear(self.d_c_q, n_heads * self.d_rope, bias=bias)

        # KV low-rank compression
        self.kv_a = nn.Linear(hidden_dim, self.d_c_kv, bias=bias)
        self.k_rope = nn.Linear(hidden_dim, self.d_rope, bias=bias)
        self.kv_norm = DeepseekV3RMSNorm(self.d_c_kv) if latent_layernorm else nn.Identity()
        self.kc_up = nn.Linear(self.d_c_kv, n_heads * self.d_nope, bias=bias)
        self.v_up = nn.Linear(self.d_c_kv, n_heads * self.d_v, bias=bias)

        self.o_proj = nn.Linear(n_heads * self.d_v, hidden_dim, bias=False)
        self.scaling = float(self.d_qk ** -0.5)

    def _shape_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        return x.view(batch_size, seq_length, self.n_heads, head_dim).transpose(1, 2).contiguous()
    
    @staticmethod
    def _causal_allow_with_prefix(T_q: int, T_k: int, past_len: int, device) -> torch.Tensor:
        """
        Builds allow-mask for causal attention when we have a prefix in K/V cache.
        Query positions correspond to absolute indices [past_len .. past_len+T_q-1]
        Keys correspond to absolute indices [0 .. T_k-1]
        Allow iff key_index <= query_abs_index
        Output shape: (1,1,T_q,T_k) boolean allow-mask.
        """
        q_abs = past_len + torch.arange(T_q, device=device)[:, None]
        k_idx = torch.arange(T_k, device=device)[None, :]
        allow = (k_idx <= q_abs)
        return allow.view(1, 1, T_q, T_k)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_kv: Optional[MLACache] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        cos, sin = position_embeddings
        device = hidden_states.device
        past_len = 0 if past_kv is None else past_kv.c_kv.shape[1]

        # Q
        c_q = self.q_norm(self.q_down(hidden_states))
        q_c = self._shape_heads(self.qc_up(c_q), self.d_nope)
        q_r = self._shape_heads(self.qr_up(c_q), self.d_rope)
        q_r = apply_rope_x(q_r, cos, sin)
        q = torch.cat([q_c, q_r], dim=-1)

        # KV
        c_kv_new = self.kv_norm(self.kv_a(hidden_states))
        k_rope_pre_new = self.k_rope(hidden_states)
        k_rope_new = apply_rope_x(k_rope_pre_new.unsqueeze(1), cos, sin).squeeze(1)

        new_cache = None
        if use_cache:
            if past_kv is None:
                new_cache = MLACache(c_kv=c_kv_new, k_rope=k_rope_new)
            else:
                new_cache = past_kv.append(c_kv_new, k_rope_new)

        if past_kv is None:
            c_kv = c_kv_new
            k_rope = k_rope_new
        else:
            c_kv = torch.cat([past_kv.c_kv, c_kv_new], dim=1)
            k_rope = torch.cat([past_kv.k_rope, k_rope_new], dim=1)

        k_c = self._shape_heads(self.kc_up(c_kv), self.d_nope)
        v = self._shape_heads(self.v_up(c_kv), self.d_v)
        k_r = k_rope.unsqueeze(1).expand(batch_size, self.n_heads, c_kv.shape[1], self.d_rope)
        k = torch.cat([k_c, k_r], dim=-1)

        # Attention mask handling for causal attention with prefix
        allow_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                allow_pad = attention_mask.to(torch.bool).view(batch_size, 1, 1, c_kv.shape[1]).expand(batch_size, self.n_heads, seq_length, c_kv.shape[1])
            else:
                allow_pad = attention_mask.to(torch.bool)
            allow_mask = allow_pad

        if past_len == 0:
            causal_allow = None
        else:
            causal_allow = self._causal_allow_with_prefix(seq_length, c_kv.shape[1], past_len, device)
            causal_allow = causal_allow.expand(batch_size, self.n_heads, seq_length, c_kv.shape[1])
            allow_mask = causal_allow if allow_mask is None else (allow_mask & causal_allow)

        if output_attentions:
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling

            if past_len == 0:
                causal = torch.tril(torch.ones(seq_length, c_kv.shape[1], device=device, dtype=torch.bool)).view(1, 1, seq_length, c_kv.shape[1])
                causal = causal.expand(batch_size, self.n_heads, seq_length, c_kv.shape[1])
                allow = causal if allow_mask is None else (allow_mask & causal)
            else:
                allow = allow_mask

            scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn = fn.dropout(attn, p=self.dropout)
            out = torch.matmul(attn, v)
            attn_weights = attn
        else:
            if past_len == 0:
                if allow_mask is not None:
                    # Cannot combine is_causal=True with attn_mask in PyTorch SDPA.
                    # Build causal mask manually and combine with the padding mask.
                    causal = torch.tril(
                        torch.ones(seq_length, c_kv.shape[1], dtype=torch.bool, device=device)
                    ).view(1, 1, seq_length, c_kv.shape[1])
                    combined_mask = allow_mask & causal
                    out = fn.scaled_dot_product_attention(
                        query=q,
                        key=k,
                        value=v,
                        attn_mask=combined_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=False,
                        scale=self.scaling,
                    )
                else:
                    out = fn.scaled_dot_product_attention(
                        query=q,
                        key=k,
                        value=v,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                        scale=self.scaling,
                    )
            else:
                out = fn.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=allow_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                    scale=self.scaling,
                )
            attn_weights = None

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.n_heads * self.d_v)
        out = self.o_proj(out)
        return out, attn_weights, new_cache