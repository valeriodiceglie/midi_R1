
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, Optional, Tuple


RoPEStyle = Literal["neox", "interleaved"]


def rotate_half(x: torch.Tensor, rope_style: RoPEStyle = "neox") -> torch.Tensor:
    """
    Rotate last-dim pairs by 90 degrees:
      (a, b) -> (-b, a)

    Two common tensor layouts exist:
      - "neox" (split-half / real-imag halves): x = [a0..aN, b0..bN]
      - "interleaved" (GPT-J): x = [a0,b0,a1,b1,...]

    If you pick a style here, you MUST build cos/sin in the matching style.
    """
    if rope_style == "neox":
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # interleaved
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)
    return x_rot


def apply_rope_x(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_style: RoPEStyle = "neox",
) -> torch.Tensor:
    """
    Apply RoPE to x (broadcasting cos/sin).
    Expected x shape: (..., T, D_rope) or (..., D_rope) depending on cos/sin shapes.
    """
    return (x * cos) + (rotate_half(x, rope_style=rope_style) * sin)


@dataclass
class RoPECached:
    cos: torch.Tensor  # (1, 1, T, D)
    sin: torch.Tensor  # (1, 1, T, D)
    max_seq_len: int
    device: torch.device
    dtype: torch.dtype


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) cache generator.

    This module produces cos/sin tensors that are broadcastable to:
      - (B, H, T, D_rope)  (typical attention tensors)
      - (B, 1, T, D_rope)  (shared-key RoPE, e.g., DeepSeek-V2 decoupled k^R)

    It supports two layout conventions:
      - rope_style="neox"       : split-half (x = [real..., imag...])
      - rope_style="interleaved": interleaved (x = [real0, imag0, real1, imag1, ...])

    Pick ONE style and use it consistently across your entire model.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        rope_style: RoPEStyle = "neox",
        max_seq_len: int = 2048,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got dim={dim}")

        self.dim = int(dim)
        self.base = float(base)
        self.rope_style = rope_style

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cache: Optional[RoPECached] = None
        self._ensure_cache(max_seq_len=max_seq_len, device=torch.device("cpu"), dtype=torch.float32)

    def _build_angles(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # angles: (T, dim/2) where angles[p, i] = p * inv_freq[i]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv = self.inv_freq.to(device=device, dtype=torch.float32)
        return torch.einsum("t,f->tf", t, inv)

    def _angles_to_cos_sin(
        self, angles: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        angles: (T, dim/2)

        For "neox" (split-half):
          emb = [angles, angles] -> (T, dim)

        For "interleaved":
          emb = repeat_interleave(angles, 2) -> (T, dim) with [a0,a0,a1,a1,...]
        """
        if self.rope_style == "neox":
            emb = torch.cat([angles, angles], dim=-1)  # (T, dim)
        else:
            emb = angles.repeat_interleave(2, dim=-1)  # (T, dim)

        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        # (1,1,T,dim) so it broadcasts over (B,H)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def _ensure_cache(self, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._cache is not None
            and self._cache.max_seq_len >= max_seq_len
            and self._cache.device == device
            and self._cache.dtype == dtype
        ):
            return

        angles = self._build_angles(max_seq_len, device=device)
        cos, sin = self._angles_to_cos_sin(angles, dtype=dtype)

        self._cache = RoPECached(
            cos=cos,
            sin=sin,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos/sin for contiguous positions [offset, offset+seq_len).

        Shapes: (1, 1, seq_len, dim)
        """
        needed = offset + seq_len
        self._ensure_cache(max_seq_len=needed, device=device, dtype=dtype)
        assert self._cache is not None
        cos = self._cache.cos[:, :, offset:offset + seq_len, :]
        sin = self._cache.sin[:, :, offset:offset + seq_len, :]
        return cos, sin

    @torch.no_grad()
    def get_cos_sin_from_position_ids(
        self,
        position_ids: torch.LongTensor,  # (B,T) or (T,)
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos/sin for arbitrary positions.

        Output shapes: (B, 1, T, dim)
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        max_pos = int(position_ids.max().item()) + 1
        self._ensure_cache(max_seq_len=max_pos, device=device, dtype=dtype)
        assert self._cache is not None
        cos = self._cache.cos[:, :, position_ids, :].squeeze(0)  # (1,B,T,dim) -> (B,1,T,dim)
        sin = self._cache.sin[:, :, position_ids, :].squeeze(0)
        return cos, sin
