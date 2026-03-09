import torch
from torch import nn
from typing import Optional, Tuple
from src.midiR1.model.MLA import MultiHeadLatentAttention
from src.midiR1.model.expert import ExpertFFN
from src.midiR1.utils.rms_norm import DeepseekV3RMSNorm


class MTPBlock(nn.Module):
    """
    A single Multi-Token Prediction depth block (DeepSeek-V3).

    For depth k, this block:
      1. Concatenates h^{k-1} with Emb(token_{t+k}) -> [B, T', 2*D]
      2. Projects down: Linear(2*D, D) -> RMSNorm -> h'
      3. Transforms: TRM(h') -> h^k   (full = MLA+FFN, lite = FFN only)
      4. Predicts: shared_norm(h^k) -> shared_output_head -> logits
    """

    def __init__(
        self,
        config: dict,
        use_attention: bool = True,
    ):
        super().__init__()
        hidden_dim = config["hidden_dim"]
        dropout_rate = config.get("dropout_rate", 0.1)

        # Projection: concat(h^{k-1}, emb) -> hidden_dim
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.proj_norm = DeepseekV3RMSNorm(hidden_dim)

        # Transformer block
        self.use_attention = use_attention
        if use_attention:
            self.attn_norm = DeepseekV3RMSNorm(hidden_dim)
            self.attention = MultiHeadLatentAttention(
                hidden_dim=hidden_dim,
                n_heads=config["num_heads"],
                qk_nope_head_dim=config["qk_nope_head_dim"],
                qk_rope_head_dim=config["qk_rope_head_dim"],
                kv_compression_dim=config["kv_compression_dim"],
                query_compression_dim=config["query_compression_dim"],
                v_head_dim=config["v_head_dim"],
                dropout=dropout_rate,
                bias=False,
                latent_layernorm=True,
            )

        self.ffn_norm = DeepseekV3RMSNorm(hidden_dim)
        intermediate_dim = config.get("intermediate_dim", hidden_dim * 4)
        self.ffn = ExpertFFN(hidden_dim, intermediate_dim, dropout_rate=dropout_rate)

        # Per-depth norm before shared output head
        self.output_norm = DeepseekV3RMSNorm(hidden_dim)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        token_emb: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            prev_hidden: [B, T', D] hidden states from previous depth (or main model)
            token_emb:   [B, T', D] embeddings of shifted tokens
            position_embeddings: (cos, sin) for RoPE
            attention_mask: optional mask [B, T']

        Returns:
            hidden: [B, T', D] transformed hidden states for this depth
        """
        # Concatenate and project
        combined = torch.cat([prev_hidden, token_emb], dim=-1)  # [B, T', 2D]
        h = self.proj_norm(self.proj(combined))                 # [B, T', D]

        # Transformer block
        if self.use_attention:
            attn_input = self.attn_norm(h)
            attn_out, _, _ = self.attention(
                attn_input, position_embeddings, attention_mask=attention_mask,
            )
            h = h + attn_out

        ffn_input = self.ffn_norm(h)
        h = h + self.ffn(ffn_input)

        return h


class MultiTokenPrediction(nn.Module):
    """
    Multi-Token Prediction module (DeepSeek-V3).

    Predicts D additional future tokens using sequential depth blocks.
    Shares the main model's embedding layer and output head (weight tying).

    For depth k = 1..D:
      - Takes hidden from depth k-1 (main model output for k=1)
      - Combines with embedding of token at position t+k
      - Processes through an MTPBlock (configurable: full MLA+FFN or lite FFN-only)
      - Produces logits for token at position t+k+1
    """

    def __init__(
        self,
        config: dict,
        shared_embedding: nn.Embedding,
        shared_output_head: nn.Linear,
    ):
        super().__init__()
        self.depth = config.get("mtp_depth", 1)
        self.shared_embedding = shared_embedding
        self.shared_output_head = shared_output_head

        use_attention = config.get("mtp_use_attention", True)

        self.blocks = nn.ModuleList([
            MTPBlock(config, use_attention=use_attention)
            for _ in range(self.depth)
        ])

    def forward(
        self,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Args:
            main_hidden: [B, T, D] output of the main transformer (after final norm)
            input_ids:   [B, T] original input token IDs
            position_embeddings: (cos, sin) each [1, 1, T, D_rope]
            attention_mask: optional [B, T]

        Returns:
            List of D logit tensors. logits[k] has shape [B, T-k-1, vocab_size]
            and predicts tokens at positions [k+1, k+2, ..., T-1].
        """
        cos, sin = position_embeddings
        B, T, D = main_hidden.shape
        all_logits = []

        prev_hidden = main_hidden

        for k, block in enumerate(self.blocks):
            # For depth k (0-indexed), we need:
            # - prev_hidden at positions [0, ..., T-k-2]  -> drop last k+1
            # - embeddings of tokens at positions [k+1, ..., T-1]  -> shifted by k+1
            # - predicts tokens at positions [k+2, ..., T]

            # Effective sequence length for this depth
            t_eff = T - k - 1
            if t_eff <= 0:
                break

            h_in = prev_hidden[:, :t_eff, :]                         # [B, t_eff, D]
            token_emb = self.shared_embedding(input_ids[:, k + 1:k + 1 + t_eff])  # [B, t_eff, D]

            # Slice position embeddings for the truncated sequence
            cos_k = cos[:, :, :t_eff, :]
            sin_k = sin[:, :, :t_eff, :]
            pos_emb_k = (cos_k, sin_k)

            mask_k = attention_mask[:, :t_eff] if attention_mask is not None else None

            h_out = block(h_in, token_emb, pos_emb_k, mask_k)        # [B, t_eff, D]

            logits_k = self.shared_output_head(block.output_norm(h_out))  # [B, t_eff, V]
            all_logits.append(logits_k)

            prev_hidden = h_out

        return all_logits

    @torch.no_grad()
    def draft_forward(
        self,
        main_hidden: torch.Tensor,
        main_logits: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Generate MTP draft predictions for speculative decoding.
        Uses greedy argmax from the previous depth's logits as the draft token.

        Args:
            main_hidden: [B, T, D] hidden states from main transformer
            main_logits: [B, T, V] logits from main output head
            position_embeddings: (cos, sin) each [1, 1, T, D_rope]

        Returns:
            List of D logit tensors, each [B, 1, V] for the last position.
        """
        cos, sin = position_embeddings
        all_logits = []
        prev_hidden = main_hidden[:, -1:, :]  # [B, 1, D]
        prev_logits = main_logits

        for block in self.blocks:
            # Greedy draft token from previous depth
            draft_token = prev_logits[:, -1:, :].argmax(dim=-1)  # [B, 1]
            token_emb = self.shared_embedding(draft_token)        # [B, 1, D]

            # Use last position's cos/sin for the single draft position
            cos_k = cos[:, :, -1:, :]
            sin_k = sin[:, :, -1:, :]
            pos_emb_k = (cos_k, sin_k)

            h_out = block(prev_hidden, token_emb, pos_emb_k, attention_mask=None)
            logits_k = self.shared_output_head(block.output_norm(h_out))  # [B, 1, V]
            all_logits.append(logits_k)

            prev_hidden = h_out
            prev_logits = logits_k

        return all_logits
