import torch
import torch.nn as nn
from src.midiR1.model.MLA import MultiHeadLatentAttention
from src.midiR1.model.moe import MoE
from src.midiR1.model.expert import ExpertFFN
from src.midiR1.model.mtp import MultiTokenPrediction
from src.midiR1.utils.rms_norm import DeepseekV3RMSNorm
from src.midiR1.utils.rotary_embedding import RotaryEmbedding
from typing import Optional, Tuple, Union


class MidiR1(nn.Module):
    """A Mixture-of-Experts Transformer with Multi-Token Prediction (DeepSeek-V3)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        hidden_dim = config["hidden_dim"]
        dropout_rate = config.get("dropout_rate", 0.1)

        # Token embedding
        self.embedding = nn.Embedding(config["vocab_size"], hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # Rotary Position Embedding (shared across all layers)
        self.rotary_emb = RotaryEmbedding(
            dim=config["qk_rope_head_dim"],
            max_seq_len=config["max_seq_len"],
        )

        # Transformer layers
        first_k_dense = config.get("first_k_dense_replace", 1)
        intermediate_dim = config.get(
            "intermediate_dim", self._compute_swiglu_intermediate(hidden_dim)
        )
        aux_loss_alpha = config.get("moe_aux_loss_alpha", 1e-5)

        layers = []
        for layer_idx in range(config["num_layers"]):
            layer_dict = {
                "attn_norm": DeepseekV3RMSNorm(hidden_dim),
                "attention": MultiHeadLatentAttention(
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
                ),
                "ffn_norm": DeepseekV3RMSNorm(hidden_dim)
            }

            if layer_idx < first_k_dense:
                # Dense FFN for first layer(s)
                layer_dict["ffn"] = ExpertFFN(
                    hidden_dim, intermediate_dim, dropout_rate=dropout_rate
                )
            else:
                # MoE for remaining layers
                layer_dict["ffn"] = MoE(
                    hidden_dim=hidden_dim,
                    num_routed_experts=config.get("num_routed_experts", 6),
                    num_shared_experts=config.get("num_shared_experts", 2),
                    segmentation_factor=config.get("segmentation_factor", 4),
                    intermediate_dim=intermediate_dim,
                    top_k=config["activated_experts"],
                    dropout_rate=dropout_rate,
                    aux_loss_alpha=aux_loss_alpha,
                )

            layers.append(nn.ModuleDict(layer_dict))

        self.layers = nn.ModuleList(layers)

        # Output
        self.final_norm = DeepseekV3RMSNorm(hidden_dim)
        self.final_dropout = nn.Dropout(dropout_rate)
        self.output_head = nn.Linear(hidden_dim, config["vocab_size"])

        # Multi-Token Prediction (shared embedding + output head)
        mtp_depth = config.get("mtp_depth", 1)
        self.mtp_depth = mtp_depth
        if mtp_depth > 0:
            self.mtp = MultiTokenPrediction(
                config=config,
                shared_embedding=self.embedding,
                shared_output_head=self.output_head,
            )
        else:
            self.mtp = None

    @staticmethod
    def _compute_swiglu_intermediate(hidden_dim: int) -> int:
        """Fallback: SwiGLU intermediate = (8/3)*hidden rounded to nearest 256."""
        raw = int(hidden_dim * 8 / 3)
        return ((raw + 255) // 256) * 256

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        use_mtp_drafts: bool = False,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]]:

        batch_size, seq_len = input_ids.shape

        x = self.embedding(input_ids)
        x = self.embedding_dropout(x)

        # Compute RoPE cos/sin with offset for cached positions
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0].c_kv.shape[1]

        cos, sin = self.rotary_emb.get_cos_sin(
            seq_len=seq_len,
            device=input_ids.device,
            dtype=x.dtype,
            offset=past_len,
        )
        position_embeddings = (cos, sin)

        # Process through transformer layers, collecting MoE aux losses
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        new_key_values = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None and layer_idx < len(past_key_values):
                past_kv = past_key_values[layer_idx]

            # Pre-norm attention with residual
            attn_input = layer["attn_norm"](x)
            attn_output, _, new_cache = layer["attention"](
                attn_input,
                position_embeddings,
                attention_mask=attention_mask,
                past_kv=past_kv,
                use_cache=use_cache,
            )
            x = x + attn_output

            if use_cache:
                new_key_values.append(new_cache)

            # Pre-norm FFN/MoE with residual
            ffn_input = layer["ffn_norm"](x)
            if isinstance(layer["ffn"], MoE):
                ffn_output, aux_loss = layer["ffn"](ffn_input)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                ffn_output = layer["ffn"](ffn_input)
            x = x + ffn_output

        # Final norm and output
        x = self.final_norm(x)
        x = self.final_dropout(x)
        logits = self.output_head(x)

        # MTP during training (with target_ids)
        if self.mtp is not None and self.training and target_ids is not None:
            mtp_logits = self.mtp(x, input_ids, position_embeddings, attention_mask)
            return logits, mtp_logits, total_aux_loss

        # MTP draft mode for speculative inference
        if self.mtp is not None and use_mtp_drafts:
            mtp_draft_logits = self.mtp.draft_forward(x, logits, position_embeddings)
            if use_cache:
                return logits, mtp_draft_logits, new_key_values
            return logits, mtp_draft_logits

        if use_cache:
            return logits, new_key_values

        return logits
