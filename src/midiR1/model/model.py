import torch
import torch.nn as nn
from src.midiR1.model.MLA import MultiHeadLatentAttention
from src.midiR1.model.moe import MoE
from src.midiR1.model.mtp import MultiTokenPrediction
from typing import Optional, Union, Tuple


class MidiR1(nn.Module):
    """A Mixture-of-Experts Transformer with Multi-Token Prediction."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Extract dropout rate from config or use default
        dropout_rate = config.get("dropout_rate", 0.1)

        self.embedding = nn.Embedding(config["vocab_size"], config["hidden_dim"])
        # Add embedding dropout
        self.embedding_dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn_norm": nn.RMSNorm(config["hidden_dim"]),
                "attention": MultiHeadLatentAttention(
                    hidden_dim=config["hidden_dim"],
                    num_heads=config["num_heads"],
                    head_dim=config["head_dim"],
                    kv_compression_dim=config["kv_compression_dim"],
                    query_compression_dim=config["query_compression_dim"],
                    rope_dim=config["rope_dim"],
                    dropout_rate=dropout_rate
                ),
                "moe_norm": nn.RMSNorm(config["hidden_dim"]),
                "moe": MoE(
                    hidden_dim=config["hidden_dim"],
                    num_experts=config["num_experts"],
                    top_k=config["activated_experts"],
                    dropout_rate=dropout_rate
                )
            }) for _ in range(config["num_layers"])
        ])

        self.final_norm = nn.LayerNorm(config["hidden_dim"])
        # Add final dropout
        self.final_dropout = nn.Dropout(dropout_rate)

        self.output_head = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.mtp = MultiTokenPrediction(
            config["hidden_dim"],
            config["vocab_size"],
            depth=1,
            dropout_rate=dropout_rate
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                target_ids: Optional[torch.Tensor] = None) -> Union[torch.Tensor| Tuple[torch.Tensor, torch.Tensor]]:
        # Embedding layer
        x = self.embedding(input_ids)
        # Apply embedding dropout
        x = self.embedding_dropout(x)

        # Process through transformer layers
        for layer in self.layers:
            # Attention block
            attn_input = layer["attn_norm"](x)
            # Center and shift normalization
            attn_input = attn_input - attn_input.mean(dim=-1, keepdim=True) + 1.0
            attn_output = layer["attention"](attn_input, attention_mask)
            x = x + attn_output

            # MoE block
            moe_input = layer["moe_norm"](x)
            moe_output = layer["moe"](moe_input)
            x = x + moe_output

        # Final normalization
        x = self.final_norm(x)
        # Apply final dropout
        x = self.final_dropout(x)

        # Main logits from final hidden state
        logits = self.output_head(x)

        # During training or when explicitly requested, also compute MTP predictions
        if (self.training and target_ids is not None) or not self.training:
            # Get multi-token predictions
            mtp_outputs = self.mtp(x)
            return logits, mtp_outputs

        return logits