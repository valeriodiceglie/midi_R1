import torch
from torch import nn
from midiR1.model.transformer import R1Transformer
from midiR1.model.prediction_head import PredictionHead


class R1Model(nn.Module):
    def __init__(self,
                n_layers:int,
                moe_experts:int,
                top_moe_k:int,
                vocab_size:int,
                n_heads:int,
                mtp_num_heads:int,
                hidden_dim:int,
                latent_dim:int,
                rotary_dim:int,
                base_seq_len:int,
                stage1_seq_len:int,
                max_seq_len:int,
                dropout:float
                ):
        super().__init__()
        self.transformer = R1Transformer(n_layers, moe_experts, top_moe_k, vocab_size, hidden_dim, latent_dim, rotary_dim, base_seq_len, stage1_seq_len, max_seq_len, dropout)
        # MTP heads
        self.mtp_heads = nn.ModuleList([
            PredictionHead(vocab_size, hidden_dim, n_heads, dropout) for _ in range(mtp_num_heads)
        ])

    def forward(self, input_ids:torch.Tensor, key_cache=None, value_cache=None, labels:torch.Tensor=None):
        # input_ids: (batch, seq)
        if labels is None:
            hidden, key_cache, value_cache = self.transformer(input_ids, key_cache, value_cache, labels)
        else:
            hidden = self.transformer(input_ids, key_cache, value_cache, labels)
        embeds = self.transformer.embeddings.token_embed(input_ids)
        # sequential multi-token prediction
        current_hidden = hidden
        all_logits = []
        for head in self.mtp_heads:
            current_hidden, logit = head(current_hidden, embeds)
            all_logits.append(logit)
        # During inference, return only the last head's last token logits
        if labels is None:
            return all_logits[-1][:, -1, :], key_cache, value_cache
        else:
            # compute losses during training
            loss_fn = nn.CrossEntropyLoss()
            losses = [
                loss_fn(l.view(-1, l.size(-1)), labels.view(-1))
                for l in all_logits
            ]
            return sum(losses) / len(losses)
