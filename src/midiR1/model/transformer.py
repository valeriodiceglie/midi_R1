import torch
from torch import nn
from midiR1.model.embeddings import Embeddings
from midiR1.model.transformer_block import DenseBlock, MoEBlock


class R1Transformer(nn.Module):
    def __init__(
            self,
            n_layers:int,
            moe_experts:int,
            top_moe_k:int, 
            vocab_size:int, 
            hidden_dim:int,
            latent_dim:int, 
            rotary_dim:int,
            base_seq_len:int,
            stage1_seq_len:int,
            max_seq_len:int, 
            dropout_rate:float):
        super().__init__()
        self.d = hidden_dim
        self.embeddings = Embeddings(vocab_size, hidden_dim, rotary_dim, dropout_rate, base_seq_len, stage1_seq_len, max_seq_len)
        # construct blocks
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i < 3: # TODO -> add to configuration the Number of transformer blocks
                self.layers.append(DenseBlock(hidden_dim, latent_dim, dropout_rate))
            else:
                if moe_experts > 0:
                    self.layers.append(MoEBlock(hidden_dim, latent_dim, moe_experts, top_moe_k))
                else:
                    self.layers.append(DenseBlock(hidden_dim, latent_dim, dropout_rate))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids:torch.Tensor, key_cache=None, value_cache=None, labels=None):
        x, freqs = self.embeddings(input_ids)
        new_key_cache = []
        new_value_cache = []
        for block, k, v in zip(self.layers, key_cache or [], value_cache or []):
            x, k, v = block(x, freqs, k, v)
            new_key_cache.append(k)
            new_value_cache.append(v)
        x = self.norm(x)
         
        if labels is None:
            # inference: return last‐token logits + full caches
            return x, new_key_cache, new_value_cache
        else:
            # training: return token‐wise hidden for PredictionHeads
            return x
