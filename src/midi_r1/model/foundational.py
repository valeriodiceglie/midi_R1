import torch
import torch.nn as nn
from midi_r1.config import Config

class R1FoundationBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.d = cfg.model.hidden_size
        self.h =  cfg.model.num_heads
        self.f = cfg.model.ffn_dim 
        self.l = cfg.model.latent_dim
        self.n = cfg.model.num_latents
        
        # standard self-attention on tokens
        self.self_attn = nn.MultiheadAttention(self.d, self.h, batch_first=True)
        # latents
        self.latents = nn.Parameter(torch.randn(self.n, self.l))
        # projections
        self.q_latent = nn.Linear(self.l, self.l)
        self.k_token = nn.Linear(self.d, self.l)
        self.v_token = nn.Linear(self.d, self.l)
        self.q_token = nn.Linear(self.d, self.d)
        self.k_latent = nn.Linear(self.l, self.d)
        self.v_latent = nn.Linear(self.l, self.d)
        # feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(self.d, self.f),
            nn.GELU(),
            nn.Linear(self.f, self.d)
        )
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)
        self.norm3 = nn.LayerNorm(self.d)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        # 1. Self-attention
        res = x
        x_attn, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=~key_padding_mask)
        x = self.norm1(res + x_attn)

        # 2. Latent attention
        B, L, d = x.size()
        k = self.k_token(x).view(B, L, -1)
        v = self.v_token(x).view(B, L, -1)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)
        q_lat = self.q_latent(lat)
        attn_w = torch.softmax(torch.matmul(q_lat, k.transpose(-2, -1)) / (self.l ** 0.5), dim=-1)
        lat_upd = torch.matmul(attn_w, v)
        q_tok = self.q_token(x)
        k_lat = self.k_latent(lat_upd)
        v_lat = self.v_latent(lat_upd)
        attn_w2 = torch.softmax(torch.matmul(q_tok, k_lat.transpose(-2, -1)) / (d ** 0.5), dim=-1)
        x_lat = torch.matmul(attn_w2, v_lat)
        x = self.norm2(x + x_lat)

        # 3. Feed-forward
        res2 = x
        x_ffn = self.ffn(x)
        x = self.norm3(res2 + x_ffn)
        return x


class R1FoundationModel(nn.Module):
    def __init__(self, cfg:Config, vocab_size):
        super().__init__()
        self.d = cfg.model.hidden_size
        self.h =  cfg.model.num_heads
        self.f = cfg.model.ffn_dim 
        self.l = cfg.model.latent_dim
        self.n = cfg.model.num_latents
        self.seq_len = cfg.data.seq_len
        self.n_layers = cfg.model.num_layers
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(self.vocab_size, self.d)
        self.pos_emb = nn.Embedding(self.seq_len, self.d)
        self.blocks = nn.ModuleList([R1FoundationBlock(cfg) for _ in range(self.n_layers)])
        self.ln_f = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        B, L = input_ids.size()
        causal = torch.triu(torch.ones(L, L, device=input_ids.device), diagonal=1).bool()
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=attention_mask)
        x = self.ln_f(x)
        return self.head(x)